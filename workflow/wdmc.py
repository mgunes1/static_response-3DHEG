#!/usr/bin/env python3
# from: tbeg/013-eha/a_tb10f/recon
# then: tbeg/020-sdw/common
# then: tbeg/025-p60-fm/b_cta/common
# then: tbeg/035-ni-exx/common/wdmc.py
import os
import numpy as np
from qharv.seed import wf_h5, xml, qmcpack_in, xml_examples
from qharv.cross import pwscf
from qharv.inspect import axes_pos, jas, grsk
from mch.eg2d.cell import make_sposet, make_detset

def tile_magnetic(axes, elem, pos, tmat):
  from qharv.inspect.axes_elem_pos import ase_tile
  lalias = ('H0' in elem) or ('H1' in elem)  # magnetic ions
  if lalias:  # map magnetic ions to difference elements
    elems = ['H', 'H0', 'H1', 'H2']
    alias = ['H', 'C', 'He', 'Li']
    elem_alias = dict()
    alias_elem = dict()
    for e, name in zip(elems, alias):
      elem_alias[e] = name
      alias_elem[name] = e
    elem1 = [elem_alias[e] for e in elem]
  else:
    elem1 = list(elem)
  axes, elem, pos = ase_tile(axes, elem1, pos, tmat)
  if lalias:  # map back to magnetic ions
    elem = [alias_elem[e] for e in elem]
  return axes, elem, pos

def simulationcell2d(axes):
  axes1 = axes.copy()
  rcut = axes_pos.rins(axes1[:2, :2])
  axes1[2, 2] = 2*rcut  # fake Lz
  sc = qmcpack_in.simulationcell_from_axes(axes1)
  xml.set_param(sc, "LR_dim_cutoff", str(30))
  xml.set_param(sc, "LR_handler", "ewald_strict2d", new=True)
  xml.set_param(sc, "bconds", "p p n")
  return sc

def solid_angle(ndim):
  return 2*(ndim-1)*np.pi/ndim

def heg_kfermi(rs, ndim=3, nspin=2):
  vol = solid_angle(ndim)*rs**ndim
  rho = 1./vol
  kf = (2*np.pi)*(rho/solid_angle(ndim)/nspin)**(1./ndim)
  return kf

def screened_rpa_jastrow(rs, nknot, rcut, ndim, ngrid=1024, kmin=1e-4, kcut_by_kf=20, nspin=2, screen_prefactor=1./3):
  # from: bwc/020-screen/a_rect/liq/workflow/wferro.py
  import chiesa_correction as chc
  # determine real-space grid
  delta = rcut/(nknot+1)
  # determine k-space grid
  kf = heg_kfermi(rs, ndim=ndim, nspin=nspin)
  kcut = kcut_by_kf*kf
  kgrid = np.linspace(kmin, kcut, ngrid)

  # screened RPA Jastrow
  uk = chc.gaskell_rpa_uk(kgrid, rs, kf, ndim=ndim)
  if screen_prefactor > 0:
    dgate = screen_prefactor*rcut
    uk = uk*np.tanh(kgrid*dgate)
  r1 = np.arange(delta/2, rcut-delta/4, delta/2)
  iftuk = grsk.ift_iso(r1, kgrid, uk, ndim=ndim)
  # fit to Bspline
  bsp = jas.BsplineFunction({'ncoef': nknot+4, 'grid_start': 0, 'delta_inv': 1./delta})
  ud_knots, cusp = jas.solve_for_knots(bsp, r1, iftuk)

  ## check Bspline
  #coeff = jas.coefficients_from_knots(ud_knots, -1, delta)
  #bsp_iftuk = [bsp({'coefs': coeff}, x) for x in r1]
  return ud_knots, cusp

def default_jastrow_knots(rs, rcut, ndim):
  nknot = int(round(np.ceil(rcut/(0.6*rs))))
  #nknot = max(14, int(round(np.ceil(rcut/(0.6*rs)))))   # was: no minimum
  cud, cusp = screened_rpa_jastrow(rs, nknot, rcut, ndim)
  return cud

def default_jastrow(species, rs, rcut, ndim):
  from qharv.inspect.jas import set_coeff
  nspin = len(species)
  cuu = cud = default_jastrow_knots(rs, rcut, ndim)
  nknot = len(cud)
  # template
  text = '''<jastrow name="J2" type="Two-Body" function="Bspline">
      <correlation speciesA="u" speciesB="u" size="%d">
        <coefficients id="uu" type="Array" optimize="yes">
        </coefficients>
      </correlation>
    </jastrow>''' % nknot
  jas = xml.parse(text)
  # set same-spin
  set_coeff(jas, 'uu', cuu)
  # set different
  for ispec in range(nspin):
    for jspec in range(ispec+1, nspin):
      spA = species[ispec]
      spB = species[jspec]
      myid = spA+spB
      corr = xml.make_node('correlation', {'speciesA': spA, 'speciesB': spB, 'size': str(nknot)})
      jas.append(corr)
      coeff = xml.make_node('coefficients', {'id': myid, 'type': 'Array', 'optimize': 'yes'})
      corr.append(coeff)
      set_coeff(corr, myid, cud)
  return jas

def default_vmc():
  text = '''<qmc method="vmc" move="pbyp" checkpoint="-1">
    <parameter name="blocks_between_recompute"> 10 </parameter>
    <parameter name="blocks">        100 </parameter>
    <parameter name="steps">          10 </parameter>
    <parameter name="warmupsteps">    40 </parameter>
    <parameter name="timestep">      5.0 </parameter>
    <parameter name="subSteps">        3 </parameter>
  </qmc>'''
  return xml.parse(text)

def extract_moire_params(inps):
  mydict = {}
  keys = ['amoire_in_ang', 'vmoire_in_mev', 'epsmoire',
    'pmoire_in_deg', 'mstar']
  for key in keys:
    if key in inps:
      mydict[key] = str(inps[key])
  return mydict

def extract_disorder_params(inps):
  if not 'ldisorder' in inps:
    return None
  if '.false.' in inps['ldisorder']:
    return None
  mydict = {}
  for key, val in inps.items():
    if 'disorder_' in key:
      val = float(val)
      if '_height' in key:
        val /= 2  # Ry -> ha
      mydict[key] = val
  return mydict

def extract_cospot_params(inps):
  if not 'lcospot' in inps:
    return None
  if '.false.' in inps['lcospot']:
    return None
  mydict = {}
  for key, val in inps.items():
    if ('cospot' in key) and (key != 'lcospot'):
      if key == 'vcospot':
        val = str(0.5*float(val))  # Ry->Ha
      mydict[key] = val
  return mydict

def cospot_ham(cospot_params):
  qvec = np.array([cospot_params['qcospot(%d)' % i] for i in range(1, 4)], dtype=float)
  attribs = {
    'type': 'cospot', 'name': 'cospot',
    'vq': cospot_params['vcospot'],
    'qvec': xml.arr2text(qvec),
    'potential': 'physical',
  }
  return xml.make_node('extpot', attribs)

def trim_jas(jas, species):
  for node in jas.findall('.//correlation'):
    for name in ['speciesA', 'speciesB']:
      sp = node.get(name)
      if sp not in species:
        xml.remove(node)
        break

def main():
  from argparse import ArgumentParser
  parser = ArgumentParser()
  parser.add_argument('feinsplein_h5')
  parser.add_argument('--fscf_inp', '-i')
  parser.add_argument('--fout', '-o', default='opt.xml')
  parser.add_argument('--nkx', type=int)
  parser.add_argument('--nky', type=int)
  parser.add_argument('--nkz', type=int, default=1)
  parser.add_argument('--nopt', type=int, default=8)
  parser.add_argument('--nsample', type=int, default=576*256)
  parser.add_argument('--dr_sig_in_rs', type=float, default=0.5)
  parser.add_argument('--spin_mass', type=float, default=0.1)
  parser.add_argument('--det', '-d', action='store_true',
    help='VMC using determinant wf to test converter')
  parser.add_argument('--no_moire', action='store_true')
  parser.add_argument('--no_disorder', action='store_true')
  parser.add_argument('--vcospot', '-vq', type=float)
  parser.add_argument('--quartic', action='store_true')
  parser.add_argument('--verbose', '-v', action='store_true')
  args = parser.parse_args()
  nkx = args.nkx
  nky = args.nky
  nkz = args.nkz
  lmoire = not args.no_moire
  ldisorder = not args.no_disorder
  inps = pwscf.input_keywords(args.fscf_inp)
  ntot_charge = 0
  if 'tot_charge' in inps:
    ntot_charge = int(inps['tot_charge'])
    print('tot_charge = %d' % ntot_charge)
  disorder_params = extract_disorder_params(inps)
  cospot_params = extract_cospot_params(inps)
  vcospot = args.vcospot  # allow user to override QE vcospot
  if (cospot_params is not None) and (vcospot is not None):
    cospot_params['vcospot'] = str(vcospot)
  if 'lmoire' in inps:
    if '.true.' in inps['lmoire'].lower():
      ndim = 2
  else:
    ndim = 3
  dgate = -1.0  # turn off gates
  if 'dgate' in inps:
    dgate = float(inps['dgate'])
  disorder_charge = 0
  if 'disorder_charge' in inps:
    disorder_charge = float(inps['disorder_charge'])
  keep_ei = abs(disorder_charge) > 1e-8

  if lmoire:
    moire_params = extract_moire_params(inps)
  if (nkx is None) or (nky is None):  # auto-detect supercell size
    with open(args.fscf_inp, 'r') as f:
      unit, kpts = pwscf.parse_kpoints(f.read())
    if unit != 'automatic':
      raise RuntimeError('need --nkx --nky, b/c unit=%s' % unit)
    nkx, nky, nkz = kpts['dims'][:3]
    assert nkz == 1
  tvec = [nkx, nky, nkz]
  source_name = 'ion0'  # !!!! HACK: hard code source name

  # read unit cell
  fp = wf_h5.read(args.feinsplein_h5)
  axes, elem, charges, pos = wf_h5.axes_elem_charges_pos(fp)
  nelecs = wf_h5.get(fp, 'nelecs')
  nelec = sum(nelecs)
  lspinor = False
  if 'has_spinors' in fp['electrons']:
    lspinor = fp['electrons']['has_spinors'][()]
  if lspinor:
    nspin = 1
  else:
    nspin = wf_h5.get(fp, 'nspin')
  lferro = (nspin == 2) and (nelecs[1] <= 0)
  fp.close()
  rs = axes_pos.rs(axes[:ndim, :ndim], nelec)
  dr_sig = args.dr_sig_in_rs * rs
  dt = rs

  # tile to supercell
  tmat = np.diag(tvec)
  axes1, elem1, pos1 = tile_magnetic(axes, elem, pos, tmat)
  elem1 = np.array(elem1)

  # make input
  qsys = qmcpack_in.bspline_qmcsystem(args.feinsplein_h5, tmat=tmat,
    run_dir = os.path.dirname(args.fout))
  # !!!! HACK: tmat combines magnetic ions

  # edit simulationcell
  sc = qsys.find(".//simulationcell")
  if ndim == 2:
    sc1 = simulationcell2d(axes1)
    if dgate > 0:  # dual-gate screening
      handler = 'ewald_screen2d' if dgate > rs else 'screen2d'
      xml.set_param(sc1, "LR_handler", handler)
      xml.set_param(sc1, "distance_to_gate", str(dgate), new=True)
  else:
    sc1 = qmcpack_in.simulationcell_from_axes(axes1)
  xml.swap_node(sc, sc1)

  # edit particleset
  spl = np.unique(elem)
  epset0 = qsys.find('.//particleset[@name="e"]')
  epset = xml.make_node('particleset', {'name': 'e'})
  xml.swap_node(epset0, epset)
  if lspinor:
    epset.set('spinor', 'yes')
  epset.set("random", "no")

  if lspinor or lferro:
    pl = [pos1]
  elif len(spl) > 1:
    pl = []
    for species in spl:
      sel = elem1 == species
      p1 = pos1[sel]
      pl.append(p1)
  elif len(spl) == 1:  # paramagnetic
    pl = [pos1[::2], pos1[1::2]]
  else:
    msg = '%d ionic species' % len(spl)
    raise NotImplementedError(msg)

  if args.verbose:  # check ions
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, aspect=1)
    for p1 in pl:
      ax.plot(*p1[:, :2].T, ls='', marker='o')
    from qharv.inspect import crystal
    crystal.draw_cell(ax, axes1[:2, :2])
    plt.show()

  # !!!! HACK use magnetic ions to initialize electrons
  if lspinor or lferro:
    epos = [np.concatenate(pl, axis=0)]
  else:
    epos = [p1 for p1 in pl if len(p1)>0]

  # !!!! HACK: random electrons
  if ntot_charge != 0:
    p1 = np.random.rand(nelecs[0], ndim)
    p2 = np.random.rand(nelecs[1], ndim)
    if ndim == 2:
      p1 = np.c_[p1, np.zeros(len(p1))]
      p2 = np.c_[p2, np.zeros(len(p2))]
    epos = [p1, p2]

  # initialize electrons
  ne_map = dict()
  for p1, ud in zip(epos, ['u', 'd']):
    n1 = len(p1)
    if n1 < 1: continue
    # add a group
    grp = xml.make_node('group', {'name': ud})
    xml.set_param(grp, 'charge', '-1', new=True)
    epset.append(grp)
    # put positions in group
    myp1 = p1.copy()
    myp1[:, :2] += dr_sig*np.random.randn(n1, 2)
    pa = qmcpack_in.pos_attrib(myp1)
    grp.append(pa)
    # complete group
    grp.set('size', str(n1))
    if lspinor:
      spins = 2*np.pi*np.random.rand(n1)
      xml.set_spins(grp, spins)
    ne_map[ud] = n1

  # edit hamiltonian
  ham = qsys.find(".//hamiltonian")
  if not keep_ei:  # !!!! remove all but e-e
    for node in ham:
      name = node.get("name")
      if name != "ElecElec":
        ham.remove(node)
  else:  # !!!! HACK: override ion charges
    ion0 = qsys.find('.//particleset[@name="%s"]' % source_name)
    for grp in ion0.findall('.//group'):
      xml.set_param(grp, 'charge', str(disorder_charge))

  #   add moire
  if not args.no_moire:
    vext = xml.make_node("extpot", dict(
      type='moire', name='moire', target='e', potential='physical', **moire_params
    ))
    ham.append(vext)

  #   add disorder
  if (not args.no_disorder) and (disorder_params is not None):
    from scipy.optimize import linear_sum_assignment
    # match atoms
    ainv = np.linalg.inv(axes1)
    with open(args.fscf_inp, 'r') as f:
      text = f.read()
    unit, data = pwscf.parse_atomic_positions(text, ndim=ndim)
    if unit == 'crystal':
      fracs = data['positions']
    elif unit == 'bohr':
      pos1 = data['positions']
      fracs = np.dot(pos1, ainv[:ndim, :ndim])
    else:
      msg = 'atomic positions unit = %s' % unit
      raise NotImplementedError(msg)
    pos0 = np.concatenate(pl, axis=0)
    fracs0 = np.dot(pos0, ainv)[:, :ndim]
    drij, rij = axes_pos.minimum_image_displacements(np.eye(ndim),
      fracs, fracs0)
    idx0, idx1 = linear_sum_assignment(rij)
    nelec = sum(nelecs)
    widths = np.ones(nelec)
    heights = np.zeros(nelec)
    for key, val in disorder_params.items():
      iat = int(key.split('(')[1].split(')')[0])
      jat = idx1[iat-1]
      if 'width' in key:
        widths[jat] = val
      if 'height' in key:
        heights[jat] = val
    dext = xml.make_node('extpot', dict(
      type='disorder', name='disorder', target='e', source=source_name, potential='physical'))
    for name, arr in zip(['widths', 'heights'], [widths, heights]):
      node = xml.make_node(name, text=xml.arr2text(arr))
      dext.append(node)
    ham.append(dext)
  if cospot_params is not None:
    ham.append(cospot_ham(cospot_params))

  # edit wf
  wf = qsys.find('.//wavefunction')
  bb = wf.find(".//sposet_builder")
  if lferro:  # trim sposet and determinant
    sposet = bb.find('.//sposet[@spindataset="1"]')
    if sposet is not None:
      spo_name = sposet.get('name')
      xml.remove(sposet)
      det = qsys.find('.//determinant[@sposet="%s"]' % spo_name)
      if det is not None:
        xml.remove(det)
  if not args.det:  # add jastrow
    rwsc = axes_pos.rwsc(axes1[:ndim, :ndim])
    names = ['u', 'd']
    eejas = default_jastrow(names, rs, rwsc, ndim)
    trim_jas(eejas, ne_map)
    if lspinor:  # set cusp
      node = eejas.find('.//correlation')
      cusp = -1./(ndim-1)
      node.set('cusp', str(cusp))
    if keep_ei:  # add e-I jastrow
      ei_rcut = rwsc
      ei_coeff = jas.get_coeff(eejas, 'uu')/5  # !!!! HACK: guess init
      ei_cusp = -disorder_charge/(ndim-1)
      nknot = len(ei_coeff)
      eijas = xml.make_node('jastrow', {'name': 'J1', 'type': 'One-Body', 'function': 'bspline', 'source': source_name, 'print': 'yes'})
      for name, charge in charges.items():
        corr_node = xml.make_node('correlation', {'elementType': name, 'cusp': str(ei_cusp), 'size': str(nknot), 'rcut': str(ei_rcut)})
        coef_node = xml.make_node('coefficients', {'id': 'e%s' % name, 'type': 'Array'})
        coef_node.text = xml.arr2text(ei_coeff)
        corr_node.append(coef_node)
        eijas.append(corr_node)
      wf.append(eijas)
    wf.append(eejas)

  # add qmc block
  calcs = []
  vmc = default_vmc()
  # edit timestep
  xml.set_param(vmc, "timestep", str(rs))
  if args.det:
    calcs.append(vmc)
  else:
    opt = xml_examples.pbyp_optimize()
    opt.set('max', str(args.nopt))
    qmc = opt.find(".//qmc")
    xml.set_param(qmc, "samples", str(args.nsample))
    if args.quartic:
      xml.set_param(qmc, "MinMethod", "quartic", new=True)
      qmc.append(xml.make_node('cost', {'name': 'energy'}, '0.95'))
      qmc.append(xml.make_node('cost', {'name': 'reweightedvariance'}, '0.05'))
      qmc.append(xml.make_node('cost', {'name': 'unreweightedvariance'}, '0.00'))
    for name in ['blocks', 'steps', 'subSteps', 'timestep']:
      param = xml.get_param(vmc, name)
      new = name == 'steps'  # !!! HACK
      xml.set_param(qmc, name, param, new=new)
    if lspinor:
      xml.set_param(qmc, "spinMass", str(args.spin_mass), new=True)
    calcs.append(opt)
  doc = qmcpack_in.assemble_project([qsys]+calcs)
  xml.write(args.fout, doc)

if __name__ == '__main__':
  main()  # set no global variable
