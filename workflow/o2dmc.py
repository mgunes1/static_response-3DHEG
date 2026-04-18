#!/usr/bin/env python3
# from: tbeg/016-pfl-fwc/a_pfl/workflow
# then: bwc/002-spdmc/ffl/workflow
# then: bwc/005-mag/b_cta/common
# then: tbeg/025-p60-fm/b_cta/common
# then: tbeg/035-ni-exx/common
import os
from copy import deepcopy
import numpy as np
from qharv.seed import xml
from qharv.inspect import axes_pos

def find_opt_xml(doc):
  proj = doc.find(".//project")
  myid = proj.get("id")
  i0 = int(proj.get("series"))
  loop = doc.find(".//loop")
  nopt = int(loop.get("max"))
  iopt = i0+nopt-1
  # filename
  flast = '.'.join([myid, 's%03d' % iopt, 'opt', 'xml'])
  return flast

def read_opt(fopt, update_wavefunction=True):
  doc = xml.read(fopt)
  if update_wavefunction:
    # delete old jastrows
    wf = doc.find('.//wavefunction')
    # transfer new jastrows
    flast = find_opt_xml(doc)
    path = os.path.dirname(fopt)
    floc = os.path.join(path, flast)
    doc1 = xml.read(floc)
    wf1 = doc1.find('.//wavefunction')
    vp = wf1.find('.//override_variational_parameters')
    if vp is not None:
      xml.remove(vp)
    xml.swap_node(wf, wf1)
  return doc

def default_dmc():
  text = '''<root><qmc method="vmc" move="pbyp" checkpoint="-1">
    <parameter name="blocks_between_recompute"> 10 </parameter>
    <parameter name="blocks">        400 </parameter>
    <parameter name="warmupsteps">   200 </parameter>
    <parameter name="timestep">     25.0 </parameter>
    <parameter name="subSteps">        5 </parameter>
    <parameter name="steps">          20 </parameter>
    <parameter name="samples"> 2304 </parameter>
  </qmc>
  <qmc method="dmc" move="pbyp" checkpoint="-1">
    <parameter name="blocks_between_recompute"> 10 </parameter>
    <parameter name="blocks">        400 </parameter>
    <parameter name="warmupsteps">   200 </parameter>
    <parameter name="timestep">      3.0 </parameter>
    <parameter name="subSteps">        5 </parameter>
    <parameter name="steps">         200 </parameter>
    <parameter name="target_walkers"> 2304 </parameter>
  </qmc>
  <qmc method="dmc" move="pbyp" checkpoint="-1">
    <parameter name="blocks_between_recompute"> 10 </parameter>
    <parameter name="blocks">        400 </parameter>
    <parameter name="warmupsteps">   200 </parameter>
    <parameter name="timestep">      3.0 </parameter>
    <parameter name="subSteps">        5 </parameter>
    <parameter name="steps">         400 </parameter>
    <parameter name="target_walkers"> 2304 </parameter>
  </qmc></root>'''
  return xml.parse(text)

def read_nspin_rs(fopt_inp, ndim=2, nlayer=1):
  from qharv.seed import xml
  doc = xml.read(fopt_inp)
  axes = xml.get_axes(doc)
  nelecs = xml.get_nelecs(doc)
  nelec = sum(nelecs.values())
  rs = axes_pos.rs(axes[:ndim, :ndim], nelec)
  nspin = len(nelecs)
  nspin_per_layer = int(round(nspin/nlayer))
  if nspin_per_layer*nlayer != nspin:
    msg = 'wrong nspin %d' % nspin
    raise RuntimeError(msg)
  return nspin_per_layer, rs, nelec

def read_ndim(fopt):
  doc = xml.read(fopt)
  sc = doc.find('.//simulationcell')
  handler = xml.get_param(sc, 'LR_handler')
  ndim = 3
  if (handler is not None) and ('2d' in handler.lower()):
    ndim = 2
  return ndim

def add_sublat(doc, ndim=2, verbose=False):
  from qharv.seed import qmcpack_in
  axes = xml.get_axes(doc)
  pos = xml.get_pos(doc)
  if type(pos) is dict:
    pos = np.concatenate([p1 for p1 in pos.values()], axis=0)

  # calculate A -> B sublattice shift
  rs = axes_pos.rs(axes[:ndim, :ndim], len(pos))
  am = (2*np.pi/3**0.5)**0.5*rs
  axes0 = am*np.array([
    [1, 0],
    [-0.5, 3**0.5/2],
  ])
  frac = [1./3, 2./3]
  bvec = np.dot(frac, axes0)

  if verbose:
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, aspect=1)
    crystal.draw_cell(ax, axes0, c='k', alpha=1)

    ax.plot(*bvec.T, ls='', marker='.')
    plt.show()

  # add sublat estimator
  source = 'hon'
  est = xml.make_node('estimator', {'type': 'sublat', 'name': 'sublat', 'source': source})
  ham = doc.find('.//hamiltonian')
  ham.append(est)

  pos1 = np.c_[pos[:, :ndim] +bvec, np.zeros(len(pos))]
  grp0 = qmcpack_in.particle_group_from_pos(pos, 'H0', 0)
  grp1 = qmcpack_in.particle_group_from_pos(pos1, 'H1', 0)
  pset = xml.make_node('particleset', {'name': source})
  pset.append(grp0)
  pset.append(grp1)

  qsys = doc.find('.//qmcsystem')
  qsys.insert(2, pset)

def main():
  from argparse import ArgumentParser
  parser = ArgumentParser()
  parser.add_argument('--fopt', '-i', type=str)
  parser.add_argument('--fout', '-o', type=str, default="dmc.xml")
  parser.add_argument('--tx', type=float)
  parser.add_argument('--ty', type=float)
  parser.add_argument('--tz', type=float)
  parser.add_argument('--twistnum', '-itw', type=int)
  parser.add_argument('--nwalker', '-nw', type=int, default=2304)
  parser.add_argument('--checkpoint', '-chk', type=int)
  parser.add_argument('--timestep', '-ts', type=float)
  parser.add_argument('--tproj', '-p', type=float)
  parser.add_argument('--dn', '-dn', type=int, default=0)
  parser.add_argument('--ndmc', '-n', type=int, default=2)
  parser.add_argument('--nlayer', type=int, default=1)
  parser.add_argument('--spin_mass', type=float, default=500.0)
  parser.add_argument('--est_sk', '-sk', action='store_true')
  parser.add_argument('--est_ssk', '-ssk', action='store_true')
  parser.add_argument('--est_gr', '-gr', action='store_true')
  parser.add_argument('--est_eigr', '-eigr', action='store_true')
  parser.add_argument('--est_nk', '-nk', action='store_true')
  parser.add_argument('--est_sdens', '-sdens', type=int)
  parser.add_argument('--est_cdens', '-cdens', type=int)
  parser.add_argument('--est_mdens', '-mdens', type=int)
  parser.add_argument('--est_vgr', '-vgr', type=int)
  parser.add_argument('--est_cpol', '-cpol', action='store_true')
  parser.add_argument('--est_latdev', '-latdev', action='store_true')
  parser.add_argument('--est_sublat', '-sublat', action='store_true')
  parser.add_argument('--record_configs', '-rc', type=int)
  parser.add_argument('--verbose', '-v', action='store_true')
  parser.add_argument('--warm', '-w', action='store_true')
  args = parser.parse_args()
  fopt = args.fopt
  fout = args.fout
  nw = args.nwalker
  timestep = args.timestep
  tproj = args.tproj

  # default times should depend on density and system size
  ndim = read_ndim(args.fopt)
  nspin, rs, nelec = read_nspin_rs(args.fopt, ndim=ndim, nlayer=args.nlayer)
  if tproj is None:
    tpmult = 15.625
    tproj = tpmult*nelec*rs
  if timestep is None:
    timestep = rs/20
    if args.warm:
      timestep = rs/50

  # step 1: change <wavefunction>
  doc = read_opt(args.fopt)
  if args.tx is not None:
    # read input twist
    tx = args.tx
    ty = args.ty
    tz = args.tz
    if ty is None:
      ty = 0
    if tz is None:
      tz = 0
    twist = np.array([tx, ty, tz])
    tt = ' '.join(twist.astype(str))
    # change twist
    bb = doc.find('.//sposet_builder')
    for spo in bb.findall('sposet'):
      spo.set('twist', tt)
  if args.twistnum is not None:
    for node in doc.findall('.//sposet_builder'):
      node.set('twistnum', str(args.twistnum))
  lbf = doc.find('.//backflow') is not None

  # is this a noncollinear run?
  pset = doc.find('.//particleset[@name="e"]')
  spinor = pset.get('spinor')
  lspinor = False if spinor is None else spinor == 'yes'

  # step 2: change <hamiltonian>
  ham = doc.find('.//hamiltonian')
  if args.est_nk:
    est = xml.make_node('estimator', {'type': 'momentum', 'name': 'nofk'})
    ham.append(est)
  if args.est_sk:
    est = xml.make_node('estimator', {'type': 'skall', 'name': 'skall', 'target': 'e', 'source': 'e', 'hdf5': 'yes'})
    ham.append(est)
    # change <LR_dim_cutoff> for S(k)
    axes = xml.get_axes(doc)[:ndim, :ndim]
    rc = axes_pos.rwsc(axes)
    if ndim == 2:
      kf = 2.0/rs/nspin**0.5
    elif ndim == 3:
      from chiesa_correction import heg_kfermi
      kf = heg_kfermi(rs, ndim=3)*2**0.5/nspin**0.5
    rckc = max(4*kf*rc, 30)
    sc = doc.find('.//simulationcell')
    xml.set_param(sc, 'LR_dim_cutoff', str(rckc), force=True)
  if args.est_ssk:
    est = xml.make_node('estimator', {'type': 'sskall', 'name': 'sskall'})
    ham.append(est)
  if args.est_gr:
    est = xml.make_node('estimator', {'type': 'gofr', 'name': 'gofr', 'num_bin': '200'})
    ham.append(est)
  if args.est_eigr:
    est = xml.make_node('estimator', {'type': 'gofr', 'name': 'grei', 'num_bin': '200', 'source': 'ion0'})
    ham.append(est)
  ldens = False
  if args.est_sdens is not None:
    ldens = True
    nx = ny = args.est_sdens
    est = xml.make_node('estimator', {'type': 'spindensity', 'name': 'spindensity', 'report': 'yes'})
    mesh = np.array([nx, ny, 1])  # !!!! HACK hard-code 2D isotropic mesh
    xml.set_param(est, 'grid', ' '.join(mesh.astype(str)), new=True)
    ham.append(est)
  if args.est_cdens is not None:
    ldens = True
    nx = ny = args.est_cdens
    mesh = np.array([nx, ny, 2])
    deltas = 1./mesh
    est = xml.make_node('estimator', {'type': 'density', 'name': 'density', 'delta': xml.arr2text(deltas)})
    ham.append(est)
  if args.est_mdens is not None:
    ldens = True
    nx = ny = args.est_mdens
    mesh = np.array([nx, ny, 1])
    est = xml.make_node('estimator', {'type': 'magdensity', 'name': 'magdensity'})
    xml.set_param(est, 'grid', ' '.join(mesh.astype(str)), new=True)
    xml.set_param(est, 'integrator', 'simpson', new=True)
    xml.set_param(est, 'samples', '9', new=True)
    ham.append(est)
  if args.est_vgr is not None:
    nx = ny = args.est_vgr
    mesh = np.array([nx, ny])
    est = xml.make_node('estimator', {'type': 'vecgofr', 'name': 'vecgofr'})
    xml.set_param(est, 'grid', ' '.join(mesh.astype(str)), new=True)
    ham.append(est)
  if args.est_cpol:
    names = ['cpol', 'cpb1', 'cpb2']
    axl = ['2 -1', '1 0', '0 1']
    for name, axis in zip(names, axl):
      est = xml.make_node('estimator', {'type': 'cpol', 'name': name, 'gvec': axis})
      ham.append(est)
  if args.est_latdev:  # !!!! HACK: assume ion is H
    name = 'ion0'
    ions = doc.find('.//particleset[@name="%s"]' % name)
    grps = ions.findall('.//group')
    if len(grps) != 1:
      if lspinor:  # concat all groups into one
        from qharv.seed import qmcpack_in
        posl = []
        for grp in grps:
          pos1 = xml.get_group_pos(grp)
          posl.append(pos1)
        grp0 = grps[0]
        pos = np.concatenate(posl, axis=0)
        new_grp = qmcpack_in.particle_group_from_pos(pos, grp0.get('name'), grp0.get('charge'))
        # add ion1 after ion0
        name = 'ion1'
        new_ions = xml.make_node('particleset', {'name': name})
        new_ions.append(new_grp)
        parent = ions.getparent()
        i = parent.index(ions)
        parent.insert(i+1, new_ions)
      else:
        msg = 'latdev for %d groups' % len(grps)
        raise RuntimeError(msg)
    est = xml.make_node('estimator', {'type': 'latticedeviation', 'name': 'latdev',
      'target': 'e', 'tgroup': 'u', 'source': name, 'sgroup': 'H',
      'hdf5': 'yes', 'per_xyz': 'yes', 'lsap': 'yes'})
    ham.append(est)
    if doc.find('.//group[@name="d"]') is not None:
      est = xml.make_node('estimator', {'type': 'latticedeviation', 'name': 'latdev1',
        'target': 'e', 'tgroup': 'd', 'source': 'ion0', 'sgroup': 'H1',
        'hdf5': 'yes', 'per_xyz': 'yes', 'lsap': 'yes'})
      ham.append(est)
  if args.est_sublat:
    add_sublat(doc)

  # step 3: change <qmc>
  # delete old qmc
  loop = doc.find('.//loop')
  qmc = loop.find('.//qmc')
  ts0 = float(xml.get_param(qmc, 'timestep'))
  nstep = int(xml.get_param(qmc, 'steps'))
  nsub = int(xml.get_param(qmc, 'subSteps'))
  #nblock = int(xml.get_param(qmc, 'blocks'))
  nblock = 400  # !!!! HACK set nblock
  move = 'pbyp'
  if lbf:
    move = 'wbyw'
  xml.remove(loop)
  # add new qmc
  qsim = doc.getroot()
  vdmc = default_dmc()
  nconf = args.record_configs
  if nconf is not None:
    for node in vdmc:
      xml.set_param(node, 'record_configs', '%d' % nconf, new=True)
      # !!!! HACK more configs from VMC
      if node.get('method') == 'vmc':
        xml.set_param(node, 'record_configs', '1')
  # edit template
  tsl = [ts0, timestep]
  idmc = 0
  for node, dt in zip(vdmc.findall('.//qmc'), tsl):
    method = node.get("method")
    node.set('move', move)
    xml.set_param(node, "timestep", str(dt))
    # determine number of steps to run
    xml.set_param(node, 'subSteps', nsub)
    xml.set_param(node, 'blocks', nblock)
    ns = int(round(tproj/nblock/nsub/dt))
    ns = max(ns, 1)
    xml.set_param(node, "steps", str(ns))
    if args.checkpoint is not None:
      node.set('checkpoint', str(args.checkpoint))
    if method == "vmc":
      xml.set_param(node, "samples", str(nw))
      smtext = '0.1'
      xml.set_param(node, 'steps', str(10))
    else:
      smtext = str(args.spin_mass)
      xml.set_param(node, "target_walkers", str(nw))
      idmc += 1
    if idmc > args.ndmc:
      xml.remove(node)
      continue
    if lspinor:
      try:
        xml.set_param(node, "spinMass", smtext, new=True)
      except RuntimeError:
        xml.set_param(node, "spinMass", smtext, new=False)
    qsim.append(node)
  # try more timesteps
  dt = timestep
  for idmc in range(args.ndmc-1):
    dt /= 2
    ns *= 2
    dmc = deepcopy(node)
    xml.set_param(dmc, "timestep", str(dt))
    xml.set_param(dmc, "steps", str(ns))
    if lspinor:
      try:
        xml.set_param(node, "spinMass", str(args.spin_mass), new=True)
      except RuntimeError:
        xml.set_param(node, "spinMass", str(args.spin_mass), new=False)
    qsim.append(dmc)
  if args.warm:  # prepend small-timestep vmc
    vmc = qsim.find('.//qmc[@method="vmc"]')
    ts = float(xml.get_param(vmc, 'timestep'))
    nblock = int(xml.get_param(vmc, 'blocks'))
    v0 = deepcopy(vmc)
    xml.set_param(v0, 'timestep', str(ts/10))
    xml.set_param(v0, 'blocks', str(nblock//2))
    idx = qsim.index(vmc)
    qsim.insert(idx, v0)

  # step 4 (optional): change particle number
  if args.dn != 0:  # always operate on "u" electrons
    dn = args.dn
    # a. particleset
    grp = xml.get_group(doc)
    nup = int(grp.get('size'))
    new = nup+dn
    grp.set('size', str(new))
    pos = xml.get_group_pos(grp)
    if dn < 0:
      pos = pos[:new]
    else:
      for j in range(dn):
        p1 = 0.5*(pos[j]+pos[j+1])
        pos = np.r_[pos, [p1]]
    xml.set_pos(grp, pos)
    if lspinor:  # edit spins
      spins = xml.get_spins(grp)
      if dn < 0:
        spins = spins[:new]
      else:
        for j in range(dn):
          spins = np.r_[spins, [2*np.pi*np.random.rand()]]
      xml.set_spins(grp, spins)
    # b. sposet
    bb = doc.find('.//sposet_builder[@type="bspline"]')
    sposet = bb.find('.//sposet')  # !!!! HACK assume first one is for u
    spo_name = sposet.get('name')
    sposet.set('size', str(new))
    # c. detset
    det = doc.find('.//determinant[@sposet="%s"]' % spo_name)
    n1 = det.get('size')
    if n1 is not None:
      det.set('size', str(new))

  # output
  xml.write(fout, doc)

if __name__ == '__main__':
  main()  # set no global variable
