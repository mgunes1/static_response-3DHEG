#!/usr/bin/env python3
import numpy as np
from qharv.inspect import axes_pos
from qharv.cross import pwscf

def solid_angle(ndim):
  return 2*(ndim-1)*np.pi/ndim

def heg_kfermi(rs, ndim=3, nspin=2):
  vol = solid_angle(ndim)*rs**ndim
  rho = 1./vol
  kf = (2*np.pi)*(rho/solid_angle(ndim)/nspin)**(1./ndim)
  return kf

# electron_maxstep = 500
# mixing_mode = 'local-TF'

def default_scf_input(thr='1.0d-8'):
  text = f'''&control
 verbosity = 'high'
 outdir = 'qeout'
 disk_io = 'low'
 pseudo_dir = '/mnt/home/pyang/scratch/tbeg/pseudo'
/
&system
 ntyp = 1
 nosym = .true.
 noinv = .true.
 ibrav = 0
/
&electrons
 conv_thr = {thr}
 mixing_mode = 'local-TF'
 electron_maxstep = 300
/

ATOMIC_SPECIES
  H 1.0 H.upf
'''
  return text

def main():
  from argparse import ArgumentParser
  parser = ArgumentParser()
  parser.add_argument('--fout', '-o', type=str, default='scf.inp')
  parser.add_argument('--rs', type=float, default=5)
  parser.add_argument('--nelec', type=int, default=54)
  parser.add_argument('--func', type=str, default='ni')
  parser.add_argument('--thr', type=str, default='1.0d-8')
  parser.add_argument('--vq', type=float, default=0.05)
  parser.add_argument('--qx', type=int, default=1)
  parser.add_argument('--qy', type=int, default=2)
  parser.add_argument('--qz', type=int, default=3)
  parser.add_argument('--ndim', type=int, default=3)
  parser.add_argument('--ecut_pre', '-epre', type=float, default=125,
    help='prefix for ecutwfc=[ecut_pre]/rs^2 for fixed FFT size at all rs')
  parser.add_argument('--degauss', '-dg', type=float, default=1e-4,
    help='smearing parameter in QE')
  parser.add_argument('--verbose', '-v', action='store_true')
  args = parser.parse_args()
  ndim = args.ndim
  rs = args.rs
  nelec = args.nelec
  func = args.func
  thr = args.thr
  qidx = np.array([args.qx, args.qy, args.qz], dtype=int)[:ndim]
  # simulation cell
  volume = nelec*solid_angle(ndim)*rs**ndim
  alat = volume**(1./ndim)
  cell = alat*np.eye(ndim)
  # cosine potential
  recvec = axes_pos.raxes(cell)
  qvec = qidx @ recvec
  kf = heg_kfermi(rs, ndim=ndim)
  print('|q|/kF = ', np.linalg.norm(qvec)/kf)
  # random initial positions
  fracs = np.random.rand(nelec, ndim)
  elem_pos = {'elements': ['H']*nelec, 'positions': fracs}
  # make QE input
  inp_text = default_scf_input(thr=thr)
  pwdict = {
    'control': {},
    'system': {
      'nat': nelec,
      'input_dft': 'lda',
      'occupations': 'smearing',
      'degauss': args.degauss,
      'ecutwfc': args.ecut_pre/rs**2,
      'lcospot': True,
      'vcospot': args.vq,
    }
  }
  if func == 'ni':
    pwdict['control']['lob'] = True
    pwdict['system']['input_dft'] = 'nox-noc'
  else:
    pwdict['system']['input_dft'] = func
  for i, q1 in enumerate(qvec):
    pwdict['system']['qcospot(%d)' % (i+1)] = q1
  for group, params in pwdict.items():
    for key, val in params.items():
      inp_text = pwscf.change_keyword(inp_text, group, key, val)
  inp_text += '\n'
  cell_text = pwscf.cell_parameters(cell)
  inp_text += cell_text
  atom_text = pwscf.atomic_positions(elem_pos)
  inp_text += atom_text
  ktext = '''\nK_POINTS automatic
    1 1 1 0 0 0
'''
  inp_text += ktext
  with open(args.fout, 'w') as f:
    f.write(inp_text)

if __name__ == '__main__':
  main()  # set no global variable
