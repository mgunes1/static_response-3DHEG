#!/usr/bin/env python3
# from: bwc/007-rs-da/b_liq/workflow
# then: bwc/010-2deg/common
import os
import numpy as np
from qharv.seed import xml
from o2dmc import read_opt

def read_nknot(doc):
  nl = []
  for corr in doc.findall('.//correlation'):
    n1 = int(corr.get('size'))
    nl.append(n1)
  nknot = max(nl)
  return nknot

def main():
  from argparse import ArgumentParser
  parser = ArgumentParser()
  parser.add_argument('--fxml', '-i', default='../opt/opt.xml')
  parser.add_argument('--fout', '-o', default='opt.xml')
  parser.add_argument('--rcut', type=float)
  parser.add_argument('--ts_frac', type=float, default=0.5)  # reduce timestep due to wbyw
  parser.add_argument('--nloop', type=int, default=16)
  parser.add_argument('--nsample', type=int)
  parser.add_argument('--freeze_jastrow', action='store_true')
  parser.add_argument('--use_old', action='store_true')
  parser.add_argument('--verbose', '-v', action='store_true')
  args = parser.parse_args()
  if os.path.relpath(args.fout, args.fxml) == '.':
    msg = 'refuse to overwrite input'
    raise RuntimeError(msg)
  rcut = args.rcut
  nsample = args.nsample

  # step 1: use optimal jastrow (opt=no)
  doc = read_opt(args.fxml, update_wavefunction=(not args.use_old))
  nknot = read_nknot(doc)
  if args.freeze_jastrow:
    xml.turn_off_jas_opt(doc)
  # step 2: put in backflow
  detset = doc.find('.//determinantset')
  species = xml.get_species(doc)
  nspec = len(species)
  if nspec not in [1, 2, 4]:
    raise NotImplementedError()
  bf = xml.make_node('backflow')
  detset.append(bf)
  bee = xml.make_node('transformation', {'type': 'e-e', 'name': 'eeB', 'function': 'Bspline'})
  bf.append(bee)
  cz = np.zeros(nknot)
  cuu = xml.build_corr(cz, species[0], species[0], cpre='ee', cusp=0, rcut=rcut)
  bee.append(cuu)
  if nspec > 1:
    cud = xml.build_corr(cz, species[0], species[1], cpre='ee', cusp=0, rcut=rcut)
    bee.append(cud)
  if nspec == 4:
    # !!!! assume 0-1 in one layer, 2-3 in the other layer
    cud = xml.build_corr(cz, species[2], species[3], cpre='ee', cusp=0, rcut=rcut)
    bee.append(cud)
    for sp1 in species[:2]:
      ilayer1 = 0 if len(sp1) == 1 else int(sp1[1:])
      for sp2 in species[2:]:
        ilayer2 = 0 if len(sp2) == 1 else int(sp2[1:])
        assert ilayer1 != ilayer2
        cinter = xml.build_corr(cz, sp1, sp2, cpre='ee', cusp=0, rcut=rcut)
        coeff = cinter.find('.//coefficients')
        #coeff.set('optimize', 'no')
        bee.append(cinter)
  # step 4: wbyw qmc (smaller timestep)
  loop = doc.find('.//loop')
  loop.set('max', str(args.nloop))
  qmc = loop.find('.//qmc')
  qmc.set('move', 'wbyw')
  ts = float(xml.get_param(qmc, 'timestep'))
  xml.set_param(qmc, 'timestep', str(ts*args.ts_frac))
  nstep = int(xml.get_param(qmc, 'steps'))
  #ns1 = max(1, int(round(nstep/args.ts_frac)))
  ns1 = nstep
  xml.set_param(qmc, 'steps', str(ns1))
  if nsample is not None:
    xml.set_param(qmc, 'samples', str(nsample))
  # change to quartic optimizer?
  xml.write(args.fout, doc)

if __name__ == '__main__':
  main()  # set no global variable
