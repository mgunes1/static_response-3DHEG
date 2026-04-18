#!/usr/bin/env python3
# from bwc/005-mag/b_cta/liq/gather/workflow
from qharv.sieve import gather

def add_meta(df, fxml):
  # extract more meta data from input
  from qharv.seed import xml
  doc = xml.read(fxml)
  sposet = doc.find('.//sposet')
  twist = sposet.get('twist')
  nelec = xml.get_nelec(doc)
  # add to df columns
  if twist is not None:
    df['twist'] = twist
  df['nelec'] = nelec

def main():
  from argparse import ArgumentParser
  parser = ArgumentParser()
  parser.add_argument('--fxml', '-i', default='dmc.xml')
  parser.add_argument('--fcsv', '-o', default='dmc.csv')
  parser.add_argument('--nequil', '-e', type=int, action='append')
  parser.add_argument('--verbose', '-v', action='store_true')
  args = parser.parse_args()
  neql = args.nequil
  fcsv = args.fcsv

  df = gather.scalar_dat(args.fxml, neql)
  add_meta(df, args.fxml)
  if args.verbose:
    print(df)
  df.to_csv(fcsv, index=False)

if __name__ == '__main__':
  main()  # set no global variable
