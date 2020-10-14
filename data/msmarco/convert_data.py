import sys
from tqdm import tqdm

'''
Usage: python convert_data.py SRC_FILE TGT_FILE
SRC_FILE: input file
TGT_FILE: output file name (in tsv format: label\\tquery\\tdocument)
'''

src_fname = sys.argv[1]
tgt_fname = sys.argv[2]
uniq = True

with open(tgt_fname, 'w') as fout:
    print('Label\tQuery\tDoc', file=fout)
    uniq_set = set()
    with open(src_fname) as fin:

        for count, line in enumerate(tqdm(fin)):
            query, rel_doc, irrel_doc = line.strip().split('\t')
            if uniq and (query, rel_doc) in uniq_set:
                continue
            uniq_set.add((query, rel_doc))
            print(f'1\t{query}\t{rel_doc}', file=fout)
            print(f'0\t{query}\t{irrel_doc}', file=fout)

print('Conversion done')
