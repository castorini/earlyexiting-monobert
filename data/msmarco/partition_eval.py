import os
import sys

split = sys.argv[1]
small = False
if len(sys.argv) > 2 and sys.argv[2] == 'small':
    small = True
query_per_partition = 500 if small else 100  # 2 partitions for small

src_fname = 'top1000.' + split + ('.small' if small else '')
collection = {}  # key: qid; value: list of lines
with open(src_fname) as fin:
    for line in fin:
        qid = line.split('\t')[0]
        if qid not in collection:
            collection[qid] = []
        collection[qid].append(line)

folder_name = split + ('-small' if small else '') + '_partitions'
if not os.path.exists(folder_name):
    os.mkdir(folder_name)
partition_count = -1
for i, qid in enumerate(collection):
    if i % query_per_partition == 0:
        partition_count += 1
        print(partition_count)
    with open(folder_name+'/partition'+str(partition_count), 'a') as fout:
        for line in collection[qid]:
            print(line, end='', file=fout)

