import os
import sys

split = sys.argv[1]
query_per_partition = 500

src_fname = 'dev.tsv'
collection = {}  # key: qid; value: list of lines
with open(src_fname) as fin:
    for line in fin:
        qid = line.split('\t')[0]
        if qid not in collection:
            collection[qid] = []
        collection[qid].append(line)

folder_name = 'dev_partitions'
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

