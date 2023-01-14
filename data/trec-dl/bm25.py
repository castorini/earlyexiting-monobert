import os
import json
from pyserini.search.lucene import LuceneSearcher

"""
1. organize the collection into json format
2. index with pyserini
3. extract queries that appear in qrels (43 in total, while there about 4100 positive qrels - so about 100 per query)
4. search the top-200 documents for each query

qrel distribution:
cat 2019qrels-pass.txt | sort -n -k4 | cut -d' ' -f4-4  | uniq -c
   5158 0
   1601 1
   1804 2
    697 3
"""

raw_dir = "raw_data"
json_dir = "collection_jsonl"
col_in = "collection.tsv"
col_out = "file0.jsonl"
qrels_in = "2019qrels-pass.txt"
queries_in = "msmarco-test2019-queries.tsv"
queries_out = "queries.tsv"
rerank_out = "dev_partitions/partition0"

os.makedirs(json_dir, exist_ok=True)
collections = {}
with open(os.path.join(json_dir, col_out), 'w') as fout:
    with open(os.path.join(raw_dir, col_in)) as fin:
        for line in fin:
            idx, txt = line.strip().split('\t')
            collections[idx] = txt
            print(json.dumps({
                "id": idx,
                "contents": txt,
            }), file=fout)
            if int(idx) % 1000000 == 0:
                print(idx)


index_cmd = f"""python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input {json_dir} \
  --index indexes/collection_jsonl \
  --generator DefaultLuceneDocumentGenerator \
  --threads 1 \
  --storePositions --storeDocvectors --storeRaw"""
os.system(index_cmd)


valid_qids = set()
with open(os.path.join(raw_dir, qrels_in)) as fin:
    for line in fin:
        qid, _, _, score = line.strip().split(' ')
        if score != '0' and qid not in valid_qids:
            valid_qids.add(qid)
with open(queries_out, 'w') as fout:
    with open(os.path.join(raw_dir, queries_in)) as fin:
        for line in fin:
            qid, txt = line.strip().split('\t')
            if qid in valid_qids:
                print(f'{qid}\t{txt}', file=fout)


searcher = LuceneSearcher('indexes/collection_jsonl')
# warning: gotta comment out
# pyserini/encode/_base.py:19  import faiss
K = 200
os.makedirs("dev_partitions", exist_ok=True)
os.makedirs("partition_cache", exist_ok=True)
with open(rerank_out, 'w') as fout:
    with open(queries_out) as fin:
        for line in fin:
            qid, query = line.strip().split('\t')
            hits = searcher.search(query, k=K)

            for i in range(len(hits)):
                docid = hits[i].docid
                print(f'{qid}\t{docid}\t{query}\t{collections[docid]}', file=fout)
