import json
p = 'f:/source/code/direction/rag/learning-roadmap/03-Claude-Skills/practice/stage2_进阶/35_description_recall_test.ipynb'
nb = json.loads(open(p, encoding='utf-8').read())
required = {'cell_type', 'metadata', 'source'}
for i, cell in enumerate(nb['cells']):
    keys = set(cell.keys())
    if not required.issubset(keys):
        missing = required - keys
        cid = cell.get('id', '?')
        print(f'cell {i} (id={cid}): missing {missing}, has {keys}')
    src = cell.get('source')
    if isinstance(src, list) and src and not all(isinstance(s, str) for s in src):
        bad = [type(s).__name__ for s in src if not isinstance(s, str)]
        cid = cell.get('id', '?')
        print(f'cell {i} (id={cid}): source has non-string elements: {bad[:3]}')
print('done')
