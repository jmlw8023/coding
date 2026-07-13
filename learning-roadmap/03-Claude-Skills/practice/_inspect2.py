import json
p = 'f:/source/code/direction/rag/learning-roadmap/03-Claude-Skills/practice/stage2_进阶/35_description_recall_test.ipynb'
nb = json.loads(open(p, encoding='utf-8').read())
# Look for malformed source lines (each source should be a list of strings, each string ending with \n only if it's not the last)
for i, cell in enumerate(nb['cells']):
    src = cell.get('source')
    if isinstance(src, list):
        for j, s in enumerate(src):
            # any source item that has a raw \n in the middle is suspicious but technically valid JSON
            # however nbformat requires specific format
            if not isinstance(s, str):
                print(f'cell {i} source[{j}] not str: {type(s).__name__}')
            # check if source is an empty list (which is invalid for cells)
            if j == 0 and s == '' and len(src) == 1:
                print(f'cell {i} has empty source')
# Check nbformat_minor
print('nbformat:', nb.get('nbformat'), 'minor:', nb.get('nbformat_minor'))
print('metadata keys:', list(nb.get('metadata', {}).keys()))
# Check kernelspec
ks = nb['metadata'].get('kernelspec', {})
print('kernelspec:', ks)
print('language_info:', nb['metadata'].get('language_info'))
