"""Fix 03-Skills notebooks: literal LF inside string in markdown table separator."""
from pathlib import Path

paths = [
    'f:/source/code/direction/rag/learning-roadmap/03-Claude-Skills/practice/stage3_高级/36_skill_subagent_compose.ipynb',
    'f:/source/code/direction/rag/learning-roadmap/03-Claude-Skills/practice/stage3_高级/37_skill_hooks_permissions.ipynb',
    'f:/source/code/direction/rag/learning-roadmap/03-Claude-Skills/practice/stage4_专家/38_skill_repo_skeleton.ipynb',
]

for p in paths:
    raw = Path(p).read_bytes()
    # Look for: a line that's a markdown table separator "|...|..." followed by raw LF
    # The actual broken pattern: |--------|\n",   (with 0x0A inside the string)
    # We want to fix to: |--------|\\n",  (with 0x5C 0x6E)
    # The strings to find: lines starting with "    \"|" then "|" then raw 0x0A then "\""

    # The pattern: any line in a "source" array that's like
    #   "    |XXX|<raw 0x0A>",  (no \\n escape)
    # We fix by replacing the raw 0x0A with \\n (two chars)

    fixed = bytearray()
    i = 0
    inside_string = False
    quote = None
    while i < len(raw):
        ch = raw[i:i+1]
        if not inside_string:
            if ch == b'"':
                inside_string = True
                quote = ch
            fixed.extend(ch)
            i += 1
        else:
            if ch == b'\\':
                # escape sequence, copy 2 bytes
                fixed.extend(ch)
                if i+1 < len(raw):
                    fixed.extend(raw[i+1:i+2])
                    i += 2
                else:
                    i += 1
            elif ch == b'"':
                inside_string = False
                fixed.extend(ch)
                i += 1
            elif ch == b'\n':
                # raw 0x0A inside string — replace with \\n
                fixed.extend(b'\\\\n')
                i += 1
            elif ch == b'\r':
                fixed.extend(b'\\\\r')
                i += 1
            else:
                fixed.extend(ch)
                i += 1
    new = bytes(fixed)
    if new == raw:
        print(f'NO-CHANGE  {p}')
    else:
        Path(p).write_bytes(new)
        print(f'FIXED  {p}')

# Validate all
import json
for p in paths:
    try:
        json.loads(Path(p).read_text(encoding='utf-8'))
        print(f'OK JSON  {p}')
    except json.JSONDecodeError as e:
        print(f'STILL BAD  {p}: {e}')
