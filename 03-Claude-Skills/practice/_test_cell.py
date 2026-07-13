skill_dir = FAKE_SKILLS / 'explain-code'
skill_dir.mkdir(exist_ok=True)
skill_md = skill_dir / 'SKILL.md'
skill_md.write_text('''---
name: explain-code
description: |
  When the user asks to explain a snippet of code, a function, a class, or a section of a file.
  Trigger phrases include "explain this code", "what does this function do", "解读这段代码。
  Do NOT trigger for: rewriting code, reviewing PRs, or generating new code.
---

# When to use
User wants a clear natural-language explanation of existing code.

# When NOT to use
- User wants to rewrite or fix the code
- User wants a PR-level review
- User wants to generate new code from scratch

# Steps
1. Read the snippet with the Read tool
2. Identify the language and key constructs
3. Write a structured explanation
4. If anything is ambiguous, ask the user instead of guessing

# Example
User: "explain this decorator"
You read the code and produce a structured explanation.
'''
print(f'已建 SKILL.md ({skill_md.stat().st_size} bytes)')
