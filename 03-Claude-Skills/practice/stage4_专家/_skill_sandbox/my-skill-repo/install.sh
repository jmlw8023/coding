#!/usr/bin/env bash
# 一键装 skill 到指定位置
# 用法: ./install.sh [user|project] [path]
set -e
TARGET="${1:-user}"
REPO_SKILLS="$(dirname "$(realpath "$0")")/skills"
if [ "$TARGET" = "user" ]; then
    DEST="${HOME}/.claude/skills"
elif [ "$TARGET" = "project" ]; then
    DEST="${2:-./.claude/skills}"
else
    echo "usage: $0 [user|project] [path]"
    exit 1
fi
echo "→ 装到 $DEST"
mkdir -p "$DEST"
for skill_dir in "$REPO_SKILLS"/*/; do
    name="$(basename "$skill_dir")"
    if [ -d "$DEST/$name" ]; then
        echo "  [skip] $name 已在"; continue
    fi
    cp -r "$skill_dir" "$DEST/$name"
    echo "  [ok]   $name"
done
echo "完成。"
