#!/usr/bin/env python3
import argparse
import json
import os
import re
from pathlib import Path

CHINESE_REGEX = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\u3000-\u303f\uFF00-\uFFEF]")
NONASCII_REGEX = re.compile(r"[^\x00-\x7F]")

DEFAULT_EXTS = {
    ".py", ".md", ".txt", ".json", ".yaml", ".yml", ".ini", ".toml",
    ".sh", ".bash", ".csv", ".tsv", ".js", ".ts", ".jsx", ".tsx",
    ".css", ".scss", ".html", ".htm", ".xml", ".cfg"
}

DEFAULT_EXCLUDE_DIRS = {
    ".git", "__pycache__", "node_modules", ".venv", "venv", ".idea", ".vscode",
    "dist", "build", "out"
}

def is_text_file(p: Path, exts: set) -> bool:
    return p.suffix.lower() in exts

def scan_file(p: Path, regex: re.Pattern):
    results = []
    try:
        with open(p, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                matches = list(regex.finditer(line))
                if matches:
                    chars = "".join(sorted(set(m.group() for m in matches)))
                    snippet = line.strip()
                    results.append({"line": i, "chars": chars, "snippet": snippet})
    except Exception:
        pass
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--pattern", choices=["chinese", "nonascii"], default="chinese")
    parser.add_argument("--extensions", default=None)
    parser.add_argument("--exclude-dir", action="append", default=None)
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    exts = DEFAULT_EXTS if args.extensions is None else set(
        [("." + e.lower().lstrip(".")) for e in args.extensions.split(",") if e.strip()]
    )
    excludes = DEFAULT_EXCLUDE_DIRS if args.exclude_dir is None else DEFAULT_EXCLUDE_DIRS.union(set(args.exclude_dir))
    regex = CHINESE_REGEX if args.pattern == "chinese" else NONASCII_REGEX

    results = {}
    for dirpath, dirnames, filenames in os.walk(root):
        parts = set(Path(dirpath).parts)
        if parts.intersection(excludes):
            continue
        for fn in filenames:
            p = Path(dirpath) / fn
            if not is_text_file(p, exts):
                continue
            r = scan_file(p, regex)
            if r:
                results[str(p)] = r

    if args.json:
        print(json.dumps(results, ensure_ascii=True, indent=2))
        return

    total_matches = sum(len(v) for v in results.values())
    print(f"Found {total_matches} lines with matches in {len(results)} files")
    for file_path, entries in sorted(results.items()):
        print(file_path)
        count = 0
        for item in entries:
            print(f"  {item['line']}: {item['chars']} | {item['snippet']}")
            count += 1
            if count >= args.limit:
                remaining = len(entries) - count
                if remaining > 0:
                    print(f"  ... ({remaining} more)")
                break

if __name__ == "__main__":
    main()

