#!/usr/bin/env python3
import os
import sys
import ast
import tokenize
import argparse
import fnmatch

def has_non_ascii(s: str) -> bool:
    return any(ord(ch) > 127 for ch in s)

def scan_python_file(path: str):
    results = []
    try:
        with tokenize.open(path) as f:
            for tok in tokenize.generate_tokens(f.readline):
                if tok.type == tokenize.COMMENT and has_non_ascii(tok.string):
                    results.append(("comment", tok.start[0], tok.string.strip()))
    except Exception:
        pass
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            src = f.read()
        tree = ast.parse(src, path)
    except Exception:
        return results
    def add_doc(node):
        doc = ast.get_docstring(node, clean=False)
        if doc and has_non_ascii(doc):
            lineno = 1
            if isinstance(node, ast.Module):
                if node.body and isinstance(node.body[0], ast.Expr):
                    lineno = node.body[0].lineno
            else:
                if getattr(node, "body", None) and isinstance(node.body[0], ast.Expr):
                    lineno = node.body[0].lineno
            results.append(("docstring", lineno, doc))
    add_doc(tree)
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            add_doc(n)
        if isinstance(n, ast.Call):
            func = None
            if isinstance(n.func, ast.Name):
                func = n.func.id
            if func == "print":
                strings = []
                def collect(x):
                    if isinstance(x, ast.Constant) and isinstance(x.value, str):
                        strings.append(x.value)
                    elif isinstance(x, ast.Str):
                        strings.append(x.s)
                    elif isinstance(x, ast.JoinedStr):
                        for v in x.values:
                            if isinstance(v, ast.Constant) and isinstance(v.value, str):
                                strings.append(v.value)
                            elif isinstance(v, ast.Str):
                                strings.append(v.s)
                    elif isinstance(x, ast.BinOp):
                        collect(x.left)
                        collect(x.right)
                    elif isinstance(x, (ast.Tuple, ast.List)):
                        for el in getattr(x, "elts", []):
                            collect(el)
                for a in n.args:
                    collect(a)
                for s in strings:
                    if has_non_ascii(s):
                        results.append(("print", getattr(n, "lineno", 0) or 0, s))
    return results

DEFAULT_SKIP = {".git", ".trae", "venv", "env", ".venv", "__pycache__", "build", "dist", ".idea", ".vscode", "data", "node_modules"}

def load_gitignore(root: str):
    patterns = []
    p = os.path.join(root, ".gitignore")
    if os.path.isfile(p):
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                patterns.append(s)
    return patterns

def match_any(rel: str, name: str, patterns):
    for pat in patterns:
        if pat.startswith("!"):
            continue
        q = pat
        if q.startswith("/"):
            q = q[1:]
        if q.endswith("/"):
            q = q.rstrip("/")
            if rel == q or rel.startswith(q + os.sep):
                return True
            continue
        if fnmatch.fnmatch(rel, q) or fnmatch.fnmatch(name, q):
            return True
    return False

def is_ignored_dir(dirpath: str, root: str, patterns) -> bool:
    base = os.path.basename(dirpath)
    if base in DEFAULT_SKIP:
        return True
    rel = os.path.relpath(dirpath, root)
    return match_any(rel, base, patterns)

def is_ignored_file(path: str, root: str, patterns) -> bool:
    rel = os.path.relpath(path, root)
    name = os.path.basename(path)
    if name.startswith('.') and name != '.gitignore':
        return True
    return match_any(rel, name, patterns)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    root = os.path.abspath(args.root)
    patterns = load_gitignore(root)
    total = 0
    files = 0
    printed = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not is_ignored_dir(os.path.join(dirpath, d), root, patterns)]
        for fn in filenames:
            if not fn.endswith(".py"):
                continue
            path = os.path.join(dirpath, fn)
            if is_ignored_file(path, root, patterns):
                continue
            res = scan_python_file(path)
            if res:
                files += 1
                for kind, lineno, text in res:
                    total += 1
                    if args.limit is None or printed < args.limit:
                        printed += 1
                        text = text.replace("\n", " ")
                        print(f"{path}:{lineno} [{kind}] {text}")
    print(f"Total matches: {total} in {files} files")

if __name__ == "__main__":
    main()
