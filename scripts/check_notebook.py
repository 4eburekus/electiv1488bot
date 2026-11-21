"""Быстрая проверка ноутбука без запуска моделей и без сети.

    python scripts/check_notebook.py [путь/к/ноутбуку.ipynb]

Проверяется: корректный JSON (nbformat 4); синтаксис каждой код-ячейки (магии !cmd и %cmd
отбрасываются); нет ошибок в сохранённых выводах; нет ключей вида sk-or-... и tvly-...;
у ячеек с тегом needs-llm нет сохранённых выводов; нет невидимых символов нулевой ширины.
Код возврата 0 — всё в порядке, 1 — есть замечания.
"""
import ast
import json
import re
import sys
from pathlib import Path

SECRET_RE = re.compile(r"(sk-or-v1-[A-Za-z0-9]{10,}|tvly-[A-Za-z0-9]{10,})")
INVISIBLE_RE = re.compile("[​-‏⁠﻿]")


def strip_magics(source: str) -> str:
    out = []
    for line in source.splitlines():
        stripped = line.lstrip()
        indent = line[: len(line) - len(stripped)]
        out.append(indent + "pass" if stripped.startswith(("!", "%")) else line)
    return "\n".join(out)


def main(path: Path) -> int:
    problems = []
    try:
        nb = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        print(f"НЕ JSON: {path}: {exc}")
        return 1
    if nb.get("nbformat") != 4:
        problems.append("nbformat не 4")
    for i, cell in enumerate(nb.get("cells", [])):
        src = "".join(cell.get("source", []))
        if SECRET_RE.search(src):
            problems.append(f"ячейка {i}: похоже на ключ")
        if INVISIBLE_RE.search(src):
            problems.append(f"ячейка {i}: невидимые символы нулевой ширины")
        if cell["cell_type"] != "code":
            continue
        try:
            ast.parse(strip_magics(src))
        except SyntaxError as exc:
            problems.append(f"ячейка {i}: синтаксис: {exc}")
        for out in cell.get("outputs", []):
            if out.get("output_type") == "error":
                problems.append(f"ячейка {i}: в выводе ошибка")
        if "needs-llm" in cell.get("metadata", {}).get("tags", []) and cell.get("outputs"):
            problems.append(f"ячейка {i}: у needs-llm сохранён вывод")
    for p in problems:
        print("-", p)
    print("OK" if not problems else f"Замечаний: {len(problems)}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1] if len(sys.argv) > 1 else "lab_notebook.ipynb")))
