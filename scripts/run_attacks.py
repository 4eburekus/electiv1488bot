"""Прогон набора атак и критерий выпуска.

    python scripts/run_attacks.py                        # учебная имитация модели (mock)
    python scripts/run_attacks.py --provider ollama --repeat 3
    LAB_SOLUTIONS=all python scripts/run_attacks.py      # эталонные слои (для сравнения)

Живая модель отвечает по-разному от запуска к запуску, поэтому каждую атаку повторяют
несколько раз и считают долю успехов (--repeat). Код возврата: 0 — выпускать можно, 1 — нельзя.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import llm  # noqa: E402
from bot.attacks import MAX_SUCCESS_RATE, run_all, verdict  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default=None, help="mock | openrouter | ollama | lmstudio | custom")
    ap.add_argument("--repeat", type=int, default=None, help="повторов на атаку (mock: 1, живая модель: 3)")
    ap.add_argument("--json", default=None, help="сохранить результаты в файл")
    args = ap.parse_args()

    probe = llm.make_llm(args.provider)
    repeat = args.repeat or (1 if probe.name == "mock" else 3)
    print(f"Модель: {probe.name}; повторов на атаку: {repeat}\n")

    results = run_all(lambda: llm.make_llm(args.provider), repeat=repeat)
    print(f"{'id':<5}{'удалось':<10}{'обяз.':<7}слой")
    for r in results:
        mark = f"{r['wins']}/{r['runs']}"
        print(f"{r['id']:<5}{mark:<10}{'да' if r['mandatory'] else '':<7}{r['layer']}")

    total = sum(r["runs"] for r in results)
    wins = sum(r["wins"] for r in results)
    ok, reasons = verdict(results)
    print(f"\nУдалось атак: {wins} из {total} ({wins / total:.0%}); порог {MAX_SUCCESS_RATE:.0%}")
    print("ВЫПУСКАТЬ МОЖНО" if ok else "ВЫПУСКАТЬ НЕЛЬЗЯ:\n  - " + "\n  - ".join(reasons))
    if args.json:
        Path(args.json).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
