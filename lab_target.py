"""Откуда брать слои защиты: ваши (`student/`) или эталонные (`solutions/`).

    LAB_SOLUTIONS=all pytest                     — все слои эталонные (проверка стенда преподавателем)
    LAB_SOLUTIONS=gateway_layer,input_layer ...  — эталонными только перечисленные слои

По умолчанию берутся ваши файлы из `student/`. Папки `solutions/` в ветке main нет.
"""
import importlib
import os


def load(name: str):
    chosen = {s.strip() for s in os.environ.get("LAB_SOLUTIONS", "").split(",") if s.strip()}
    package = "solutions" if ("all" in chosen or name in chosen) else "student"
    return importlib.import_module(f"{package}.{name}")
