"""Бот целиком: вход → запрос к модели → цикл инструментов → проверка ответа.

Этот файл — «проводка». Решения о безопасности принимают слои из `student/`:
каждый слой вызывается здесь в нужной точке. Менять app.py не нужно.
"""
import json
import time
from dataclasses import dataclass

import lab_target

from . import kb, tools
from .world import World

ALLOWED_HOSTS = {"docs.company.example"}
KILL_FLAG = "STOP_BOT"            # если файл с таким именем существует, бот отключён
HARD_CAP = 200                    # страховка самой лабораторной от бесконечного цикла


class Layers:
    """Слои защиты, загруженные из student/ (или solutions/, см. lab_target.py)."""

    def __init__(self):
        self.input = lab_target.load("input_layer")
        self.prompt = lab_target.load("prompt_layer")
        self.rag = lab_target.load("rag_layer")
        self.output = lab_target.load("output_layer")
        self.gateway = lab_target.load("gateway_layer")
        self.secrets = lab_target.load("secrets_layer")
        self.obs = lab_target.load("obs_layer")


@dataclass
class BotResult:
    reply: str
    status: str        # ok | rejected | rate_limited | killed | budget


class Bot:
    def __init__(self, llm, world: World | None = None, confirm=None, kill_flag: str = KILL_FLAG, clock=time.time):
        self.llm = llm
        self.world = world or World()
        self.confirm = confirm or (lambda tool, args: False)   # «человек» по умолчанию отказывает
        self.kill_flag = kill_flag
        self.clock = clock
        self.L = Layers()
        self.limiter = self.L.input.RateLimiter(limit=5, window_s=60)
        self.chunks = self._ingest()

    # --- приём документов в индекс ------------------------------------------------------------
    def _ingest(self) -> list:
        chunks = []
        for raw in kb.load_raw_chunks():
            if not self.L.rag.is_trusted_source(raw["source"]):
                continue
            chunks.append({**raw, "text": self.L.rag.clean_document(raw["text"])})
        return chunks

    # --- журнал -------------------------------------------------------------------------------
    def _log(self, user_id: str, event: str, text: str):
        record = self.L.obs.audit_event(user_id, event, text)
        self.world.log.append(self.L.secrets.redact(json.dumps(record, ensure_ascii=False)))

    # --- основной вход ------------------------------------------------------------------------
    def handle(self, user_id: str, text: str) -> BotResult:
        w, L = self.world, self.L
        user = kb.USERS[user_id]
        if L.obs.kill_switch_active(self.kill_flag):
            return BotResult("Бот временно отключён.", "killed")
        if not self.limiter.allow(user_id, self.clock()):
            return BotResult("Слишком много запросов, попробуйте позже.", "rate_limited")
        try:
            clean = L.input.sanitize_input(text)
        except L.input.InputRejected:
            return BotResult("Не удалось обработать запрос.", "rejected")
        w.user_chars = max(w.user_chars, len(clean))
        self._log(user_id, "request", clean)

        messages = [{"role": "system", "content": L.prompt.build_system_prompt()},
                    {"role": "user", "content": clean}]
        budget = L.gateway.Budget()
        final, status = "", "ok"
        while True:
            if L.obs.kill_switch_active(self.kill_flag):
                final, status = "Бот временно отключён.", "killed"
                break
            if not budget.spend() or w.llm_calls >= HARD_CAP:
                final, status = "Достигнут лимит шагов.", "budget"
                break
            resp = self.llm.chat(messages, tools.TOOLS)
            w.llm_calls += 1
            calls = resp.get("tool_calls") or []
            if not calls:
                final = resp.get("content") or ""
                break
            messages.append({"role": "assistant", "content": resp.get("content"), "tool_calls": [
                {"id": c["id"], "type": "function",
                 "function": {"name": c["name"], "arguments": json.dumps(c["arguments"], ensure_ascii=False)}}
                for c in calls]})
            for c in calls:
                messages.append({"role": "tool", "tool_call_id": c["id"], "content": self._run_call(c, user)})

        reply = L.output.check_output(final, ALLOWED_HOSTS)
        self._log(user_id, "reply", reply)
        return BotResult(reply, status)

    def _run_call(self, call: dict, user: dict) -> str:
        name, args = call["name"], call["arguments"]
        decision = self.L.gateway.decide(name, args)
        if decision == "ask" and not self.confirm(name, args):
            decision = "deny"
        self.world.tool_calls.append({"tool": name, "args": args, "decision": decision})
        if decision != "allow" and decision != "ask":
            return "ОТКАЗ: действие запрещено политикой."
        return tools.run_tool(name, args, user=user, chunks=self.chunks, layers=self.L, world=self.world)
