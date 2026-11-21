from bot.app import Bot
from bot.mock_llm import MockLLM
from bot.world import World


def bot_and_world(**kw):
    w = World()
    return Bot(MockLLM(), w, **kw), w
