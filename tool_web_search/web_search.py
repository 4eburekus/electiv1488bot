import time
from concurrent.futures import ThreadPoolExecutor

from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

DEFAULT_MAX_RESULTS = 5
DEFAULT_TIMELIMIT = "y"  # проиндексировано за последнийy y — год
DEFAULT_QUERY = "что нового в больших языковых моделях"

MAX_RETRIES = 3  # всего попыток, включая первую
RETRY_DELAY = 1.0  # пауза между попытками, сек
SEARCH_TIMEOUT = 30  # лимит времени на одну попытку поиска, сек


def _search_once(query: str, max_results: int, timelimit: str) -> list[dict]:
    """Один запрос к DuckDuckGo через langchain-обёртку."""
    search = DuckDuckGoSearchAPIWrapper(max_results=max_results, time=timelimit)
    return search.results(query, max_results=max_results)


def web_search(
    query: str,
    max_results: int = DEFAULT_MAX_RESULTS,
    timelimit: str = DEFAULT_TIMELIMIT,
) -> list[dict]:
    """Ищет в DuckDuckGo и возвращает список результатов.

    Делает не больше MAX_RETRIES попыток; каждая ограниченна SEARCH_TIMEOUT
    секундами. Если ни одна попытка не удалась, возвращает пустой список,
    а не бросает исключение.

    Каждый результат — dict с ключами title, link, snippet.
    """
    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        executor = ThreadPoolExecutor(max_workers=1)
        try:
            future = executor.submit(_search_once, query, max_results, timelimit)
            return future.result(timeout=SEARCH_TIMEOUT)
        except Exception as exc:  # noqa: BLE001 - любую ошибку перехватываем и повторяем
            last_error = exc
            print(f"⚠️  Попытка {attempt}/{MAX_RETRIES} не удалась: {type(exc).__name__}: {exc}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    print(f"❌ Поиск не выполнен после {MAX_RETRIES} попыток: {type(last_error).__name__}: {last_error}")
    return []


def print_results(results: list[dict], query: str | None = None) -> None:
    """Красиво печатает результаты поиска в консоль."""
    if query:
        print(f"🔍 Запрос: {query}")
        print("=" * 80)

    if not results:
        print("\n📋 Результаты не найдены.\n")
        return

    print(f"\n📋 Найдено результатов: {len(results)}\n")
    for i, result in enumerate(results, 1):
        title = result.get("title", "N/A")
        link = result.get("link", "N/A")
        snippet = result.get("snippet", "N/A")
        print(f"{i}. {title}")
        print(f"   🔗 Ссылка: {link}")
        if len(snippet) > 150:
            print(f"   📝 Описание: {snippet[:150]}...")
        else:
            print(f"   📝 Описание: {snippet}")
        print()

    print("✅ Поиск завершён успешно!")


def main() -> None:
    query = DEFAULT_QUERY
    results = web_search(query, max_results=DEFAULT_MAX_RESULTS, timelimit=DEFAULT_TIMELIMIT)
    print_results(results, query=query)


if __name__ == "__main__":
    main()
