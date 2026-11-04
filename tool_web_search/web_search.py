from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

DEFAULT_MAX_RESULTS = 5
DEFAULT_TIMELIMIT = "y"  # d — сутки, w — неделя, m — месяц, y — год
DEFAULT_QUERY = "что нового в больших языковых моделях"


def web_search(
    query: str,
    max_results: int = DEFAULT_MAX_RESULTS,
    timelimit: str = DEFAULT_TIMELIMIT,
) -> list[dict]:
    """Ищет в DuckDuckGo и возвращает список результатов.
    Каждый результат — dict с ключами title, link, snippet.
    """
    search = DuckDuckGoSearchAPIWrapper(max_results=max_results, time=timelimit)
    return search.results(query, max_results=max_results)


def print_results(results: list[dict], query: str | None = None) -> None:
    """Красиво печатает результаты поиска в консоль."""
    if query:
        print(f"🔍 Запрос: {query}")
        print("=" * 80)
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
