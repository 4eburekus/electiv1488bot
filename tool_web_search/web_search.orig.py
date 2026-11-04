import ddgs
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

search = DuckDuckGoSearchAPIWrapper(max_results=5)
query = "что нового в больших языковых моделях"
print(f"🔍 Запрос: {query}")
print("=" * 80)
results = search.results(query, max_results=5)

print(f"\n📋 Найдено результатов: {len(results)}\n")
for i, result in enumerate(results, 1):
    title = result.get('title', 'N/A')
    link = result.get('link', 'N/A')
    snippet = result.get('snippet', 'N/A')
    print(f"{i}. {title}")
    print(f"   🔗 Ссылка: {link}")
    print(f"   📝 Описание: {snippet[:150]}..." if len(snippet) > 150 else f"   📝 Описание: {snippet}")
    print()

print("✅ Поиск завершён успешно!")

# if __name__ == "__main__":


