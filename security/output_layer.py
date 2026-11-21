"""Слой 4. Выход: ответ модели — тоже недоверенный текст.

Regex-логика перенесена из secure_ai_agents-main/student/output_layer.py дословно
(картинки → ссылки → голые адреса, проверка хоста через urlparse().hostname, поддомены).

Одно отличие принципиальное: в учебнике в конце ответ экранировался для HTML-страницы
(< → &lt;). Здесь канал — Telegram с parse_mode="Markdown", где &lt; показался бы
пользователю буквально. Поэтому экранирование вынесено в флаг escape_html и по умолчанию
выключено; включать его нужно, если ответ когда-нибудь пойдёт на веб-страницу.
"""
import re
from urllib.parse import urlparse

IMAGE_REMOVED = "[изображение удалено]"
LINK_REMOVED = "[ссылка удалена]"

# Разбор Markdown-конструкций. У ссылки/картинки может быть титул: [текст](url "название")
# — его тоже нужно съесть, иначе он останется в ответе после отсечения адреса.
# Мишень ссылки берём с одним уровнем вложенных скобок: иначе [x](javascript:alert(1))
# вообще не распознаётся как ссылка и опасная схема уходит наружу нетронутой.
_TARGET = r"\(\s*((?:[^()\s]|\([^()]*\))+)(?:\s+[\"'][^\"']*[\"'])?\s*\)"
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


def check_output(text: str, allowed_hosts: set, escape_html: bool = False) -> str:
    if not isinstance(text, str) or not text:
        return text if isinstance(text, str) else ""
    hosts = set(allowed_hosts or ())

    # Порядок важен: сначала картинки (иначе ссылка съест «[alt](url)» изнутри картинки),
    # потом ссылки, потом адреса, которые остались без скобок.
    out = _MD_IMAGE.sub(_markdown_sub("image", hosts), text)
    out = _MD_LINK.sub(_markdown_sub("link", hosts), out)
    out = _BARE_URL.sub(lambda m: m.group(0) if _host_allowed(m.group(0), hosts)
                       else LINK_REMOVED, out)

    if escape_html:
        # Ответ — тоже чужой текст: на HTML-страницу его вставляют уже экранированным.
        out = out.replace("<", "&lt;").replace(">", "&gt;")
    return out


def extract_hosts(text: str) -> set:
    """Хосты из ссылок, реально пришедших в выдаче инструментов.

    В учебнике allowed_hosts собирает проводка цикла; у нас цикл спрятан внутри create_agent,
    поэтому список достраиваем по ToolMessage: это ровно те адреса, которые бот имел право
    назвать, и ничего кроме них.
    """
    hosts: set = set()
    if not isinstance(text, str) or not text:
        return hosts
    for m in _BARE_URL.finditer(text):
        host = (urlparse(m.group(0)).hostname or "").rstrip(".").lower()
        if host:
            hosts.add(host)
    return hosts
