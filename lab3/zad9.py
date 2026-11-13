import html
import webbrowser
from pathlib import Path

from transformers import pipeline

ner = pipeline(
    "ner",
    model="pczarnik/herbert-base-ner",
    tokenizer="pczarnik/herbert-base-ner",
    aggregation_strategy="simple",
)


def build_html(text):
    entities = sorted(
        (e for e in ner(text) if e["entity_group"] == "PER"),
        key=lambda e: e["start"],
    )
    parts = []
    cursor = 0
    for ent in entities:
        start, end = ent["start"], ent["end"]
        parts.append(html.escape(text[cursor:start]))
        parts.append(
            f'<span style="background-color: yellow">{html.escape(text[start:end])}</span>'
        )
        cursor = end
    parts.append(html.escape(text[cursor:]))
    body = "".join(parts)
    return (
        '<!DOCTYPE html>\n'
        '<html lang="pl">\n'
        '<head>\n'
        '<meta charset="utf-8">\n'
        '<title>Rozpoznawanie encji</title>\n'
        '</head>\n'
        f'<body>\n<p>{body}</p>\n</body>\n'
        '</html>\n'
    )


text = input("Podaj tekst: ")

output = Path("wynik.html")
output.write_text(build_html(text), encoding="utf-8")
webbrowser.open(output.resolve().as_uri())
print(f"Zapisano: {output.resolve()}")