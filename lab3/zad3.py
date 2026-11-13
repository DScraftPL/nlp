import re
from pprint import pprint


def word_features(tokens, i):
    word = tokens[i]
    return {
        "word": word.lower(),
        "is_title": word.istitle(),
        "is_upper": word.isupper(),
        "is_digit": word.isdigit(),
        "is_punct": bool(re.fullmatch(r"[^\w\s]+", word)),
        "prefix2": word[:2],
        "suffix3": word[-3:],
        "prev": tokens[i - 1].lower() if i > 0 else "<BOS>",
        "next": tokens[i + 1].lower() if i < len(tokens) - 1 else "<EOS>",
        "length": len(word),
    }


tokens = ["A125", "Aleksandra", "Szymański", ",", "ul", ".", "Leśna", "29", ",", "Poznań"]

for i, token in enumerate(tokens):
    print(token)
    pprint(word_features(tokens, i))