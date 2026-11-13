import json
import re

import sklearn_crfsuite
from sklearn.model_selection import train_test_split
from sklearn_crfsuite import metrics

lines = ['A703 Anna Nowak, ul. Polna 5, Warszawa, zgłasza problem z produktem lodówka Samsung RB34, zamrażarka nie mrozi, wszystko w środku rozmrożone, SKU 882104\n',
 'B102 Anna Kovalenko, ul. Jana Pawła II 6/2, Warszawa, zgłasza problem z produktem lodówka Kenwood A4, zamrażarka nie mrozi i dzieją się rzeczy niestworzone, SKU 882105\n',
 'C455 Marek Wiśniewski, al. Wolności 12/4, Kraków, zgłasza problem z produktem zmywarka Beko DFS, naczynia są brudne po myciu, chyba zapycha się jakiś filtr, SKU 115672\n',
 'D912 Katarzyna Mazur, ul. Kwiatowa 21, Wrocław, zgłasza problem z produktem piekarnik Electrolux EOV, termoobieg nie hula, ciasto piecze się tylko z jednej strony, SKU 443901\n',
 'E331 Piotr Zieliński, ul. Leśna 8/1, Gdańsk, zgłasza problem z produktem odkurzacz Dyson V11, odkurzacz pada po 2 min, bateria chyba do wymiany, SKU 900231\n',
 'F678 Maria Wójcik, ul. Słoneczna 14, Poznań, zgłasza problem z produktem płyta indukcyjna Amica PI, jedno pole nie działa, nie chce go „odklikać”, SKU 332118\n',
 'G229 Tomasz Lewandowski, ul. Długa 3/12, Katowice, zgłasza problem z produktem ekspres do kawy DeLonghi Magnifica, cieknie z dołu, cała kuchnia w wodzie, SKU 554872\n',
 'H884 Magdalena Kozłowska, ul. Spacerowa 10, Łódź, zgłasza problem z produktem mikrofalówka LG NeoChef, talerz stoi w miejscu, nie kręci się, SKU 221094\n',
 'I541 Janusz Kamiński, ul. Ogrodowa 44, Szczecin, zgłasza problem z produktem suszarka do ubrań Whirlpool TDU, pranie wychodzi mokre, mimo że program skończony, SKU 667321\n',
 'J110 Ewa Kwiatkowska, ul. Nowa 7/2, Białystok, zgłasza problem z produktem robot kuchenny Kenwood Titanium, strasznie wyje i trzęsie całym blatem, SKU 778403\n',
 'K763 Robert Zając, ul. Parkowa 19, Sosnowiec, zgłasza problem z produktem okap kuchenny Faber, światło nie świeci, pewnie żarówki padły albo coś z kablem, SKU 112005\n',
 'L112 Janusz Nowak, ul. Lipowa 4, Poznań, zgłasza problem z produktem pralka Samsung WW, bębnowy huczy jak samolot przy wirowaniu, SKU 441029\n',
 'M543 Beata Lis-Gąsiorowska, ul. Polna 12/1, Warszawa, zgłasza problem z produktem czajnik Tefal SafeKettle, nie wyłącza się sam, woda wyparowała i prawie spalił grzałkę, SKU 223110\n',
 'N887 Piotr Gąsior, ul. Reymonta 5, Kraków, zgłasza problem z produktem lodówka LG InstaView, ekran na drzwiach zgasł i nie reaguje na nic, SKU 990124\n',
 'O211 Monika Szymańska, ul. Krótka 8, Gdańsk, zgłasza problem z produktem zmywarka Bosch Serie 4, rzuca błędem E24 i nie chce ruszyć z miejsca, SKU 334551\n',
 'P663 Andrzej Kania, ul. Śląska 15, Katowice, zgłasza problem z produktem piekarnik Gorenje, drzwiczki nie domykają się do końca, ucieka ciepło, SKU 118762\n',
 'Q309 Karolina Wilk, ul. Dębowa 3/4, Wrocław, zgłasza problem z produktem odkurzacz Miele C3, stracił ssanie, pewnie gdzieś coś w środku utknęło, SKU 882190\n',
 'Q310 Andrzej Nguyen, pl. 3 Maja 3/4, Kolonia Rozpłucie, zgłasza problem z produktem pralka Samsung WW, wciąż piszczy, SKU 882190\n',
 'R774 Marek Twardy, ul. Jasna 22, Łódź, zgłasza problem z produktem płyta gazowa Whirlpool, jedna palnik nie chce zapalić, tylko syczy, SKU 554103\n',
 'S412 Anna Brzezińska, ul. Kwiatowa 11, Szczecin, zgłasza problem z produktem frytkownica Philips Airfryer, nie grzeje, wiatrak chodzi a jedzenie zostaje zimne, SKU 667811\n',
 'T902 Paweł Nowicki, ul. Leśna 101, Białystok, zgłasza problem z produktem ekspres Jura, kawa leci ledwo co, chyba zapchał się jakiś przewód, SKU 771209\n',
 'U551 Ewa Małecka, ul. Polna 2, Bielsko-Biała, zgłasza problem z produktem mikser Zelmer, silnik śmierdzi spalenizną po 2 minutach ubijania, SKU 110443']
TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
PRODUCT_RE = re.compile(r"(?<=produktem )[^,]+")


def load_jsonl(path):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def token_features(token, prefix):
    return {
        f"{prefix}word": token.lower(),
        f"{prefix}is_title": token.istitle(),
        f"{prefix}is_upper": token.isupper(),
        f"{prefix}is_digit": token.isdigit(),
        f"{prefix}has_digit": any(c.isdigit() for c in token),
        f"{prefix}is_punct": bool(re.fullmatch(r"[^\w\s]", token)),
    }


def word2features(tokens, i):
    word = tokens[i]
    features = {
        "bias": 1.0,
        "word": word.lower(),
        "is_title": word.istitle(),
        "is_upper": word.isupper(),
        "is_digit": word.isdigit(),
        "has_digit": any(c.isdigit() for c in word),
        "is_alnum_mix": any(c.isdigit() for c in word) and any(c.isalpha() for c in word),
        "is_punct": bool(re.fullmatch(r"[^\w\s]", word)),
        "prefix2": word[:2],
        "suffix2": word[-2:],
        "suffix3": word[-3:],
        "length": len(word),
    }
    for offset in (-2, -1, 1, 2):
        j = i + offset
        prefix = f"{offset:+d}:"
        if 0 <= j < len(tokens):
            features.update(token_features(tokens[j], prefix))
        else:
            features[f"{prefix}EDGE"] = True
    return features


def sent2features(tokens):
    return [word2features(tokens, i) for i in range(len(tokens))]


def build_dataset(path):
    X, y = [], []
    for record in load_jsonl(path):
        tokens, labels = record["tokens"], record["labels"]
        assert len(tokens) == len(labels), f"Mismatch: {record.get('text', tokens)}"
        X.append(sent2features(tokens))
        y.append(list(labels))
    return X, y


def extract_products(line, crf):
    spans = [m.span() for m in TOKEN_RE.finditer(line)]
    tokens = [line[s:e] for s, e in spans]
    preds = crf.predict([sent2features(tokens)])[0]
    products = []
    current = None
    for (start, end), pred in zip(spans, preds):
        if pred == "B-PRODUCT":
            if current:
                products.append(current)
            current = [start, end]
        elif pred == "I-PRODUCT" and current:
            current[1] = end
        elif pred == "I-PRODUCT":
            current = [start, end]
        else:
            if current:
                products.append(current)
            current = None
    if current:
        products.append(current)
    return [(line[s:e], s, e) for s, e in products]


X, y = build_dataset("lab3/synthetic_bio_data.jsonl")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

crf = sklearn_crfsuite.CRF(
    algorithm="lbfgs",
    c1=0.1,
    c2=0.1,
    max_iterations=100,
    all_possible_transitions=True,
)
crf.fit(X_train, y_train)

labels = [label for label in crf.classes_ if label != "O"]
print(metrics.flat_classification_report(y_test, crf.predict(X_test), labels=labels, digits=3))

confused = 0
exact = 0
for line in lines:
    text = line.strip()
    header = text.split(",")[0]
    header_end = len(header)
    expected = PRODUCT_RE.search(text)
    products = extract_products(text, crf)
    in_personal_data = [p for p in products if p[1] < header_end]
    confused += bool(in_personal_data)
    exact += bool(expected and any(p[0] == expected.group(0) for p in products))
    print(f"Zgłoszenie: {text[:4]}")
    print(f"Oczekiwany produkt: {expected.group(0) if expected else None}")
    print(f"Wykryte produkty: {[p[0] for p in products]}")
    print(f"Pomylenie z danymi osobowymi: {'tak' if in_personal_data else 'nie'}")

print(f"Zgłoszeń: {len(lines)}")
print(f"Dokładne dopasowanie produktu: {exact}")
print(f"Pomylenia z danymi osobowymi: {confused}")