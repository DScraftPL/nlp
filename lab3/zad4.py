import json
import re

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

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


def load_jsonl(path):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


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


def build_dataset(jsonl_path):
    X, y = [], []
    for record in load_jsonl(jsonl_path):
        tokens = record["tokens"]
        labels = record["labels"]
        assert len(tokens) == len(labels), f"Mismatch in record: {record['text']}"
        for i in range(len(tokens)):
            X.append(word_features(tokens, i))
            y.append(labels[i])
    return X, y


def anonymize(line, vec, clf):
    spans = [m.span() for m in TOKEN_RE.finditer(line)]
    tokens = [line[s:e] for s, e in spans]
    feats = [word_features(tokens, i) for i in range(len(tokens))]
    preds = clf.predict(vec.transform(feats))
    result = line
    for (start, end), pred in reversed(list(zip(spans, preds))):
        if pred == "PRIV":
            result = result[:start] + "<REDACTED>" + result[end:]
    return result


X, y = build_dataset("lab3/synthetic_training_data.jsonl")

vec = DictVectorizer(sparse=True)
X_vec = vec.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.3, random_state=42, stratify=y
)

clf = LogisticRegression(max_iter=1000, class_weight="balanced")
clf.fit(X_train, y_train)

print(classification_report(y_test, clf.predict(X_test)))

for line in lines:
    print(f"Zgłoszenie: {line.strip()[:4]}")
    print(anonymize(line.strip(), vec, clf))