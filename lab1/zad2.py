import re


class ProstyTokenizer:
    def __init__(self, slownik) -> None:
        self.token_na_id = slownik
        self.id_na_token = {id: token for token, id in slownik.items()}

    def zakoduj(self, tekst):
        tokeny = re.split(r'([,:;?_!"…().\']|--|\s)', tekst)
        tokeny = [t.strip() for t in tokeny if t.strip()]
        tokeny = [t if t in self.token_na_id else "<|nierozpoznany|>" for t in tokeny]
        return [self.token_na_id[t] for t in tokeny]

    def dekoduj(self, ids):
        tokeny = " ".join(self.id_na_token[i] for i in ids)
        return re.sub(r'\s+([,.?_!"()…\']|--|\s)', r"\1", tokeny)


with open("./lab1/sherlock.txt", "r", encoding="utf-8") as plik:
    tekst = plik.read()
print(len(tekst))

tokeny = re.split(r'([,:;?_!"…().\']|--|\s)', tekst)
tokeny = [t.strip() for t in tokeny if t.strip()]
print(len(tokeny))
print(tokeny[:50])

slowa = sorted(set(tokeny))
slowa = list(slowa)
slowa.extend(["<|koniec|>", "<|nierozpoznany|>"])
print(slowa[-5:])
slownik = {token: id for id, token in enumerate(slowa)}
print(len(slownik))

tokenizer = ProstyTokenizer(slownik)
fragment = (
    "Widzę, że irytował się podczas rozmowy. Serweta jego zmięta i rzucona na podłogę."
)
ids = tokenizer.zakoduj(fragment)
print(ids)
print(tokenizer.dekoduj(ids))

tokenizer = ProstyTokenizer(slownik)
fragment = "Mój przyjaciel otworzył drzwi i potknął się o próg"
ids = tokenizer.zakoduj(fragment)
print(ids) #wynik: [189, 1529, 1294, 675, 783, 2400, 1694, 1185, 2400]
print(tokenizer.dekoduj(ids))
