import itertools
from collections import Counter, defaultdict


class Trigram:
    def __init__(self):
        self.czest_trojek = Counter()
        self.kontekst = Counter()  # liczność bigramów (kontekstów)
        self.prawdopodobienstwa = defaultdict(float)

    @staticmethod
    def _trojki(tokeny):
        return zip(tokeny, tokeny[1:], tokeny[2:])

    @staticmethod
    def _pary(tokeny):
        return itertools.pairwise(tokeny)

    def przygotuj_model(self, korpus):
        for zdanie in korpus:
            tokeny = zdanie.split()
            self.kontekst.update(self._pary(tokeny))
            self.czest_trojek.update(self._trojki(tokeny))

        # P(w3 | w1, w2) = C(w1, w2, w3) / C(w1, w2)
        for (w1, w2, w3), liczba in self.czest_trojek.items():
            self.prawdopodobienstwa[(w1, w2, w3)] = liczba / self.kontekst[(w1, w2)]

    def sekwencja(self, tokeny):
        p = 1.0
        for trojka in self._trojki(tokeny):
            p *= self.prawdopodobienstwa.get(trojka, 0.0)
        print(p)

    def sprawdz_rozklad_prawd(self):
        for w1, w2 in self.kontekst:
            suma = sum(
                p
                for (a, b, _), p in self.prawdopodobienstwa.items()
                if (a, b) == (w1, w2)
            )
            print((w1, w2), suma)


korpus_2 = [
    "<|s|> <|s|> czarny kot przeskoczył przez brązowy płot <|k|>",
    " <|s|> <|s|> brązowy pies ma biały dom <|k|>",
    " <|s|> <|s|> mój dom ma czarny płot <|k|>",
    " <|s|> <|s|> mój kot ma biały ogon <|k|>",
]

przyklad_2 = "<|s|> <|s|> brązowy pies ma biały ogon <|k|>"
trigram = Trigram()
trigram.przygotuj_model(korpus_2)
trigram.sprawdz_rozklad_prawd()
trigram.sekwencja(przyklad_2.split())  # wynik: 0.125
