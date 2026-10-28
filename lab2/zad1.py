import itertools
from collections import Counter, defaultdict


class Bigram:
    def __init__(self):
        self.czestosc_par = Counter()
        self.czestosc_tokenow = Counter()
        self.prawdopodobienstwa = defaultdict(float)

    def przygotuj_model(self, korpus):
        for zdanie in korpus:
            tokeny = zdanie.split()
            self.czestosc_tokenow.update(tokeny)
            self.czestosc_par.update(itertools.pairwise(tokeny))

        for (w1, w2), liczba in self.czestosc_par.items():
            self.prawdopodobienstwa[(w1, w2)] = liczba / self.czestosc_tokenow[w1]

    def sekwencja(self, zdanie):
        tokeny = zdanie.split()
        p = 1.0
        for para in itertools.pairwise(tokeny):
            p *= self.prawdopodobienstwa[para]
        print(p)

    def sprawdz_rozklad_prawd(self):
        for token in self.czestosc_tokenow:
            suma = sum(
                p for (w1, _), p in self.prawdopodobienstwa.items() if w1 == token
            )
            print(token, suma)


korpus_1 = [
    "<|s|> czarny kot przeskoczył przez brązowy płot <|k|>",
    "<|s|> brązowy pies ma biały dom <|k|>",
    "<|s|> mój dom ma czarny płot <|k|>",
    "<|s|> mój kot ma biały ogon <|k|>",
]


przyklad = "<|s|> brązowy pies ma biały ogon <|k|>"
bigram = Bigram()
bigram.przygotuj_model(korpus_1)  # korpus_1 -> patrz listing 2.1
bigram.sprawdz_rozklad_prawd()
bigram.sekwencja(przyklad)  # przyklad -> patrz listing 2.5 #wynik: 0.0417
