import itertools
from collections import Counter

from zad2 import Trigram
from zad3 import trigram_k_smoothing, trigram_laplace

korpus_2 = [
    "<|s|> <|s|> czarny kot przeskoczył przez brązowy płot <|k|>",
    " <|s|> <|s|> brązowy pies ma biały dom <|k|>",
    " <|s|> <|s|> mój dom ma czarny płot <|k|>",
    " <|s|> <|s|> mój kot ma biały ogon <|k|>",
]

tokenizowane_zdania_k2 = [z.split() for z in korpus_2]

trigram = Trigram()
trigram.przygotuj_model(korpus_2)

unigramy = Counter(t for z in tokenizowane_zdania_k2 for t in z)
bigramy = Counter(b for z in tokenizowane_zdania_k2 for b in itertools.pairwise(z))
N = sum(unigramy.values())


def stupid_backoff(w1, w2, w3, lam=0.4):
    if (w1, w2, w3) in trigram.czest_trojek:
        return trigram.czest_trojek[(w1, w2, w3)] / trigram.kontekst[(w1, w2)]
    if (w2, w3) in bigramy:
        return lam * bigramy[(w2, w3)] / unigramy[w2]
    return lam**2 * unigramy[w3] / N


def prawd_sekwencji(tekst, smoothing_function):
    tokeny = tekst.split()
    p_zdanie = 1
    for i in range(len(tokeny) - 2):
        w1, w2, w3 = tokeny[i : i + 3]
        p_zdanie *= smoothing_function(w1, w2, w3)

    print(p_zdanie)


z1 = "<|s|> <|s|> brązowy pies ma biały ogon <|k|>"
z2 = "<|s|> <|s|> brązowy pies ma biały kot <|k|>"
prawd_sekwencji(z1, trigram_laplace)  # wynik: 4.389574759945129e-06
prawd_sekwencji(z2, trigram_laplace)  # wynik: 1.175778953556731e-06
prawd_sekwencji(z1, trigram_k_smoothing)  # wynik: 0.002908295070911166
prawd_sekwencji(z2, trigram_k_smoothing)  # wynik: 4.120370820653364e-05
prawd_sekwencji(z1, stupid_backoff)  # wynik: 0.125
prawd_sekwencji(z2, stupid_backoff)  # wynik: 4.70156106519743e-05
