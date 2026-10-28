import random

from zad1 import Bigram
from zad2 import Trigram


def generuj_następny_token(
    w1,
    w2,
    trigram: Trigram,
    bigram: Bigram,
    unigram: dict[str, float],
    k_backoff=0.4,
    słownik_wyrazów=None,
):
    kandydaci = {
        W3: p
        for (W1, W2, W3), p in trigram.prawdopodobienstwa.items()
        if (W1, W2) == (w1, w2)
    }
    if kandydaci:
        tokeny, wagi = zip(*kandydaci.items())
    else:
        kandydaci = {
            W3: k_backoff * p
            for (W2, W3), p in bigram.prawdopodobienstwa.items()
            if (W2) == (w2)
        }
        if kandydaci:
            tokeny, wagi = zip(*kandydaci.items())
        else:
            kandydaci = {t: k_backoff**2 * p for t, p in unigram.items()}
            tokeny, wagi = zip(*kandydaci.items())

    return random.choices(tokeny, weights=wagi, k=1)[0]


import pickle
from collections import Counter

korpus = pickle.load(open("lab2/korpus_lalka.pkl", "rb"))  # noqa: SIM115

trigram = Trigram()
trigram.przygotuj_model(korpus)
bigram = Bigram()
bigram.przygotuj_model(korpus)

tokeny = [t for z in korpus for t in z.split()]
unigram = {t: c / len(tokeny) for t, c in Counter(tokeny).items()}

w1, w2, zdanie = "<|s|>", "<|s|>", []
while True:
    w3 = generuj_następny_token(w1, w2, trigram, bigram, unigram)
    if w3 == "<|k|>" or len(zdanie) > 40:
        break
    zdanie.append(w3)
    w1, w2 = w2, w3

print(" ".join(zdanie))
