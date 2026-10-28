from zad2 import Trigram

korpus_2 = [
    "<|s|> <|s|> czarny kot przeskoczył przez brązowy płot <|k|>",
    " <|s|> <|s|> brązowy pies ma biały dom <|k|>",
    " <|s|> <|s|> mój dom ma czarny płot <|k|>",
    " <|s|> <|s|> mój kot ma biały ogon <|k|>",
]

tokenizowane_zdania_k2 = [z.split() for z in korpus_2]
slownik_tokenow = set()

trigram = Trigram()
trigram.przygotuj_model(korpus_2)
for z in tokenizowane_zdania_k2:
    slownik_tokenow.update(z)

print(slownik_tokenow)
V = len(slownik_tokenow)
print(V)


def trigram_laplace(w1, w2, w3):
    t = trigram.czest_trojek.get((w1, w2, w3), 0)
    kontekst = trigram.kontekst.get((w1, w2), 0)
    return (t + 1) / (kontekst + V)


def trigram_k_smoothing(w1, w2, w3, k=0.1):
    t = trigram.czest_trojek.get((w1, w2, w3), 0)
    kontekst = trigram.kontekst.get((w1, w2), 0)
    return (t + 1 * k) / (kontekst + k * V)


print(trigram_laplace("ma", "biały", "dom"))  # wynik: 0.125
print(trigram_laplace("ma", "biały", "ogon"))  # wynik. 0.125
print(trigram_laplace("ma", "biały", "pies"))  # wynik: 0.0625

print(trigram_k_smoothing("ma", "biały", "dom", k=5))
print(trigram_k_smoothing("ma", "biały", "ogon", k=5))
print(trigram_k_smoothing("ma", "biały", "pies", k=5))
