import itertools
from collections import Counter

surowy_tekst = "set new new renew reset renew"

czest_slow = dict(Counter(surowy_tekst.split()))
print(czest_slow)
# wynik: {'set': 1, 'new': 2, 'renew': 2, 'reset': 1}

bpe_slownik = set(surowy_tekst)

bpe_czest_slow = {}
for s, i in czest_slow.items():
    bpe_czest_slow[" ".join(s) + " <|koniec|>"] = i

print(bpe_czest_slow)
# wynik:
# {'s e t <|koniec|>': 1, 'n e w <|koniec|>': 2,
#  'r e n e w <|koniec|>': 2, 'r e s e t <|koniec|>': 1}


def częstość_wystapienia(bpe_czest: dict[str, int]):
    pary: dict[tuple[str, str], int] = {}
    for slowo, czestosc in bpe_czest.items():
        tokeny = slowo.split()
        for para in itertools.pairwise(tokeny):
            pary[para] = pary.get(para, 0) + czestosc
    return pary


czest_par = częstość_wystapienia(bpe_czest_slow)
print(czest_par)
# wynik:
# {('s', 'e'): 2, ('e', 't'): 2, ('t', '<|koniec|>'): 2, ('n', 'e'): 4, ('e', 'w'): 4,
#  ('w', '<|koniec|>'): 4, ('r', 'e'): 3, ('e', 'n'): 2, ('e', 's'): 1}

top = sorted(czest_par.items(), key=lambda x: (-x[1], x[0]))
print(top[0][0])  # wynik: ('e', 'w')

# Utwórz nowy token i dodaj go do obiektu bpe_slownik
nowy_token = "".join(top[0][0])
bpe_slownik.add(nowy_token)
print(bpe_slownik)
# wynik: {'w', 'ew', 'r', 'n', ' ', 'e', 's', 't'}  (kolejność elementów może się różnić)


def scal_pary(para_do_scalenia, bpe_czest: dict[str, int]):
    nowe_bpe_czest = {}
    for slowo, czestosc in bpe_czest.items():
        tokeny = slowo.split()
        nowe_tokeny = []
        i = 0
        while i < len(tokeny):
            if i < len(tokeny) - 1 and (tokeny[i], tokeny[i + 1]) == para_do_scalenia:
                nowe_tokeny.append(tokeny[i] + tokeny[i + 1])
                i += 2
            else:
                nowe_tokeny.append(tokeny[i])
                i += 1
        nowe_bpe_czest[" ".join(nowe_tokeny)] = czestosc
    return nowe_bpe_czest


nowe_bpe_czest = scal_pary(top[0][0], bpe_czest_slow)
print(nowe_bpe_czest)
# wynik:
# {'s e t <|koniec|>': 1, 'n ew <|koniec|>': 2,
#  'r e n ew <|koniec|>': 2, 'r e s e t <|koniec|>': 1}
