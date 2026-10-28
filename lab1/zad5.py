import tiktoken

TEKST = "Dwudziesty pierwszy października roku dwa tysiące trzydziestego piątek. Miasto Lublin, Czwarta Rzeczpospolita"

for nazwa in ["o200k_base", "cl100k_base", "gpt2"]:
    enc = tiktoken.get_encoding(nazwa)
    tokeny = enc.encode(TEKST)
    print(nazwa)
    for t in tokeny:
        print(t, enc.decode([t]))
    print()
