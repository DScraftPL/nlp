import ollama
import torch

nazwa_modelu = "nomic-embed-text"

słowa = ["king", "queen", "man", "woman", "car"]
wektory_embedding = {}

for s in słowa:
    rezultat = ollama.embed(model=nazwa_modelu, input=s)
    wektory_embedding[s] = torch.tensor([rezultat["embeddings"][0]])

wektor_krol = wektory_embedding["king"]
wektor_krolowa = wektory_embedding["queen"]
wektor_mezczyzna = wektory_embedding["man"]
wektor_kobieta = wektory_embedding["woman"]
wektor_samochod = wektory_embedding["car"]

print(torch.cosine_similarity(wektor_krol, wektor_krolowa))
print(torch.cosine_similarity(wektor_mezczyzna, wektor_kobieta))
print(torch.cosine_similarity(wektor_mezczyzna, wektor_samochod))
print(torch.cosine_similarity(wektor_kobieta, wektor_samochod))
print(
    torch.cosine_similarity(
        wektor_krol - wektor_krolowa, wektor_mezczyzna - wektor_kobieta
    )
)
print(
    torch.cosine_similarity(
        wektor_krol - wektor_mezczyzna + wektor_kobieta, wektor_krolowa
    )
)
