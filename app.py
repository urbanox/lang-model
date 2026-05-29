import matplotlib.pyplot as plt

names = []

with open("mena.txt", "r", encoding="utf-8") as f:
    for line in f:
        names.append(f".{line.strip().lower()}.")

alfabet = []

for name in names:
    for c in name:
        if not c in alfabet:
            alfabet.append(c)

alfabet.sort()
def stoi(c):
    return alfabet.index(c)

print(alfabet)

model = [[0] * len(alfabet) for _ in range(len(alfabet))]
for name in names:
    for i, c in enumerate(name[:-1]):
        x = stoi(name[i])
        y = stoi(name[i+1])
        print(f"{x}, {y}")
        model[x][y] += 1

print(model)

