import torch;
import matplotlib.pyplot as plt
import torch.nn.functional as F

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

def itos(index):
    return alfabet[index]

alfabet_len = len(alfabet)

model = torch.zeros(alfabet_len, alfabet_len, dtype=torch.float)

x = []
y = []
for name in names:
    # print(name)
    for i, c in enumerate(name[:-1]):
        x.append(stoi(name[i]))
        y.append(stoi(name[i+1]))

X = F.one_hot(torch.tensor(x), alfabet_len).float()
Y = torch.tensor(y)

# Y = F.one_hot(torch.tensor(y), alfabet_len).float()

g = torch.Generator()
g.manual_seed(2)

W = torch.randn(alfabet_len, alfabet_len, generator=g )

F = X@W
print(F[0])
F = F.exp()
print(F[0])
F = F/F.sum(dim=1, keepdim=True)
print(F[0])

F = F[list(range(F.shape[0])), Y]

print(F)

F = -F.log().sum()

print(F)
