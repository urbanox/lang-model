import torch;
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

alfabet_len = len(alfabet)

model = torch.zeros(alfabet_len, alfabet_len, dtype=torch.int)
print(model[0][0])

for name in names:
    for i, c in enumerate(name[:-1]):
        x = stoi(name[i])
        y = stoi(name[i+1])
        model[x, y] += 1

print(model[5, 5].data)

plt.imshow(model.cpu().numpy().T)

for i in range(alfabet_len):
    for j in range(alfabet_len):
        plt.text(i, j, model[i, j].item(), fontsize=6, ha='center', va='center')
        
plt.xticks(range(alfabet_len), alfabet)
plt.yticks(range(alfabet_len), alfabet)
ax = plt.gca()
ax.xaxis.tick_top()
plt.colorbar()
plt.title("SK mena")
plt.show()

