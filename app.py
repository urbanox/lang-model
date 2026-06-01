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

def itos(index):
    return alfabet[index]

alfabet_len = len(alfabet)

model = torch.zeros(alfabet_len, alfabet_len, dtype=torch.float)

for name in names:
    for i, c in enumerate(name[:-1]):
        x = stoi(name[i])
        y = stoi(name[i+1])
        model[x, y] += 1


def print_image():
    plt.imshow(model.cpu().numpy().T)

    for i in range(alfabet_len):
        for j in range(alfabet_len):
            plt.text(i, j, f"{model[i, j].item():.2f}", fontsize=6, ha='center', va='center')
            
    plt.xticks(range(alfabet_len), alfabet)
    plt.yticks(range(alfabet_len), alfabet)
    ax = plt.gca()
    ax.xaxis.tick_top()
    plt.colorbar()
    plt.title("SK mena")
    plt.show()



for x_index in range(alfabet_len):
    sum = 0
    for y_index in range(alfabet_len):
        sum += model[x_index, y_index].item()

    for y_index in range(alfabet_len):
        model[x_index, y_index] = model[x_index, y_index] / sum

for i in range(10):
    name = ""
    current_index = 0
    while True:
        if not torch.isclose(model[current_index, :].sum(), torch.tensor(1.0)):
            raise "Wrong probability"
        
        next_index = torch.multinomial(model[current_index, :], num_samples=1).item()
        # next_index = torch.multinomial(torch.ones(alfabet_len), num_samples=1).item()
        # print(model[current_index, :])
        # print(f"index: {next_index}, char {itos(next_index)} probability: {model[current_index, next_index]}")
        
        next_char = itos(next_index)
        if next_char == '.':
            if len(name) < 3:
                continue

            print(f"\033[32m {name} \033[0m") 
            break

        name += next_char
        current_index = next_index

# print(model.sum(dim=1, keepdim=True))

# print(model[0])
# print_image()
