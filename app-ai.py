import torch;
import matplotlib.pyplot as plt
import torch.nn.functional as F

def stoi(c, alfabet):
    return alfabet.index(c)

def itos(index, alfabet):
    return alfabet[index]

# names, alfabet = load_names()
def load_names() -> tuple[list, list]:
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
    
    return names, alfabet
    
def generate_ai():


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

    g = torch.Generator()
    g.manual_seed(2)
    W = torch.randn(alfabet_len, alfabet_len, generator=g, requires_grad=True)

    for i in range(10000):
        F = X@W
        F = F.exp()
        F = F/F.sum(dim=1, keepdim=True)
        F = F[list(range(F.shape[0])), Y]
        F = -F.log().mean()
        print(F.item())

        W.grad = None
        F.backward()
        W.data += -0.5 * W.grad
        
    torch.save(W, "model-ap.pt")
    
def generate_names():
    W = torch.as_tensor(torch.load("model-ap.pt"))
    alfabet_len = W.shape[0]
    print(alfabet_len)
    
    
    for i in range(10):
        name = ""
        current_index = 0
        while True:
            X = F.one_hot(torch.tensor(current_index) , alfabet_len).float()
            F = X@W
            next_index = torch.multinomial(F, num_samples=1).item()
            
            next_char = itos(next_index)
            if next_char == '.':
                if len(name) < 3:
                    continue

                print(f"\033[32m {name} \033[0m") 
                break

        name += next_char
        current_index = next_index
    
