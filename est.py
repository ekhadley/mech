#%%
import torch as t
from tqdm import trange
#from mechlibs import endc, red, blue
from transformers import AutoTokenizer, AutoModelForCausalLM

device = t.device('cuda' if t.cuda.is_available() else 'cpu')

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-2.8b")
def tokenize(inp):
    return tokenizer(inp, return_tensors='pt')['input_ids']

#%%

model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-2.8b")

def yap(model, _prompt, ntok=30, show=False):
    out = _prompt
    prompt = model.tokenize(_prompt)['input_ids'].to(device).squeeze()
    for i in range(ntok):
        logits = model.forward(prompt).squeeze()
        nexttok = model.sample(logits)
        prompt = t.cat([prompt, nexttok.unsqueeze(0)], dim=-1)
        out += model.tk.decode(nexttok)
        if show:
            model.tokprint()
            print()
    model.tokprint(prompt)
    return out

#%%

inp = "Hello There!"
toks = tokenize(inp)
out = model(toks).logits

for i in trange(100):
    model(toks).logits

#%%

out