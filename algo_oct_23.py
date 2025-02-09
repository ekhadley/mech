#%%
from mechlibs import *

from ARENA.chapter1_transformer_interp.exercises.monthly_algorithmic_problems.october23_sorted_list.sorted_list_model import create_model
from ARENA.chapter1_transformer_interp.exercises.monthly_algorithmic_problems.october23_sorted_list.dataset import SortedListDataset

device = t.device('cuda' if t.cuda.is_available() else 'cpu')
if not t.cuda.is_available():
    if input(f"{bold+yellow}CUDA was not available. proceed on cpu? [y/n]") == 'n':
        exit()

t.no_grad()
t.set_grad_enabled(False)
t.manual_seed(42)
print(bold, yellow, f"running on device: {device}", endc)

#%%

filename = "/home/ek/wgmn/mech/ARENA/chapter1_transformer_interp/exercises/monthly_algorithmic_problems/october23_sorted_list/sorted_list_model.pt"

model = create_model(
    list_len=10,
    max_value=50,
    seed=0,
    d_model=96,
    d_head=48,
    n_layers=1,
    n_heads=2,
    normalization_type="LN",
    d_mlp=None
)

state_dict = t.load(filename)

state_dict = model.center_writing_weights(t.load(filename))
state_dict = model.center_unembed(state_dict)
state_dict = model.fold_layer_norm(state_dict)
state_dict = model.fold_value_biases(state_dict)
n_layers = 1
n_heads = 1
model.load_state_dict(state_dict, strict=False)
print(model)
print(f"vocab size: {model.W_E.shape[0]}")
print(f"d_model: {model.W_E.shape[-1]}")
print(f"d_head: {model.W_Q.shape[-1]}")
print(f"n_layers: {model.W_Q.shape[0]}")

#%%

N = 1000
dataset = SortedListDataset(size=N, list_len=10, max_value=50)

logits, cache = model.run_with_cache(dataset.toks)
logits: t.Tensor = logits[:, dataset.list_len:-1, :]

targets = dataset.toks[:, dataset.list_len+1:]

logprobs = logits.log_softmax(-1) # [batch seq_len vocab_out]
probs = logprobs.softmax(-1)

batch_size, seq_len = dataset.toks.shape
logprobs_correct = eindex(logprobs, targets, "batch seq [batch seq]")
probs_correct = eindex(probs, targets, "batch seq [batch seq]")

avg_cross_entropy_loss = -logprobs_correct.mean().item()

print(f"Average cross entropy loss: {avg_cross_entropy_loss:.3f}")
print(f"Mean probability on correct label: {probs_correct.mean():.3f}")
print(f"Median probability on correct label: {probs_correct.median():.3f}")
print(f"Min probability on correct label: {probs_correct.min():.3f}")
#%%
def show(batch_idx: int, dataset=dataset):
    logits: Tensor = model(dataset.toks)[:, dataset.list_len:-1, :]
    logprobs = logits.log_softmax(-1) # [batch seq_len vocab_out]
    probs = logprobs.softmax(-1)

    str_targets = dataset.str_toks[batch_idx][dataset.list_len+1: dataset.seq_len]

    imshow(
        probs[batch_idx].T,
        y=dataset.vocab,
        x=[f"{dataset.str_toks[batch_idx][j]}<br><sub>({j})</sub>" for j in range(dataset.list_len+1, dataset.seq_len)],
        labels={"x": "Token", "y": "Vocab"},
        xaxis_tickangle=0,
        title=f"Sample model probabilities:<br>Unsorted = ({','.join(dataset.str_toks[batch_idx][:dataset.list_len])})",
        text=[
            ["〇" if (str_tok == target) else "" for target in str_targets]
            for str_tok in dataset.vocab
        ],
        width=400,
        height=1000,
    )
def list_to_toks(inp: List[Int]) -> t.Tensor:
    s = sorted(inp)
    return t.tensor(inp + [51] + s, dtype=t.int32).unsqueeze(0)

def cache_for_input(inp: List[Int]) -> Dict:
    return model.run_with_cache(list_to_toks(inp))[1]
def to_str_toks(toks):
    return [dataset.vocab[i] for i in toks.squeeze()]

def show_heads_on_input(patterns, tokens=None):
    if isinstance(tokens, t.Tensor): toks = to_str_toks(tokens.squeeze())
    elif isinstance(tokens[0], int): toks = [dataset.vocab[i] for i in tokens]
    else: toks = tokens
    return cv.attention.attention_heads(
        patterns.squeeze(),
        attention_head_names=[f"head{i}.{j}" for i in range(n_layers) for j in range(n_heads)],
        tokens=toks,
    )
def show_probs(logits, toks):
    logprobs = logits.squeeze()[dataset.list_len:-1, :].log_softmax(-1) # [batch seq_len vocab_out]
    probs = logprobs.softmax(-1)
    strtoks = to_str_toks(toks)
    imshow(
        probs.T,
        y=dataset.vocab,
        x=[f"{strtoks[i]}<br><sub>({i})</sub>" for i in range(11, 21)],
        labels={"x": "Token", "y": "Vocab"},
        xaxis_tickangle=0,
        title=f"Sample model probabilities:<br>Unsorted = ({','.join(strtoks[:11])})",
        text=[
            ["〇" if (str_tok == target) else "" for target in strtoks[11:]]
            for str_tok in dataset.vocab
        ],
        width=400,
        height=1000,
    )

# ok so pre-investigation speculation: how might a 1 layer 2 head transformer model learn to sort a list of numbers?
# What do you need to do to sort: you need to make sure that for token x, all the tokens less than x come before it,
# and all the tokens greater than x come after. So we have to know for each pair of numbers which is greater. This is
# an obvious operation to implement in the  QK circuit, as it operates between pairs of tokens. Suppose we are given
# a boolean matrix containing the result of a greater than operator between every pair of elements. how do you go from
# that to a sorted list?
# list = [9, 3, 8, 4]
# m = [[0 1 1 1],
#      [0 0 0 0],
#      [0 1 0 1],
#      [0 1 0 0]]
# the matrix value at index [c, r] tells us "list[r] > list[c]" To go from this to the sorted list,
# we first take a sum along rows to get indices: m.sum(dim=-1) = [3, 0, 2, 1]. so the index of the
# first element (9) is 3. index of the 3 is 0, etc.
#
# The way this model specifically works is that it sees the full list, then a separator token, and is
# trained to predict the tokens that come after the sep token in sorted order.
# lets go step by step. you've just seen the separator token. your prediction should be the first number
# in the sorted list. what do you need to know? for the first number, you need to know the smallest number
# in the input list. This would be the one with the smallest number of true 'greater than' operations
# in the comparison table. Now we predict the second number. What do we need to know? I'd guess that the
# model looks for the smallest number that has not already been output.
#%%
seq_toks = list_to_toks([50, 5, 40, 4, 30, 3, 20, 2, 10, 1])
seq_logits, seq_cache = model.run_with_cache(seq_toks)
show_probs(seq_logits, seq_toks)
show_heads_on_input(seq_cache['pattern', 0], seq_toks)
#%%