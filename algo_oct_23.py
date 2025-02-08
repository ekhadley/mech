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
model.load_state_dict(state_dict, strict=False);
print(model)

#%%

def show(dataset: SortedListDataset, batch_idx: int):

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

show(dataset, 0)

#%%

print(dataset.str_toks[0])

