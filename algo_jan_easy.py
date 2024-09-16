#%%
from mechlibs import *

device = t.device('cuda' if t.cuda.is_available() else 'cpu')

from ARENA.chapter1_transformer_interp.exercises.monthly_algorithmic_problems.january24_caesar_cipher.model import create_model
from ARENA.chapter1_transformer_interp.exercises.monthly_algorithmic_problems.january24_caesar_cipher.dataset import CodeBreakingDataset
#%%

def show(model: HookedTransformer, dataset: CodeBreakingDataset, batch_idx: int):

    logits = model(dataset.toks[batch_idx].unsqueeze(0)).squeeze() # [seq_len vocab_out]
    probs = logits.softmax(dim=-1) # [seq_len vocab_out]

    imshow(
        probs.T,
        y=dataset.vocab_out,
        x=[f"{s}<br><sub>({j})</sub>" for j, s in enumerate(dataset.str_toks[batch_idx])],
        labels={"x": "Token", "y": "Vocab"},
        xaxis_tickangle=0,
        title=f"Sample model probabilities:<br>{''.join(dataset.str_toks[batch_idx])} ({''.join(dataset.str_toks_raw[batch_idx])})",
        text=[
            ["〇" if (s == dataset.str_labels[batch_idx]) else "" for _ in range(seq_len)]
            for s in dataset.vocab_out
        ],
        width=750,
        height=600,
    )

# shifts characters of string by `shift` amount
def shift(s: str, shift: int) -> str:
    return "".join([chr((ord(c) - 97 + shift) % 26 + 97) for c in s])

#%%

dataset = CodeBreakingDataset(mode="easy", size=5, word_list_size=100, seq_len=30, path="hitchhikers.txt")

table = Table("Pre-encoding", "Post-encoding", "Rotation", title="Easy mode")
for i in range(5):
    # Rotation is the thing we're trying to predict; it's stored as a string in `str_labels`
    rotation = int(dataset.str_labels[i])
    # Make a long string explaining the rotation, by showing where `a` and `b` are mapped to
    rotation_explained = f"{rotation:02}: a -> {string.ascii_lowercase[rotation % 26]}, b -> {string.ascii_lowercase[(rotation + 1) % 26]}, ..."
    # Add data to the table
    table.add_row(
        "".join(dataset.str_toks_raw[i]),
        "".join(dataset.str_toks[i]),
        rotation_explained,
    )
rprint(table)

#%%

filename = "C:\\Users\\ekhad\\Desktop\\wgmn\\mech\\ARENA\\chapter1_transformer_interp\\exercises\\monthly_algorithmic_problems\\january24_caesar_cipher\\caesar_cipher_model_easy.pt"
state_dict = t.load(filename)

model = create_model(
    d_vocab=27, # vocab in easy/medium mode is abcd...xyz plus space character
    seq_len=32,
    seed=42,
    d_model=48,
    d_head=24,
    n_layers=2,
    n_heads=2,
    d_mlp=None,
    normalization_type="LN",
    device=device,
)

state_dict = model.center_writing_weights(t.load(filename))
state_dict = model.center_unembed(state_dict)
state_dict = model.fold_layer_norm(state_dict)
state_dict = model.fold_value_biases(state_dict)
model.load_state_dict(state_dict, strict=False);
print(yellow, model, endc)

#%%

seq_len = 32
dataset = CodeBreakingDataset(mode="easy", seq_len=seq_len, size=1000, word_list_size=100, path="hitchhikers.txt").to(device)

#%%

logits, cache = model.run_with_cache(dataset.toks)

logprobs = logits.log_softmax(-1) # [batch seq_len vocab_out]
probs = logprobs.softmax(-1) # [batch seq_len vocab_out]

#%%
# We want to index like `logprobs_correct[batch, seq] = logprobs[batch, seq, labels[batch]]`
logprobs_correct = eindex(logprobs, dataset.labels, "batch seq [batch]")
probs_correct = eindex(probs, dataset.labels, "batch seq [batch]")

print(f"Average cross entropy loss: {-logprobs_correct.mean().item():.3f}")
print(f"Mean probability on correct label: {probs_correct.mean():.3f}")
print(f"Median probability on correct label: {probs_correct.median():.3f}")
print(f"Min probability on correct label: {probs_correct.min():.3f}")

#%%

show(model, dataset, batch_idx=0)
#%%

first_letter_freqs = t.zeros(27, device=device)
for word in dataset.word_list:
    for letter in word:
        first_letter_freqs[ord(letter) - 97] += 1
first_letter_freqs /= first_letter_freqs.sum()

# show frequencies with imshow and letter labels
imshow(
    first_letter_freqs.unsqueeze(0),
    x=[chr(i + 97) for i in range(27)],
    labels={"x": "Letter", "y": "Frequency"},
    title="Frequency of first letters in word list",
    width=500,
    height=300,
)