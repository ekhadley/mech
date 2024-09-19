#%%
from mechlibs import *

device = t.device('cuda' if t.cuda.is_available() else 'cpu')

from ARENA.chapter1_transformer_interp.exercises.monthly_algorithmic_problems.january24_caesar_cipher.model import create_model
from ARENA.chapter1_transformer_interp.exercises.monthly_algorithmic_problems.january24_caesar_cipher.dataset import CodeBreakingDataset

t.manual_seed(42)

#%%

def show(model: HookedTransformer, dataset: CodeBreakingDataset, batch_idx: int, input=None):
    if input is None:
        logits = model(dataset.toks[batch_idx].unsqueeze(0)).squeeze() # [seq_len vocab_out]
    else:
        logits = model(input.unsqueeze(0)).squeeze() # [seq_len vocab_out]

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
# This visualizes the probabilities for a single sequence position, where the label is the probability distn over the true letter, not the shift
# (its the same thing it just lets us see what the model things the word is easier)
def shift_show(model: HookedTransformer, dataset: CodeBreakingDataset, batch_idx: int, seq_pos: int):
    shifted, original, label = ''.join(dataset.str_toks[batch_idx]), ''.join(dataset.str_toks_raw[batch_idx]), dataset[batch_idx][1]
    title = f"'{shifted}' {label}<- '{original}' true letter distn for seq pos {seq_pos}"
    logits = model(dataset.toks[batch_idx].unsqueeze(0)).squeeze() # [seq_len vocab_out]
    probs = logits[seq_pos].softmax(dim=-1) # [vocab_out]
    imshow(
        probs.unsqueeze(0),
        x=[shift(dataset.str_toks[batch_idx][seq_pos], -i) for i in range(26)],
        #x=[i for i in range(26)],
        labels={"x":"Letter"},
        title=title,
        text = [["〇" if i == label else "" for i in range(26)]],
        height=220,
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
n_heads, n_layers = model.cfg.n_heads, model.cfg.n_layers

state_dict = model.center_writing_weights(t.load(filename))
state_dict = model.center_unembed(state_dict)
state_dict = model.fold_layer_norm(state_dict)
state_dict = model.fold_value_biases(state_dict)
model.load_state_dict(state_dict, strict=False);
print(model)

def loss(logits:t.Tensor, labels:t.Tensor):
    logprobs = logits.log_softmax(-1)
    print(red, logits.shape, blue, labels.shape, endc)
    loss = t.nn.functional.cross_entropy(
        einops.rearrange(logprobs, "batch seq vocab_out -> (batch seq) vocab_out"), 
        einops.rearrange(labels, "batch seq -> (batch seq)"),
    )
    return loss

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
avg_loss = -logprobs_correct.mean().item()

print(f"Average cross entropy loss: {-logprobs_correct.mean().item():.3f}")
print(f"Mean probability on correct label: {probs_correct.mean():.3f}")
print(f"Median probability on correct label: {probs_correct.median():.3f}")
print(f"Min probability on correct label: {probs_correct.min():.3f}")

#%%

show(model, dataset, batch_idx=0)
#%%

letter_freqs, first_letter_freqs = t.zeros(26, device=device), t.zeros(26, device=device)
for word in dataset.word_list:
    first_letter_freqs[ord(word[0]) - 97] += dataset.freq_dict[word]
    for letter in word:
        letter_freqs[ord(letter) - 97] += dataset.freq_dict[word]
letter_freqs /= letter_freqs.sum()
first_letter_freqs /= first_letter_freqs.sum()

imshow(
    letter_freqs.unsqueeze(0),
    x=[chr(i + 97) for i in range(26)],
    labels={"x": "Letter"},
    title="distribution of all letters in words from our word list",
    height=220,
)
imshow(
    first_letter_freqs.unsqueeze(0),
    x=[chr(i + 97) for i in range(26)],
    labels={"x": "Letter"},
    title="distribution of first letters in words from our word list",
    height=220,
)

#%%

first_and_second_letter_freqs = t.zeros((26, 26), device=device)
for word in dataset.word_list:
    first_and_second_letter_freqs[ord(word[0]) - 97, ord(word[1]) - 97] += dataset.freq_dict[word]
first_and_second_letter_freqs /= first_and_second_letter_freqs.sum()

imshow(
    first_and_second_letter_freqs,
    x=[chr(i + 97) for i in range(26)],
    y=[chr(i + 97) for i in range(26)],
    labels={"y": "First Letter", "x": "Second Letter"},
    title="distn over possible first letter and second letter combinations",
)

#%%
shift_show(model, dataset, 0, 0)
shift_show(model, dataset, 0, 1)
shift_show(model, dataset, 0, 2)

# so we have some nice graphs. Time to speculate on how a 2l model might accomplish this task.
# We make some sequences of 3 letter words, where the distribution over words matches
# their frequency in 'hitchikers guide to the galaxy'. The text is shifted by some uniformly
# selected amount and then given to the model which attempts to identify the shift amount.
# 
# how would I solve this? Well what is known by me and learnable by the model is the order of
# the letters or in other words the distance between two letters. The goal is to find the
# shift that transforms the text into the highest probability sample possible. For example,
# 'kww' has probably zero probability to be sampled from the word list beacuse it is not a word.
# If we shift this text by -18 we get 'see' which is a relatively common 3 letter word. Plausibly,
# for some peice of text there are multtiple shifts that make the text non-zero  probability. In
# these cases the model should learn to output a distribution of shifts that corresponds to the 
# probability of each post-shifted text being sampled, for each shift.

# When the model has only seen the first letter of the first word, its output distribution simply
# matches the distribution of first letters. For example when given the letter 'k', the model assigns
# ~0.3 probability to the shift being -17 (becoming k->t). 't' is the starting letter of a word about
# 0.3 of the time. It assigns ~0.21 to the shift being 10 (becoming k->a). 'a' is the starting letter
# of a word about 0.21 of the time. And so on.

# This algorithm works for sequences of any length. consider when the model sees the first 2 letters.
# We can construct a 2d distribution of frequencies for first and second letter combinations. The
# probability the model should assign to 'lz' being an n-shift, is the probability of sampling
# '(shift(l, -n)shift(z, -n))'. A 3d distribution can be constructed for the first word and so on.

# The model must be storing frequencies of the combinations somehow. The only other part of this
# task is to figure out the frequency of a given text under a shift. As in not just outputting the
# probability of the current text, but the probability of the text after shifting, for each possible
# shift amount.

# The naive approach of storin n-gram distribution for n-length sequences is highly
# unlikely do to the space required. Potentially the model is basically ignoring most of the sequence
# and is just storing a 3d ngram distn and yoloing the rest. The ngram distributions are highly sparse,
# meaning somehow the model is probably storing a smaller representation. 

# goal: recover some n-gram frequency distributions from the model weights or activations

#%%

def unembed(resid: Float[Tensor, "... seq d_model"]):
    return einops.einsum(resid, model.W_U, '... seq d_model, d_model d_vocab -> ... seq d_vocab') + model.b_U

def loss(logits:t.Tensor, labels:t.Tensor, logprobs:bool=False, mean=True):
    logprobs = logits.log_softmax(-1) if logprobs else logits
    losses = -eindex(logprobs, labels, "batch seq [batch]")
    return losses.mean() if mean else losses

#%%

# early unembeds
checkpoints = ['blocks.0.hook_resid_pre', 'blocks.0.hook_attn_out', 'blocks.0.hook_resid_post', 'blocks.1.hook_resid_pre', 'blocks.1.hook_attn_out', 'blocks.1.hook_resid_post', 'ln_final.hook_normalized']
checkpoint_losses = [loss(unembed(cache[checkpoint]), dataset.labels) for checkpoint in checkpoints]
line(
    checkpoint_losses,
    x=checkpoints,
    labels={"x": "", "y": "Loss"},
    title="losses if unembed is applied at various earlier points"
)
# we have basically no performance at any point but at the very end, after the last layernorm???

#%%
def show_attn_heads(batch_idx, cache=cache):
    batch_indices = [batch_idx] if isinstance(batch_idx, int) else batch_idx
    patterns = t.stack([cache['pattern', i][k, j] for i in range(n_layers) for j in range(n_heads) for k in batch_indices], dim=0)
    return cv.attention.attention_heads(
        patterns,
        attention_head_names=[f"head{i}.{j} on dataset[{k}]" for i in range(n_layers) for j in range(n_heads) for k in batch_indices],
        tokens=dataset.str_toks[batch_indices[0]],
    )

#%%
show_attn_heads(batch_idx=0)


#%%