#%%
from collections import defaultdict
from mechlibs import *

from ARENA.chapter1_transformer_interp.exercises.monthly_algorithmic_problems.january24_caesar_cipher.model import create_model
from ARENA.chapter1_transformer_interp.exercises.monthly_algorithmic_problems.january24_caesar_cipher.dataset import CodeBreakingDataset

device = t.device('cuda' if t.cuda.is_available() else 'cpu')
t.no_grad()
t.set_grad_enabled(False)
t.manual_seed(42)

#%% bunch of utility functions

def ngram_distn(seq, start=0, ret=False):
    assert len(seq) + start <= 3
    out = t.zeros((1, 26), device=device, dtype=t.float)
    for i in range(len(dataset)):
        subi = dataset.stoks[i].find(seq)
        if subi%4 == start:
            out[0, dataset.labels[i]] += 1
    s = out.sum()
    if s == 0: imshow(out, title=f"distn of shifts for dataset examples containing '{seq}' starting at position {start} in any word (0 examples)")
    else: imshow(out/s, title=f"distn of shifts for dataset examples containing '{seq}' starting at position {start} in any word ({int(s)} examples)")
    if ret: return out/s

def ldist(a: str, b: str):
    x = ord(a) - ord(b)
    return x if x >= 0 else x + 26

def summary(dataset_idx: int):
    "prints the string, raw string, tokens, shift"
    print(f"{''.join(dataset.str_toks[dataset_idx])} ({''.join(dataset.str_toks_raw[dataset_idx])}) [{dataset.str_labels[dataset_idx]}]")

def show_logits(logits, dataset_idx: int = None, seq = None, title=None, scale=1.0):
    assert not (dataset_idx is None and seq is None), f"must provide either dataset_idx or seq"
    assert not (dataset_idx is not None and seq is not None), f"cannot provide both dataset_idx and seq"
    seq = seq if seq is not None else dataset.str_toks[dataset_idx]
    logits = logits[dataset_idx] if logits.ndim == 3 else logits
    imshow(
        logits.softmax(dim=-1).T, 
        y=dataset.vocab_out,
        x=[f"{s}<br><sub>{j}</sub>" for j, s in enumerate(seq)],
        labels={"x": "Token", "y": "Shift"},
        xaxis_tickangle=0,
        title=title if title is not None else f"'{dataset.stoksr[dataset_idx]}' shift {dataset.str_labels[dataset_idx]}",
        text=[
            ["〇" if (s == dataset.str_labels[dataset_idx]) else "" for _ in range(len(seq))]
            for s in dataset.vocab_out
        ] if dataset_idx is not None else None,
        width=int(750 * scale),
        height=int(600 * scale),
    )

def search_dataset_for_start(start: str): return [i for i, s in enumerate(dataset.str_toks) if "".join(s).startswith(start)]
def search_dataset_for_start_raw(start: str): return [i for i, s in enumerate(dataset.str_toks_raw) if "".join(s).startswith(start)]
def search_dataset_for_contains(start: str): return [i for i, s in enumerate(dataset.str_toks) if start in "".join(s)]
def search_dataset_for_contains_raw(start: str): return [i for i, s in enumerate(dataset.str_toks_raw) if start in "".join(s)]

def show_probs(probs, dataset_idx: int, title="", scale=1.0):
    probs = probs[dataset_idx] if probs.ndim == 3 else probs
    imshow(
        probs.T, 
        y=dataset.vocab_out,
        x=[f"{s}<br><sub>{j}</sub>" for j, s in enumerate(dataset.str_toks[dataset_idx])],
        labels={"x": "Token", "y": "Shift"},
        xaxis_tickangle=0,
        title=title,
        text=[
            ["〇" if (s == dataset.str_labels[dataset_idx]) else "" for _ in range(seq_len)]
            for s in dataset.vocab_out
        ],
        width=int(750 * scale),
        height=int(600 * scale),
    )
# This visualizes the probabilities for a single sequence position, where the label is the probability distn over the true letter, not the shift
# (its the same thing it just lets us see what the model things the word is easier)
def shift_show(dataset_idx: int, seq_pos: int):
    shifted, original, label = ''.join(dataset.str_toks[dataset_idx]), ''.join(dataset.str_toks_raw[dataset_idx]), dataset[dataset_idx][1]
    title = f"'{shifted}' {label}<- '{original}' true letter distn for seq pos {seq_pos}"
    logits = model(dataset.toks[dataset_idx].unsqueeze(0)).squeeze() # [seq_len vocab_out]
    probs = logits[seq_pos].softmax(dim=-1) # [vocab_out]
    imshow(
        probs.unsqueeze(0),
        x=[f"{shift(dataset.str_toks[dataset_idx][seq_pos], -i)}<br><sub>(-{i})</sub>" for i in range(26)],
        labels={"x":"Letter"},
        title=title,
        text = [["〇" if i == label else "" for i in range(26)]],
        height=220,
    )

# shifts characters of string by `shift` amount
def shift(s: str, shift: int) -> str:
    return "".join([(chr((ord(c) - 97 + shift) % 26 + 97)) if c != " " else " " for c in s ])

def tokenize(s: Union[str, List]) -> t.Tensor:
    if isinstance(s, str): return t.tensor([dataset.vocab.index(i) for i in s], device=device)
    return t.tensor([[dataset.vocab.index(j) for j in i] for i in s], device=device)

def show_mpc_by_seq_pos(inp, title=None):
    if inp.ndim == 3:
        line(
            inp.softmax(-1)[bidx, :, dataset.labels].mean(0),
            x=[f"{c}<sub>{i}</sub>" for i, c in enumerate(dataset.stoks[0])],
            labels={"x": "Sequence Position", "y": "Mean Probability on Correct Label"},
            title="Mean Probability on Correct (MPC) label by Sequence Position" if title is None else title,
            yaxis_range=[0,1]
        )
    elif inp.ndim == 1:
        line(
            inp,
            x=[f"{c}<sub>{i}</sub>" for i, c in enumerate(dataset.stoks[0])],
            labels={"x": "Sequence Position", "y": "Mean Probability on Correct Label"},
            title="Mean Probability on Correct (MPC) label by Sequence Position" if title is None else title,
            yaxis_range=[0,1]
        )
    else: raise ValueError(f"input must be 1 or 3 dimensional, not shape {inp.shape}")
#%% loading in the dataset and making some strings out of all the examples
seq_len = 32
dataset = CodeBreakingDataset(mode="easy", seq_len=seq_len, size=1000, word_list_size=100, path="hitchhikers.txt").to(device)
dataset.stoks = ["".join(seq) for seq in dataset.str_toks]
dataset.stoksr = ["".join(seq) for seq in dataset.str_toks_raw]
bidx = t.arange(len(dataset), device=device)
#%% loading in the model

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
model.load_state_dict(state_dict, strict=False)
print(model)


#%% 
#%%  getting the model's logits and probs for the whole dataset
logits, cache = model.run_with_cache(dataset.toks)
logprobs = logits.log_softmax(-1) # [batch seq_len vocab_out]
probs = logprobs.softmax(-1) # [batch seq_len vocab_out]
#%%
# We want to index like `logprobs_correct[batch, seq] = logprobs[batch, seq, labels[batch]]`
logprobs_correct = eindex(logprobs, dataset.labels, "batch seq [batch]")
probs_correct = eindex(probs, dataset.labels, "batch seq [batch]")
mpc_by_seq_pos = probs_correct.mean(dim=0)
print(f"Average cross entropy loss: {-logprobs_correct.mean().item():.3f}")
print(f"Mean probability on correct label: {probs_correct.mean():.3f}")
print(f"Median probability on correct label: {probs_correct.median():.3f}")
print(f"Min probability on correct label: {probs_correct.min():.3f}")

#%% getting global lettrer frequencies and first letter frequencies
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
#%% getting frequencies for starting word 2-grams

first_and_second_letter_freqs = t.zeros((26, 26), device=device)
for word in dataset.word_list:
    first_and_second_letter_freqs[ord(word[0]) - 97, ord(word[1]) - 97] += dataset.freq_dict[word]
first_and_second_letter_freqs /= first_and_second_letter_freqs.sum()
def show_first_and_second_letter_freqs():
    imshow(
        first_and_second_letter_freqs,
        x=[chr(i + 97) for i in range(26)],
        y=[chr(i + 97) for i in range(26)],
        labels={"y": "First Letter", "x": "Second Letter"},
        title="distn over possible first letter and second letter combinations",
    )
show_first_and_second_letter_freqs()

#%%
# the model's proper output at seq_pos 1 is the distribution over first and second letters that have
# a certain gap between them. (the model knows the gap between letters, not the letters themselves)
# this means selecting a diagonal from the 2d distn above:
def show_2_letter_freq_gap_diagonal(gap: int):
    diag = t.cat([first_and_second_letter_freqs.diagonal(gap), first_and_second_letter_freqs.diagonal(gap-27)], dim=0).unsqueeze(0)
    imshow(
        diag,
    )
show_2_letter_freq_gap_diagonal(ldist('e', 'u'))
# to confirm,
#shift_show(dataset_idx=dataset_idx, seq_pos=1)
# is very similair to:
#show_2_letter_freq_gap_diagonal(12)
#%% the model is assumed to be looking up the shift distribution for various n-grams
# This dictionary gives us the shifts of various literal 1, 2, and 3-grams in the dataset
gram_freqs = defaultdict(list)
for i in range(len(dataset)):
    seq = dataset.stoks[i]
    s = dataset.labels[i].item()
    for j in range(0, seq_len, 4):
        for k in range(1, 4):
            gram_freqs[seq[j:j+k]].append(s)

# these probs represent the correct distribution of the model at each sequence position assuming
# it only uses the largest possible n-gram for each word individually, without looking at the
# rest. basically it represents the model's best possible performance for first words,
# repeated for each word in the sequence as if it was the first word (where best performance
"""
# just means perfectly memorizing and outputting the distribution of the dataset)
gram_probs = t.zeros((len(dataset), seq_len, 26), device=device)
for i in trange(len(dataset)):
    seq = dataset.stoks[i]
    for j in range(0, seq_len, 4):
        for k in range(1, 4):
            d = t.zeros((26), device=device, dtype=t.float)
            for s in gram_freqs[seq[j:j+k]]:
                d[s] += 1
            gram_probs[i, j+k-1] = d/d.sum()
        gram_probs[i, j+3] = d/d.sum()
show_probs(gram_probs, 0, title="model probabilities if it only used the largest possible n-gram for each word")
t.save(gram_probs, "gram_probs.pt")
"""
gram_probs = t.load("gram_probs.pt")

# Our MPC on first letters is 0.2
# MPC on second letters (first and second letter bigram) is 0.78
# MPC on third letters (first, second, and third letter trigram) is 1.0 meaning each 3 letter literal in the dataset comes from only a single shift of a single word.

#%% # checking for periodicity. no periodicity in the embeddings, posembeddings, or attn W_O

def make_fourier_basis(p: int) -> Tuple[Tensor, List[str]]:
    x = t.arange(p, device=device)
    F = t.ones((p, p), device=device)
    for row in range(1, p//2 + 1):
        freqs = x*2*t.pi*row / p
        F[row*2 - 1] = t.cos(freqs)
        F[row*2] = t.sin(freqs)
    F /= F.norm(dim=1, keepdim=True)
    return F
fourier_basis = make_fourier_basis(49)[:-1, :-1]
imshow(einops.einsum(model.W_E, fourier_basis, "... p2, p2 p1 -> ... p1"), title="embeddings in fourier basis. not sparse = not periodic")

#%% functions for unembedding and finding loss

def detokenize(toks: t.Tensor) -> str:
    if toks.ndim == 1: return "".join([dataset.vocab[i] for i in toks])
    else: return [detokenize(i) for i in toks]

def unembed(resid: Float[Tensor, "... seq d_model"], ln=True):
    #if ln: return einops.einsum(model.ln_final(resid), model.W_U, '... seq d_model, d_model d_vocab -> ... seq d_vocab') + model.b_U
    #return einops.einsum(resid, model.W_U, '... seq d_model, d_model d_vocab -> ... seq d_vocab') + model.b_U
    if ln: return model.unembed(model.ln_final(resid))
    return model.unembed(resid)

def loss(logits:t.Tensor, labels:t.Tensor, logprobs:bool=False, mean=True):
    logprobs = logits.log_softmax(-1) if logprobs else logits
    losses = -eindex(logprobs, labels, "batch seq [batch]")
    return losses.mean() if mean else losses

#%% testing performance of unembedding at various early points
checkpoints = ['blocks.0.hook_resid_pre', 'blocks.0.hook_attn_out', 'blocks.0.hook_resid_post', 'blocks.1.hook_resid_pre', 'blocks.1.hook_attn_out', 'blocks.1.hook_resid_post', 'ln_final.hook_normalized']
line(
    [loss(unembed(cache[checkpoint], ln=True), dataset.labels) for checkpoint in checkpoints],
    x=checkpoints,
    labels={"x": "", "y": "Loss"},
    title="losses if unembed is applied at various earlier points"
)
# the layernorm performs a large scale on the residual stream, which is necessary to see the logit impacts.
# So, it turns out that, unmysteriously, the model's real work is contained in the output of the second
# attn, (not output + resid) and the final ln's job is essentially to scale the output
# stream up by a few ~2 ooms.
#%% functions for getting attention patterns

_all_attn_patterns = t.cat((cache['pattern', 0], cache['pattern', 1]), dim=1)
def get_all_attn_patterns(cache=None):
    if cache is None: return _all_attn_patterns
    return t.cat((cache['pattern', 0], cache['pattern', 1]), dim=1)

def show_attn_heads(cache, dataset_idx):
    patterns = get_all_attn_patterns(cache=cache)[dataset_idx]
    return cv.attention.attention_heads(
        patterns,
        attention_head_names=[f"head{i}.{j}" for i in range(n_layers) for j in range(n_heads)],
        tokens=dataset.str_toks[dataset_idx],
    )
def sah(cache, dataset_idx):
    return show_attn_heads(cache, dataset_idx)

#%% avg attention patterns over whole dataset
avg_patterns = get_all_attn_patterns().mean(dim=0)
cv.attention.attention_heads(
    avg_patterns,
    attention_head_names=[f"head{i}.{j}" for i in range(n_layers) for j in range(n_heads)],
    tokens=list(dataset.stoksr[0]),
)
#%% shows the average attention probability for each source position. That is, how strongly each position is attended to on average across the whole dataset
lines(
    get_all_attn_patterns().mean(dim=(0, 2)),
    labels=[f"head{i}.{j}" for i in range(2) for j in range(2)],
    x=[f"{s}<br><sub>{j}</sub>" for j, s in enumerate(dataset.str_toks[0])],
    title="How strongly is each token attended to on average?\n(how much do tokens pay attn to this seq pos)",
)

#%% shows the average attention probability for each destination position when the input sequence is all spaces

def show_patterns_for_input(input: Union[str, t.Tensor]):
    if isinstance(input, str): input = t.tensor([dataset.vocab.index(i) for i in input], device=device)
    input_logits, input_cache = model.run_with_cache(empty_seq.unsqueeze(0))
    input_patterns = t.stack([input_cache['pattern', i][0,j] for i in range(n_layers) for j in range(n_heads)], dim=0)
    return cv.attention.attention_heads(
        input_patterns,
        attention_head_names=[f"head{i}.{j}" for i in range(n_layers) for j in range(n_heads)],
        tokens=[str(i) for i in range(seq_len)],
    )

empty_seq = t.tensor([26]*seq_len)
_, empty_seq_cache = model.run_with_cache(empty_seq.unsqueeze(0))
lines(
    get_all_attn_patterns(cache=empty_seq_cache).mean(dim=(0,2)),
    labels=[f"head{i}.{j}" for i in range(2) for j in range(2)],
    x=[f"{s}<br><sub>{j}</sub>" for j, s in enumerate(dataset.str_toks[0])],
    title="dest average of token positions on sequence of all spaces. (0.0, 0.1 same, 1.0, 1.1 cooked?)",
)
# we see that the attn0 heads attent to the normal positions, but attn1 heads have toally uniform pattern. They arent looking at specific positions, only for information that attn0 puts in.
#%% some functions for patching

def replace_act_hook(orig_act: Float[t.Tensor, "batch seq nhead dmodel"], hook: HookPoint, new_act: Float[t.Tensor, "batch seq dmodel"], seq_pos: int = None):
    if seq_pos is not None: orig_act[:, seq_pos] = new_act
    else: orig_act = new_act
    return orig_act

def replace_attn_result_hook(orig_attn_result: Float[t.Tensor, "batch seq nhead dmodel"], hook: HookPoint, new_attn_result: Float[t.Tensor, "batch seq nhead dmodel"], head: int = None):
    if head is not None: orig_attn_result[:, :, head] = new_attn_result
    else: orig_attn_result = new_attn_result
    return orig_attn_result


# def word_patch
def seq_patch(dest_word: str, src_word: str, dest_shift: int, src_shift: int, seq_pos_dest: int, seq_pos_src:int, hookpoint: str, scale=1.0, show_original=True, show=True):
    """this function will take the residual value at hookpoint during the forward pass of src_word (shifted by dest_shift) and patch it into the same position during the forward pass of word2 (shifted by src_shift)"""   
    model.reset_hooks()
    src_word_shifted, dest_word_shifted = shift(src_word, src_shift), shift(dest_word, dest_shift)
    src_word_toks, dest_word_toks = tokenize(src_word_shifted), tokenize(dest_word_shifted)

    src_word_logits, src_word_cache = model.run_with_cache(src_word_toks)
    src_word_resid = src_word_cache[hookpoint][0, seq_pos_src]
    dest_word_logits = model(dest_word_toks).squeeze()
    
    hook = partial(replace_act_hook, new_act=src_word_resid, seq_pos=seq_pos_dest)
    patched_logits = model.run_with_hooks(dest_word_toks, fwd_hooks=[(hookpoint, hook)]).squeeze()
    if show:
        if show_original:
            show_logits(dest_word_logits, seq=dest_word_shifted, title=f"original logits for '{dest_word}'->{dest_shift}='{dest_word_shifted}'", scale=scale)
            show_logits(src_word_logits.squeeze(), seq=src_word_shifted, title=f"original logits for '{src_word}'->{src_shift}='{src_word_shifted}'", scale=scale)
        show_logits(patched_logits, seq=dest_word_shifted, title=f"patching at {hookpoint}<br>from '{src_word_shifted}'[{seq_pos_src}] into '{dest_word_shifted}'[{seq_pos_dest}]", scale=scale)
    return patched_logits

def vec_patch(dest_word: str, dest_shift: int, seq_pos_dest: int, vec: Tensor, hookpoint: str, scale=1.0, show_original=True, show=True):
    """this function will take the residual value at hookpoint during the forward pass of src_word (shifted by dest_shift) and patch it into the same position during the forward pass of word2 (shifted by src_shift)"""   
    model.reset_hooks()
    dest_word_shifted = shift(dest_word, dest_shift)
    dest_word_toks = tokenize(dest_word_shifted)

    dest_word_logits = model(dest_word_toks).squeeze()
    hook = partial(replace_act_hook, new_act=vec, seq_pos=seq_pos_dest)
    patched_logits = model.run_with_hooks(dest_word_toks, fwd_hooks=[(hookpoint, hook)]).squeeze()
    if show:
        if show_original:
            show_logits(dest_word_logits, seq=dest_word_shifted, title=f"original logits for '{dest_word}'->{dest_shift}='{dest_word_shifted}'", scale=scale)
        show_logits(patched_logits, seq=dest_word_shifted, title=f"patching at {hookpoint} into '{dest_word_shifted}'[{seq_pos_dest}]", scale=scale)
    return patched_logits

def random_words(n_seqs, n_words, spaces=True):
    assert n_words >= 1
    words = t.randint(26, (n_seqs, n_words*4), device=device)
    if spaces:
        for i in range(n_words): words[:, i*4-1] = 26
    return words
#%% how well does the model perform if we get rid of embeddings from the residual stream after attn0?
def fuck_with_embeddings(act):
    model.reset_hooks()
    no_emb_probs = model.run_with_hooks(dataset.toks, fwd_hooks=[(act, partial(replace_act_hook, new_act=cache[act] - cache["hook_embed"]))]).softmax(-1)
    random_seq_toks = random_words(1, n_words=8)
    rand_emb_probs = model.run_with_hooks(dataset.toks, fwd_hooks=[(act, partial(replace_act_hook, new_act=cache[act] - cache["hook_embed"] + model.embed(random_seq_toks)))]).softmax(-1)
    print(f"mean prob on correct with embeddings subtracted: {no_emb_probs[bidx, :, dataset.labels].mean():.3f}")
    print(f"mean prob on correct with random embeddings in all (non-space) positions: {rand_emb_probs[bidx, :, dataset.labels].mean():.3f}")
    model.reset_hooks()
    return lines(
        [mpc_by_seq_pos, rand_emb_probs[bidx, :, dataset.labels].mean(0), no_emb_probs[bidx, :, dataset.labels].mean(0)],
        x=[f"{s}<br><sub>{i}</sub>" for i, s in enumerate(dataset.str_toks[0])],
        title=f"MPC by sequence position when embedding directions are modified at '{act}'",
        labels=["ablation", "scrambled embeddings", "subtracted embeddings"]
    )
fuck_with_embeddings("blocks.0.hook_resid_pre") # 0 performance. duh
fuck_with_embeddings("blocks.0.hook_resid_post") # bad performance. about 0.5 avg over all posn, but rises over the sequence
fuck_with_embeddings("blocks.1.hook_resid_post") # perfect performance, scrambling or subtracting
# The takeaway is that embeddings are important to attn1 and totally unecessary to the unembed.

# we see that in general
#%% here we plot the similarity between different positional embeddings (which are normal learned embeddings with this model)
pos_sims = t.zeros((seq_len, seq_len), device=device)
for i in range(seq_len):
    for j in range(seq_len):
        pos1, pos2 = model.pos_embed.W_pos[i], model.pos_embed.W_pos[j]
        npos1 = (pos1 - pos1.mean()) / pos1.norm()
        npos2 = (pos2 - pos2.mean()) / pos2.norm()
        pos_sims[i, j] = einops.einsum(npos1, npos2, 'p, p ->')
imshow(
    pos_sims,
    title="mc+norm cosine similarity between position embeddings",
    x=[f"{c}<sub>{i}</sub>" for i, c in enumerate(dataset.stoksr[0])],
    y=[f"{c}<sub>{i}</sub>" for i, c in enumerate(dataset.stoksr[0])]
)
# this is an interesting pattern!
# we see that all the (nonspace) potisional embeddings on and after the third word are very similair to each othe.
# The spaces are less similair to teh characters but very similair to eachother.
# have negative sim with the first word and ~0 sim with the second word. It seems like the first and second words get
# their own spaces, and the rest of the words occuppy the same position space.
# Also there is relatively high sim at every sequences position with the positions 4 before and after, decaying with each additional 4
# idk actually if this tells us anything important but its interesting.
# I did the same thing with the normal token embeddings and it was uninteresting. noisy, around ~0.4 average sim, and uniformly positive.
#%% this does the same as above but works with any length sequence, and uses random prefixes
hookpoint = "blocks.0.hook_attn_out"
dest_seq = "his"
seq_pos = 2

n_random_ctx = 250
random_prefixes = random_words(n_random_ctx, n_words=dest_seq.count(" ") + 1)
random_prefixes[:, seq_pos] = dataset.vocab.index(dest_seq[seq_pos])
random_seq_logits, random_seq_cache = model.run_with_cache(random_prefixes)

dest_seq_toks = tokenize(dest_seq).repeat(n_random_ctx, 1)
dest_seq_logits, dest_seq_cache = model.run_with_cache(dest_seq_toks)

hook = partial(replace_act_hook, new_act=random_seq_cache[hookpoint][:, seq_pos], seq_pos=seq_pos)
random_seq_patched_logits = model.run_with_hooks(dest_seq_toks, fwd_hooks=[(hookpoint, hook)])
show_logits(dest_seq_logits[0], seq=dest_seq, title=f"original logits for '{dest_seq}'")
print(f"mean prob on 0 after patching: {random_seq_patched_logits.softmax(-1)[:, seq_pos, 0].mean().item():.3f}")
# in both experiments, we see that (for the last letter of a 1 word input), this patch is a large performance hit, about halving performance.
# it seems that attn0 is moving some information about surrounding letters to each sequence position.
i = 11
show_logits(random_seq_patched_logits[i], seq=dest_seq, title=f"patching {hookpoint}<br>from '{detokenize(random_prefixes[i])}'[{seq_pos}] into '{dest_seq}'[{seq_pos}]")
#%% # define some functions for replacing and masking attention patterns and scores
def replace_attn_scores_hook(orig_attn_scores: Tensor, hook: HookPoint, new_attn_scores: Tensor, head: int=None):
    if head is not None: orig_attn_scores[:, head] = new_attn_scores
    else: orig_attn_scores = new_attn_scores
    return orig_attn_scores

def mask_attn_scores_hook(orig_attn_scores: Tensor, mask: Tensor, hook: HookPoint, head: int=None):
    if head is not None:
        orig_attn_scores[:, head, mask] = -t.inf
    else: orig_attn_scores[:, :, mask] = -t.inf
    return orig_attn_scores

def mask_attn_pattern_hook(orig_pattern: Tensor, hook: HookPoint, mask: Tensor, head: int=None):
    if head is not None: orig_pattern[:, head] = orig_pattern[:, head] * mask
    else: orig_pattern *= mask
    return orig_pattern

def get_masked_attn_pattern_logits(mask, layer, head=None):
    hook = partial(mask_attn_pattern_hook, mask=mask, head=head)
    model.reset_hooks()
    model.add_hook(f"blocks.{layer}.attn.hook_pattern", hook)
    masked_logits, masked_cache = model.run_with_cache(dataset.toks)
    model.reset_hooks()
    return masked_logits, masked_cache

def get_masked_attn_scores_logits(mask, layer, head=None):
    hook = partial(mask_attn_scores_hook, mask=mask, head=head)
    model.reset_hooks()
    model.add_hook(f"blocks.{layer}.attn.hook_attn_scores", hook)
    masked_logits, masked_cache = model.run_with_cache(dataset.toks)
    model.reset_hooks()
    return masked_logits, masked_cache

def get_replaced_attn_scores_logits(new_attn_scores, layer, head=None):
    assert new_attn_scores.ndim > 2, "new_attn_scores must have a batch, seqq, and seqk dim"
    if head is None: assert new_attn_scores.ndim == 4, f"if not providing a head, new_attn_scores must have a head dimension"
    else: assert new_attn_scores.ndim == 3, "if providing a head, new_attn_scores must not have a head dimension"
    hook = partial(replace_attn_scores_hook, new_attn_scores=new_attn_scores, head=head)
    model.reset_hooks()
    model.add_hook(f"blocks.{layer}.attn.hook_attn_scores", hook)
    replaced_logits, replaced_cache = model.run_with_cache(dataset.toks)
    model.reset_hooks()
    return replaced_logits, replaced_cache

def replace_attn_pattern_hook(orig_attn_pattern: Tensor, hook: HookPoint, new_attn_pattern: Tensor, head: int=None):
    if new_attn_pattern.ndim == 2: new_attn_pattern = new_attn_pattern.unsqueeze(0)
    if head is not None: orig_attn_pattern[:, head] = new_attn_pattern
    else: orig_attn_pattern[:] = new_attn_pattern
    return orig_attn_pattern

#%% ablating the attention pattern of certain heads to be an identity matrix so that the head acts like mlp
dataset_idx = 219
eye = t.eye(seq_len, device=device, dtype=t.int32)
attn_ident_probs, attn_ident_caches = [], []
for layer in range(2):
    for head in range(2):
        hook = partial(replace_attn_pattern_hook, new_attn_pattern=eye, head=head)
        model.reset_hooks()
        model.add_hook(f"blocks.{layer}.attn.hook_pattern", hook)
        attn_ident_logits, attn_ident_cache = model.run_with_cache(dataset.toks)
        model.reset_hooks()
        attn_ident_probs.append(attn_ident_logits.softmax(-1))
        attn_ident_caches.append(attn_ident_cache)
        show_logits(attn_ident_logits, dataset_idx, title=f"performance when h{layer}.{head} pattern are identity")

# with both attn1 heads being identity:
model.reset_hooks()
model.add_hook(f"blocks.{1}.attn.hook_pattern", partial(replace_attn_pattern_hook, new_attn_pattern=eye, head=None))
attn_ident_logits, attn_ident_cache = model.run_with_cache(dataset.toks)
model.reset_hooks()
attn_ident_probs.append(attn_ident_logits.softmax(-1))
attn_ident_caches.append(attn_ident_cache)

#%% summarizing attn identity performance
_ = [print(f"head {l}.{h} identity MPC: {attn_ident_probs[l*2 + h][bidx, :, dataset.labels].mean():.3f}") for l in range(2) for h in range(2)]
print(f"h1.0 and h1.1 identity MPC: {attn_ident_probs[-1][bidx, :, dataset.labels].mean():.3f}")
attn_ident_mpc_by_seq_pos = [mpc_by_seq_pos] + [probs[bidx, :, dataset.labels].mean(0) for probs in attn_ident_probs]
lines( # MPC by sequence position when each head's attention pattern is identity matrix 
    attn_ident_mpc_by_seq_pos,
    labels=["unablated"] + [f"head{i}.{j}" for i in range(2) for j in range(2)] + ["h1.0 and h1.1"],
    x=[f"{s}<br><sub>{j}</sub>" for j, s in enumerate(dataset.str_toks[0])],
    title="MPC by sequence position while ablating each heads's attention pattern to an identity",
)
attn1_ident_2nd_and_3rd_letter_mpc = (attn_ident_mpc_by_seq_pos[-1][1:32:4] + attn_ident_mpc_by_seq_pos[-1][2:32:4]).mean() * 0.5
print(f"(mpc for second and last letters when both attn1 patterns are identity: {attn1_ident_2nd_and_3rd_letter_mpc.item():.3f}")
#%% we try masking the attention pattern of various heads to be only intra-word (between letters of the same word)
intraword_mask = t.zeros((seq_len, seq_len), device=device, dtype=t.bool)
for i in range(0, seq_len, 4):
    intraword_mask[i:i+4, i:i+4] = 1
intraword_mask[*t.triu_indices(*intraword_mask.shape, offset=1)] = 1

attn0_intraword_scores = cache['attn_scores', 0].clone()
for i in range(0, seq_len, 4): attn0_intraword_scores[:, :, ~intraword_mask] = -t.inf
attn1_intraword_scores = cache['attn_scores', 1].clone()
for i in range(0, seq_len, 4): attn1_intraword_scores[:, :, ~intraword_mask] = -t.inf

intraword_logits, intraword_caches = [], []
for layer in range(2):
    for head in range(2):
        if layer: intraword_logit, intraword_cache = get_replaced_attn_scores_logits(attn1_intraword_scores[:, head], layer=layer, head=head)
        else: intraword_logit, intraword_cache = get_replaced_attn_scores_logits(attn0_intraword_scores[:, head], layer=layer, head=head)
        intraword_logits.append(intraword_logit)
        intraword_caches.append(intraword_cache)
attn0_intraword_logits, attn0_intraword_cache = get_masked_attn_scores_logits(~intraword_mask, layer=0)
attn0_intraword_mpc_bsp = attn0_intraword_logits.softmax(-1)[bidx, :, dataset.labels].mean(0)
attn1_intraword_logits, attn1_intraword_cache = get_masked_attn_scores_logits(~intraword_mask, layer=1)
attn1_intraword_mpc_bsp = attn1_intraword_logits.softmax(-1)[bidx, :, dataset.labels].mean(0)
lines(
    [attn0_intraword_mpc_bsp, attn1_intraword_mpc_bsp],
    x=[f"{s}<br><sub>{i}</sub>" for i, s in enumerate(dataset.str_toks[0])],
    title="MPC by sequence position when attn layers have intraword patterns",
    labels=["h0.0+h0.1", "h1.0+h1.1"]
)
# we see that the model does well with attn0 intraword ablated, except on first letters whihch is expected.
# when attn1 is intraword, e see the model gets normal first word performance for the first 3 words then it
# degrades. This is very likely due to the similarity of sequence position embeddings later in the sequence,
# which was shown abnove. makes sense.
    
#%% here we make it so that the attn1 heads src from two selected sequence positions

upper_indices = t.triu_indices(seq_len, seq_len, offset=1, device=device)

p1, p2 = 1, 6

new_attn1_scores = cache['attn_scores', 1].clone()
for example in new_attn1_scores:
    h0_scores, h1_scores = example[0], example[1]
    
    #h0_scores[:, 1:32:4] = -t.inf
    #h0_scores.fill_(-t.inf)
    #h0_scores.diagonal(0).fill_(1)
    #h0_scores[:, p1] = 1
    #h0_scores[:, p2] = 1
 
    h1_scores[:, 2:32:4] = -t.inf
    #h1_scores.fill_(-t.inf)
    #h1_scores.diagonal(0).fill_(1)
    #h1_scores[:, p1] = 1
    #h1_scores[:, p2] = 1

    h0_scores[*upper_indices] = -t.inf # ensure above-diagonal scores remain -inf
    h1_scores[*upper_indices] = -t.inf

model.reset_hooks()
model.add_hook(f"blocks.1.attn.hook_attn_scores", partial(replace_attn_scores_hook, new_attn_scores=new_attn1_scores))
model.add_hook("blocks.1.attn.hook_attn_scores", partial(mask_attn_scores_hook, mask=~intraword_mask))
apl, apc = model.run_with_cache(dataset.toks)
app = apl.softmax(-1)
model.reset_hooks()

#dataset_idx = 25
#show_logits(apl, dataset_idx, title=f"performance on [{dataset_idx}] when h1.1 is identity and h1.1 can only attend to seq pos {p1} and {p2}")
apl_mpc = app[bidx, :, dataset.labels].mean(0)
lines(
    [mpc_by_seq_pos, attn1_intraword_mpc_bsp, apl_mpc],
    x=[f"{s}<br><sub>{i}</sub>" for i, s in enumerate(dataset.str_toks[0])],
    title="MPC when attn1 heads attend to two selected sequence positions",
    labels=["original", "just intraword", "ablated"]
)

#%% all attn heads intraword only

model.reset_hooks()
model.add_hook(f"blocks.0.attn.hook_attn_scores", partial(replace_attn_scores_hook, new_attn_scores=attn0_intraword_scores, head=None))
model.add_hook(f"blocks.1.attn.hook_attn_scores", partial(replace_attn_scores_hook, new_attn_scores=attn1_intraword_scores, head=None))
all_intra_logits, all_intra_cache = model.run_with_cache(dataset.toks)
model.reset_hooks()

show_mpc_by_seq_pos(all_intra_logits, title="MPC by sequence position when all heads are intraword")

# %% here we scramble the values of various activations along the batch dimension at particular sequence positions
randperm = t.randperm(1000, device=device)
dest_letter_idx = 0
src_letter_idx = 0

head = None
act = f"blocks.1.attn.hook_v"

scrambled_act = cache[act].clone()
if head is not None: scrambled_act[:, dest_letter_idx:32:4, head] = scrambled_act[randperm, src_letter_idx:32:4, head]
else: scrambled_act[:, dest_letter_idx:32:4] = scrambled_act[randperm, src_letter_idx:32:4]
#scrambled_act[:, dest_letter_idx, head] = scrambled_act[randperm, src_letter_idx, head]

model.reset_hooks()
model.add_hook(act, partial(replace_act_hook, new_act=scrambled_act))
model.add_hook("blocks.1.attn.hook_attn_scores", partial(mask_attn_scores_hook, mask=~intraword_mask))
scrambled_logits, scrambled_cache = model.run_with_cache(dataset.toks)
model.reset_hooks()

scrambled_probs = scrambled_logits.softmax(-1)
scrambled_mpc_bsp = scrambled_probs[bidx, :, dataset.labels].mean(0)
scrambled_mpc = scrambled_mpc_bsp.mean()

lines(
    [mpc_by_seq_pos, attn1_intraword_mpc_bsp, scrambled_mpc_bsp],
    x=[f"{s}<br><sub>{i}</sub>" for i, s in enumerate(dataset.str_toks[0])],
    title=f"MPC by seq pos when scrambling h1.{head} '{act}'<br>src: {src_letter_idx}, dest: {dest_letter_idx}",
    labels=["original", "just intraword", "ablated"]
)

#%% This shows the average attention scores for normal passes vs the scrambled passes. it demonstrates better than attn patterns what the model is 'wanting' to attend to.
imshow(cache['attn_scores', 0][:, 0].mean(0), title="average h0.0 scores", margin=30)
imshow(cache['attn_scores', 0][:, 1].mean(0), title="average h0.1 scores", margin=30)
imshow(cache['attn_scores', 1][:, 0].mean(0), title="average h1.0 scores", margin=30)
imshow(cache['attn_scores', 1][:, 1].mean(0), title="average h1.1 scores", margin=30)
imshow(scrambled_cache['attn_scores', 1][:, 0].mean(0), title="average scrambled h1.0 scores", margin=30)
imshow(scrambled_cache['attn_scores', 1][:, 1].mean(0), title="average scrambled h1.1 scores", margin=30)
# summary/conclusions:
# all positions (including spaces), excepting only the very first letter of the input, have the same query, for both heads.
# the first position has a different query to other positions, but not to other first letters.
# Scrambling or scrambling+swapping the keys has moderate impact for head 0 (~0.15), and small impact for head 1 (~0.06).
# Basically each head attends to all types of letters in the same way. h1.0 attends mostly uniformly but with an emphasis
# on last letters and the very first letter of the sequence. h1.1 attends away from spaces and first letters, and attends
# to all second and last letters, with an emphasis on the second. The keys are just saying "i am a second letter", "i am a
# space", etc, and the queries of each head just say "attend this much to every second letter", "attend this much to every
# space", etc and the queries dont vary position to position becuase all positions should attend to this type of other
# position in the same way.
# The next thing to understand is the values. 

# %%
dest_word, src_word = "th", "yh"
dest_pos, src_pos = 1, 1
scores = t.zeros(26, device=device)
for s in range(26):
    plogits = seq_patch(dest_word, src_word, s, s, dest_pos, src_pos, "blocks.1.attn.hook_v", scale=0.8, show=False)
    pprobs = plogits.softmax(-1)
    scores[s] = pprobs[-1, s].item()
imshow(pprobs.T, title=f"patching '{src_word}'[{src_pos}]'->{s}='{shift(src_word, s)}' into '{dest_word}'[{dest_pos}]->{s}='{shift(dest_word, s)}' at 'blocks.1.attn.hook_v'")

print(scores.mean())
#%% def ctx_sens3(...)
# in order to understand how to interpret the value vectors in attn1, we can scramble or swap vectors
# for a particular sequence position, then look at how this changes the logits for a different seq pos.
def ctx_sens3(dest_seq, hookpoint, src_pos, dest_pos, matching_swapped=True):
    ctx_words, ctx_labels_l = [], []
    for word in dataset.word_list: # only look at letter pairs if given a letter pair
        for s in range(26):
            sword = shift(word, s)
            if not matching_swapped or sword[src_pos] == dest_seq[dest_pos]:
                ctx_words.append(sword)
                ctx_labels_l.append(s)
    ctx_toks = tokenize(ctx_words)
    dest_toks = tokenize(dest_seq).repeat(len(ctx_words), 1)
    ctx_labels = t.tensor(ctx_labels_l, device=device)
    assert ctx_toks.numel() > 0, f"{bold + red}no matching words found{endc}"

    model.reset_hooks()
    src_logits, src_cache = model.run_with_cache(ctx_toks)
    model.add_hook(hookpoint, partial(replace_act_hook, new_act=src_cache[hookpoint][:, src_pos], seq_pos=dest_pos))
    #head = 1
    #model.add_hook(f"blocks.{head}.attn.hook_result", partial(replace_attn_result_hook, new_attn_result=src_cache["result", 1][:, :, head], head=head))
    #head = 0
    #model.add_hook(f"blocks.{head}.attn.hook_result", partial(replace_attn_result_hook, new_attn_result=src_cache["result", 1][:, :, head], head=head))
    dest_logits, dest_cache = model.run_with_cache(dest_toks)
    model.reset_hooks()
    del src_cache, src_logits, ctx_toks
    return dest_cache, dest_logits, ctx_words, ctx_labels

#%%
unpatched_mpcs, dest_probs_mpcs = [], []
for s in trange(26):
    for dest_word_u in dataset.word_list[:10]:
        dest_word = shift(dest_word_u, s)
        
        dest_cache, dest_logits, ctx_words, ctx_labels = ctx_sens3(dest_word, "blocks.1.attn.hook_v", 1, 1, matching_swapped=True)
        dest_probs = dest_logits.softmax(-1)
        
        dest_probs_mpc = dest_probs[t.arange(len(ctx_words)), :, s].mean(0) 
        unpatched_mpc = model(tokenize(dest_word)).squeeze().softmax(-1)[:, s]
        
        unpatched_mpcs.append(unpatched_mpc)
        dest_probs_mpcs.append(dest_probs_mpc)

print(f"original logits for dest_words: {sum(unpatched_mpcs) / len(unpatched_mpcs)}")
print(f"swapped logits: {sum(dest_probs_mpcs) / len(dest_probs_mpcs)}")
# findings from this type of experiment:
# If we totally randomize the first letter, we see evry minor degradation of second letter performance (.714->.672), and no
# degradation of last letter performance. This is very strong evidence that second and last letters dont care about the value of
# the first letter. Patching between middle letter positions basically only entails changing the first letter info that is
# contained in the middle letter, and was moved there by attn0, becuase we only patch between words where the src and dest
# letters are the same. The surprising thing is that this patch actually does degrade last letter mpc a lot, from 0.87 to 0.52. 
# It seems that first letters do impact last letters, but only indirectly through the first letter info containe in the middle letter.

# Why this indirection as opposed to attending directly to the first letter? The model could be using this to save space in
# activation space. For example, after summing the first letter and second letter values in attn0, we have a different direction
# for every possible bigram, becuase each embedding (and therefore each value vector) must have a unique direction to its own in
# residual space. But if two bigrams have the same direction, they don't  need to occuppy different directions in residual space,
# and if a bigram never occurs in the dataset (the overwhelmingly common option), it doesn't need to have a direction at all.

# A shorter way to put this is that, on the first letter position, the attention mechanism creates 'bigrams' of value vectors,
# and the out projection of attn0 creates bigram detection features.
# A stronger hypothesis is potentially that the model is turning bigrams into distribution detection features, as in two bigrams
# which have the same/similair shift distributions will point in the same/similair direction in activation space (which activation
# space? attn1 value space right?)
# Actually this doesnt seem totally sound for the model but might be worth it in practice. Suppose we have two bigrams with the
# same shift distribution but different literals. Lets say for both of these bigrams there is a 50/50 that the shift is either
# 5 or 11. Each bigram will have only two possible last letters.

# The followup questions:
# How do we test the hypothesis that attn0 is turning sum of embeddings into bigram detection features? 
# well to test the strong hypothesis we simply need to patch between middle letter positions with similair
# distributions, without or without shared literal characters, and see if it distirubs the last letter
# performance. this follows immediately

# %% testing the 'attn0 makes bigram detection features out of z values on middle letter positions' hypothesis
# here we gather literal trigrams and bigrams with varying probabilities of having the same shift as the trigram.
# we patch between middle letters and examine mean prob on middle and last letters to see if what matters is the
# literal letters or the distribution of the bigram in the middle letter position.

zzz = []
def ctx_sens4(hookpoint, prop_min=0.0, prop_max=1.0, count_cutoff=10):
    src_words, dest_words, dest_labels = [], [], []
    words = [(w, s) for w, s in gram_freqs.items() if len(w)==3]
    bigrams = [(w, s) for w, s in gram_freqs.items() if len(w)==2]
    for word, wshifts in words:
        #for wshift in wshifts:
        if len(set(wshifts)) == 1:
            for bigram, bshifts in bigrams:
                #prop = bshifts.count(wshift) / len(bshifts) 
                prop = bshifts.count(wshifts[0]) / len(bshifts) 
                if len(bshifts) > count_cutoff and bigram[0] != word[0] and bigram[1] != word[1]: # and occurrs >coutn_cutoff times and doesnt share any letters with the src word
                    if prop >= prop_min and prop <= prop_max: # and the proportion of the occurrences of the bigram shift distn is within given bounds
                        src_words.append(bigram)
                        dest_words.append(word)
                        dest_labels.append(wshifts[0])
    if (nseq := len(src_words)) > 5000:
        sampled_indices = t.randperm(nseq)[:5000]
        src_words = [src_words[i] for i in sampled_indices]
        dest_words = [dest_words[i] for i in sampled_indices]
        dest_labels = [dest_labels[i] for i in sampled_indices]
    src_toks, dest_toks = tokenize(src_words), tokenize(dest_words)
    dest_labels = t.tensor(dest_labels, device=device, dtype=t.int32)
    assert src_toks.numel() > 0, f"{bold + red}no matching words found{endc}"
    
    model.reset_hooks()
    src_logits, src_cache = model.run_with_cache(src_toks)
    model.add_hook(hookpoint, partial(replace_act_hook, new_act=src_cache[hookpoint][:, 1], seq_pos=1)) # patching only between middle letters
    #model.add_hook("blocks.1.attn.hook_k", partial(replace_act_hook, new_act=src_cache["blocks.1.attn.hook_k"][:, 1], seq_pos=1)) # patching only between middle letters
    dest_logits, dest_cache = model.run_with_cache(dest_toks)
    model.reset_hooks()
    zzz.append(src_logits.softmax(-1)[t.arange(src_toks.shape[0]), 1, dest_labels].mean().item())
    del src_cache, src_logits, src_toks, dest_toks, dest_cache
    return dest_logits, src_words, dest_words, dest_labels

mpc_seq_pos = 1
act = "blocks.1.hook_attn_out"
scores = []
for prop in trange(12):
    dest_logits, src_words, dest_words, dest_labels = ctx_sens4(act, prop_min=(prop-1)/10, prop_max=prop/10)
    dest_probs = dest_logits.softmax(-1)
    dest_probs_mpc = dest_probs[t.arange(len(src_words)), mpc_seq_pos, dest_labels]
    scores.append(dest_probs_mpc.mean().item())
lines(
    [zzz, scores],
    x=[f"{(i-1)/10}-{i/10}" for i in range(12)],
    labels=["original bigram mpc", "patched bigram mpc"],
)

#%%
mpc_seq_pos = 1
acts = ["blocks.0.hook_attn_out", "blocks.0.hook_resid_post", "blocks.1.attn.hook_v", "blocks.1.hook_resid_post"]
scores = [[] for _ in acts]
for i in trange(len(acts)):
    act = acts[i]
    for prop in range(12):
        dest_logits, src_words, dest_words, dest_labels = ctx_sens4(act, prop_min=(prop-1)/10, prop_max=prop/10)
        dest_probs = dest_logits.softmax(-1)
        dest_probs_mpc = dest_probs[t.arange(len(src_words)), mpc_seq_pos, dest_labels]
        scores[i].append(dest_probs_mpc.mean().item())
lines(
    scores,
    labels=acts,
    x=[f"{(i-1)/10}-{i/10}" for i in range(12)],
    title="Performance when patching the second letter of certain bigrams into the second letter of full words (dest words)",
    xaxis="0.1-0.2 means only src'ing from bigrams where the probability of the bigram on shift s, the shift of the dest word, is 0.1 <= s <= 0.2",
    yaxis=f"mean prob on dest_word shift in {mpc_seq_pos} letter position of dest word, on the true shift of dest word"
)

# results:
# We patch using 4 different activations. our dest sequences are 3 letter sequences who only have one possible shift value, but adds its modified output to the original embedding in that seqeunce position right before the
# unembed.
# Our src sequences are bigrams with no shared characters with the dest words' first and last letters. For each dest word,
# we filter the bigrams by those which only have a certain probability of occurring as the shift of the dest word.
# We look at the mpc for both middle letters (the patched position) and last letters.
# Basically by patching in things downstream of attn0, based on the mpc on the patched position and later, we can figure
# out what about the input matters to the components downstream. For example, if we patch in attn_out and find that the
# third letter position gets high MPC for giveaway bigrams, it means that, despite attn1 attending between a normal sequence
# position value and patched one, it means that despite the literals from which attn0's output was formed, the thing it produced
# still correctly composes inside attn1 with the last letter position to form a uinfied prediction. In other words, if the
# model is not impacted by patching after attn0 between sequences with the same distn but different literals, it means that
# attn0's output does not encode literal token information, but distribution information, to the extent that these are not the
# same info. (or it at least means that the info attn1 cares about from attn0 is not the literal token info, but i beleive attn0
# basically does nothing but pass info to attn1)

# middle letter mpc conclusions:
# patching with 'blocks.0.hook_resid_post' or 'blocks.1.hook_resid_post' show basically unablated performance for the middle
# letter. using attn1 values or attn0 output shows much worse performance. What are the differences between these patches and
# what does the discrepancy tell us? One difference is that the first two actually modify the information in the residual stream
# aned not just changing what gets added to it. However, patching at attn1 values means the input to the attn1 values are the
# same as for the src bigram, but adds its modified output to the original embedding in that seqeunce position right before the
# unembed. We know previously that if all we do is fuck with the embeddings before the unembed we get perfectly fine performance.
# The only difference in that patch then is the effect of having keys and queries mistmatched (where by mistmatched i mean not
# matching between the dest and src sequence), and a mistmatched first letter. By additionally patching in keys and by only
# patching between sequences where the first letters are the same, we can recover basically all the performance. This tells us
# that 

#%% how are attn1 heads different from eachother?
# based on the attention scores for each head:
# h1.0 attends to last letters (and the first letter before there is a last letter)
# h1.1 attends primarily to the second letter, and half as much to alst letters.


# With the understanding that attn0 is creating bigram detection features in the middle letter position, what would
# the different attn1 heads be doing? We know that one/both attn1 head is combining the bigram detection feature with
# the value of the last letter to form a trigram distribution. It seems natural that one head would be looking just
# at the bigram detection feature and outputting the appropriate distribution. This would definitely be h1.1's job, as
# it is the head that is most interested in the second letter.

# We attempt to test the theory that h1.1 is primarily reponsible for outputting the distribution of the bigram feature
# computed by attn0. We do this by scrambling the attn1 values for each head 

dest_letter_idx = 5
src_letter_idx = 1

head = None
act = f"blocks.1.attn.hook_v"

scrambled_act = cache[act].clone()
#if head is not None: scrambled_act[:, dest_letter_idx, head] = scrambled_act[randperm, src_letter_idx, head] # src and dest come from different sequence positions in same sequence
#else: scrambled_act[:, dest_letter_idx] = scrambled_act[randperm, src_letter_idx] # src and dest come from different sequence positions in same sequence
scrambled_act[:, dest_letter_idx] = scrambled_act[:, src_letter_idx] # src and dest come from different sequence positions in same sequence

model.reset_hooks()
model.add_hook(act, partial(replace_act_hook, new_act=scrambled_act))
model.add_hook("blocks.1.attn.hook_attn_scores", partial(mask_attn_scores_hook, mask=~intraword_mask))
scrambled_logits, scrambled_cache = model.run_with_cache(dataset.toks)
model.reset_hooks()

scrambled_probs = scrambled_logits.softmax(-1)
scrambled_mpc_bsp = scrambled_probs[bidx, :, dataset.labels].mean(0)
scrambled_mpc = scrambled_mpc_bsp.mean()
idx = [i for i in range(1000) if (s:=dataset.stoks[i])[1] != s[5] and s[0] != s[4]]
print(len(idx))
lines(
    [mpc_by_seq_pos, attn1_intraword_mpc_bsp, scrambled_mpc_bsp, scrambled_probs[idx, :, dataset.labels[idx]].mean(0)],
    x=[f"{s}<br><sub>{i}</sub>" for i, s in enumerate(dataset.str_toks[0])],
    title=f"MPC by seq pos when scrambling h1.{head} '{act}'<br>src: {src_letter_idx}, dest: {dest_letter_idx}",
    labels=["original", "just intraword", "ablated", "ablated on examples<br>where the src and dest<br>bigrams are the same"]
)
# We find moderate impact (~0.5 down from 0.7 unablated) when we scramble the values of h1.0 and very bad (~0.1) performance when we scramble the values of h1.1.
# so second letters are mostly but not entirely the job of h1.0.
# If we instead patch between sequence positions in the same sequence, we drop to about 0.5 mpc for middle letters when using 'blocks.1.attn.hook_v'.
# We get perfect performance if we use 'blocks.0.hook_reisd_post' and no performance when we do the same for 'blocks.0.hook_attn_out'. 
# The relative patchability of these activations suggest that the embeddings before and after attn1 are relevant for predicting middle letter positions,
# unlike in bigram->trigram middle letter patching show above where saw high patchability using blocks.0.hook_attn_out, meaning 

#%%

hookpoint = "blocks.1.hook_resid_post"

model.reset_hooks()
no_emb_probs = model.run_with_hooks(dataset.toks, fwd_hooks=[(hookpoint, partial(replace_act_hook, new_act=cache['resid_post', 0] - cache['resid_pre', 0]))]).softmax(-1)

random_seq_toks = random_words(1, n_words=8)
rand_emb_probs = model.run_with_hooks(dataset.toks, fwd_hooks=[(hookpoint, partial(replace_act_hook, new_act=cache['resid_post', 0] - cache['resid_pre', 0] + model.embed(random_seq_toks)))]).softmax(-1)

print(f"mean prob on correct with embeddings subtracted: {no_emb_probs[bidx, :, dataset.labels].mean():.3f}")
print(f"mean prob on correct with random embeddings in all (non-space) positions: {rand_emb_probs[bidx, :, dataset.labels].mean():.3f}")

lines(
    [mpc_by_seq_pos, rand_emb_probs[bidx, :, dataset.labels].mean(0), no_emb_probs[bidx, :, dataset.labels].mean(0)],
    x=[f"{s}<br><sub>{i}</sub>" for i, s in enumerate(dataset.str_toks[0])],
    title=f"MPC by sequence position when embeddings are subtracted from '{hookpoint}'",
    labels=["unablated", "scrambled embeddings", "subtracted embeddings"]
)