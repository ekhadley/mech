#%%
from mechlibs import *
t.no_grad()
# in toy models, and gptj (and larger, i think), we see
# models using periodicity to solve problems of arithmetic.
# gpt2-sm is not very good at arithmetic. Does it still
# exhibit peridicity with respect to numbers? (prediction: no)

device = t.device("cuda" if t.cuda.is_available() else "cpu")


model = HookedTransformer.from_pretrained(
    "gpt2-small",
    center_unembed=True,
    center_writing_weights=True,
    fold_ln=True,
    refactor_factored_attn_matrices=True,
    device=device
)
print(model)

#%%

print("getting gpt2-sm's output on a few simple arithmetic problems:")
easy_problems = ["1+1=","2*1=","3+2=","4*2="]
easy_problem_tokens = model.tokenizer(easy_problems, return_tensors='pt')['input_ids'].to(device)
easy_problem_logits = model(easy_problem_tokens)
easy_problem_predictions = t.topk(easy_problem_logits, 5).indices
for problem, predictions in zip(easy_problems, easy_problem_predictions):
    print(f"'{problem}'")
    for prediction in predictions:
        print(f"\t{repr(model.tokenizer.decode(prediction))}")

#%%
def make_fourier_basis(p: int) -> Tuple[Tensor, List[str]]:
    '''
    Returns a pair `fourier_basis, fourier_basis_names`, where `fourier_basis` is
    a `(p, p)` tensor whose rows are Fourier components and `fourier_basis_names`
    is a list of length `p` containing the names of the Fourier components (e.g. 
    `["const", "cos 1", "sin 1", ...]`). You may assume that `p` is odd.
    '''
    x = t.arange(p, device=device)
    F = t.ones((p, p), device=device)
    names = ['const']
    for row in range(1, p//2 + 1):
        freqs = x*2*t.pi*row / p
        F[row*2 - 1] = t.cos(freqs)
        F[row*2] = t.sin(freqs)
        names.extend([f'cos {row}', f'sin {row}'])
    F /= F.norm(dim=1, keepdim=True)
    return F, names

fourier_basis, fourier_basis_names = make_fourier_basis(769) # 768 is the dimension of the GPT-2 hidden states
fourier_basis, fourier_basis_names = fourier_basis[:-1, :-1], fourier_basis_names[:-1]

def fft1d(x: t.Tensor) -> t.Tensor:
    '''
    Returns the 1D Fourier transform of `x`,
    which can be a vector or a batch of vectors.

    x.shape = (..., p)
    '''
    return einops.einsum(x, fourier_basis, "... p2, p2 p1 -> ... p1")

# %%

# find all the number-only tokens in the vocabulary (the normal integers without spaces, leading zeros, or punctuation)
vocab = model.tokenizer.get_vocab()
number_str_tokens = []
for token in vocab:
    try:
        itok = int(token)
        stritok = str(itok)
        if stritok == token and itok not in number_str_tokens:
            number_str_tokens.append(stritok)
    except ValueError:
        continue
# sort them by value
number_str_tokens.sort(key=lambda token: int(token))
print(number_str_tokens)
f"{len(number_str_tokens)} (proper) number tokens found in the vocabulary"
number_tokens = model.tokenizer(number_str_tokens, return_tensors='pt')['input_ids'].to(device)
print(f"{number_tokens.shape=}")
#%% 
# find the embeddings of the number tokens
_, numbers_cache = model.run_with_cache(number_tokens)
numbers_embed = numbers_cache['hook_embed'].squeeze()
print(f"{numbers_embed.shape=}")
# %%

animate_lines(
    numbers_embed[:30], 
    snapshot_index=number_str_tokens, 
    snapshot='Number Token', 
    title='Graphs of Number Token Embeddings'
)
print("on first inspection, it looks like the embeddings of the number tokens are not periodic")
# %%

print("checking out the fourier basis of the numeric embeddings:")
print(red, numbers_embed.shape, blue, fourier_basis.shape, endc)
numbers_fft = fft1d(numbers_embed)
animate_lines(
    numbers_fft[:30], 
    snapshot_index=number_str_tokens, 
    snapshot='Number Token', 
    title='Graphs of Number Token Embeddings Fourier Components'
)
print("looks basically like noise. not sparse at all.")
# %%

print("comparing the numeric embeddings to embeddings of random tokens:")
random_tokens = t.randint(0, len(vocab), (30,), device=device).unsqueeze(-1)
random_str_tokens = [model.tokenizer.decode(token) for token in random_tokens]
_, random_cache = model.run_with_cache(random_tokens)
random_embed = random_cache['hook_embed'].squeeze()
animate_lines(
    random_embed, 
    snapshot_index=random_str_tokens, 
    snapshot='Random Token', 
    title='Graphs of Random Token Embeddings'
)
#%%
print("and those random embeds in fourier space:")
random_fft = fft1d(random_embed)
animate_lines(
    random_fft, 
    snapshot_index=random_str_tokens, 
    snapshot='Random Token', 
    title='Graphs of Random Token Embeddings Fourier Components'
)
print("yeah they look vaguely similair")
#%%

print("but mlp0 kind of works like an extended embedding in language models. Let's see if the output of mlp0 is periodic. in normal space:")
post_mlp0 = numbers_cache['blocks.0.hook_mlp_out'].squeeze()
print(f"{post_mlp0.shape=}")
animate_lines(
    post_mlp0[:30], 
    snapshot_index=number_str_tokens, 
    snapshot='Number Token', 
    title='Graphs of Number Token Embeddings'
)
print("It looks very weird, but definitely not periodic. The output of mlp0 is basically identical for all the number tokens.")
print("neurons 138, 378, 447, and 674 are very clearly the only active neurons, and their activations are constant across the number tokens.")
# %%
print("and in the fourier basis:")
post_mlp0_fft = fft1d(post_mlp0)
animate_lines(
    post_mlp0_fft[:30], 
    snapshot_index=number_str_tokens, 
    snapshot='Number Token', 
    title='Graphs of Number Token Embeddings Fourier Components'
)

# %%
print("for completeness, mlp0's output on random tokens in normal and fourier basis:")
random_post_mlp0 = random_cache['blocks.0.hook_mlp_out'].squeeze()
animate_lines(
    random_post_mlp0, 
    snapshot_index=random_str_tokens, 
    snapshot='Random Token', 
    title='Graphs of Random Token Embeddings'
)
#%%
random_post_mlp0_fft = fft1d(random_post_mlp0)
animate_lines(
    random_post_mlp0_fft, 
    snapshot_index=random_str_tokens, 
    snapshot='Random Token', 
    title='Graphs of Random Token Embeddings Fourier Components'
)

#%%
print("lets just double check by doing the same as above, but with <|endoftext|> at the start. This is the output of mlp0 at sequence position 1:")
numtoks = t.cat([t.full((len(number_tokens), 1), 50256, device=device), number_tokens], dim=1)
_, numtoks_cache = model.run_with_cache(numtoks)
numtoks_post_mlp0 = numtoks_cache['blocks.0.hook_mlp_out'].squeeze()
print(f"{numtoks_post_mlp0.shape=}")
animate_lines(
    numtoks_post_mlp0[:30, 1], 
    snapshot_index=number_str_tokens, 
    snapshot='Number Token', 
    title='Graphs of Number Token Embeddings'
)
print("again sparse but clearly not periodic. A different set of neurons now consistely activate: 64, 266, 373, 447, and some weaker ones.")
#%%
print("output of mlp0 with random tokens with endoftext at the start")
random_toks = t.cat([t.full((len(random_tokens), 1), 50256, device=device), random_tokens], dim=1)
_, random_toks_cache = model.run_with_cache(random_toks)
random_toks_post_mlp0 = random_toks_cache['blocks.0.hook_mlp_out'].squeeze()
print(f"{random_toks_post_mlp0.shape=}")
animate_lines(
    random_toks_post_mlp0[:30, 1], 
    snapshot_index=random_str_tokens, 
    snapshot='Random Token', 
    title='Graphs of Random Token Embeddings'
)
print("and we weirdly see, like before, the same neurons (same as for the numbers, different from without endoftext) firting constantly here with noise elsewhere. \shrug")
#%%
print("so yeah seems like no periodicity here. At least no more than with any given token, and specifically in the embed and outputs of mlp0")

#%%