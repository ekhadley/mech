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
n_heads = 2
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
def show(batch_idx: int, dataset=dataset, probs=True):
    logits: Tensor = model(dataset.toks)[:, dataset.list_len:-1, :]
    if probs:
        logits = logits.log_softmax(-1).softmax(-1)

    str_targets = dataset.str_toks[batch_idx][dataset.list_len+1: dataset.seq_len]

    imshow(
        logits[batch_idx].T,
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
def show_logits(logits: t.Tensor, toks: t.Tensor, probs=True, title=None):
    logits = logits.squeeze()[dataset.list_len:-1, :]
    if probs:
        logits = logits.log_softmax(-1).softmax(-1)
    strtoks = to_str_toks(toks)
    imshow(
        logits.T,
        y=dataset.vocab,
        x=[f"{strtoks[i]}<br><sub>({i})</sub>" for i in range(11, 21)],
        labels={"x": "Token", "y": "Vocab"},
        xaxis_tickangle=0,
        title= (f"Sample model probabilities:<br>Unsorted = ({','.join(strtoks[:11])})" if title is None else title),
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
# The way this model specifically works is that it sees the full list, then a separator token. After the
# separator token, is is autoregressively trained to predict the next token in the sorted list. (meaning
# given all the unsorted tokens and the previous sorted token, predict the next sorted token).

# Here is a different algorithm: the first element of the sorted list is the smallest element. Then if you
# put the closest number to the previous one which has not already been placed, you get an ascending
# sorted list (ignoring duplicates).

# Ok just found out that the lists the model was trained on contain all unique numbers, so no duplicates.
# This makes the 'closest number not already placed' heuristic seem pretty good.
# As a first guess, the model might just be boosting the logits for tokens which are close to the current
# AND come before the sep token, and reducing the logits for any token after the sep token.
# Oh and the model could just assign the 'sep' token to act like a low value, so that when the model is
# on the sep token, and looks for a token close in value to the current token, it will look for low values.
# So let's test the hypothesis: what would we be likely to observe in the attention patterns if this algorithm is being implemented?
# - We might see that the model gets confused when the next largest token is much larger than the current.
#    - specifically since this algorithm mostly happens in the QK circuit, we should see that tokens will src from the correct next token less when the next token is much larger.
# - If the sep token is being given a semantically low value like close to 0, we should see low tokens src from sep more than high ones
# - We would expect the model to attend to all the post-sep tokens, as welblocks.0.hook_resid_post logits are all positive, because the only reason it would attend there is to boost the correct next token.
# - If we take the embeddings and/or the Q, K values for all the tokens and dot them together ot get a similaaity matrix, we should expect the values to be proportional (?) to the distance to the main diagonal

#%% attention patterns and logits on a sequence with a large jump
jump_seq = [0, 20, 25, 26, 27, 34, 46, 47, 48, 49]
#random.shuffle(jump_seq)
jump_seq_toks = list_to_toks(jump_seq)
jump_seq_logits, jump_seq_cache = model.run_with_cache(jump_seq_toks)
show_logits(jump_seq_logits, jump_seq_toks)
show_heads_on_input(jump_seq_cache['pattern', 0], jump_seq_toks)
# The model does get confused when the sequence contains a large jump, for just the index of the jump.
# After sep, h0.1 sources strongly from pre-sep tokens whose value is greater than the current one. It
# puts the largest weight on the correct next token, and decaying less weight for others as they get larger.
# Once we get to 5, where the next correct token is 45, we dont attend at all to any of the large pre-sep
# tokens. We instead attend mostly to the sep token itself, a bit to the current token from before the 'sep',
# and a bit to the pre-sep tokens immediately smaller than itself.
# This is strong evidence that h0.1 is trying to attend to tokens close in value to ourselves, and failing. It
# also makes sense that we would attend to SEP from 5 as opposed to 45, under the hypothesis that the sep token
# is being given a low value, numerically close to 0.
# However, h0.1 also definitely attends more to tokens whcih are greater than the current, so its not just as simple
# as 'source from the closest token'. The fact that 5 attends to sep so strongly is somewhat strange in light of this.

# So post-sep tokens appear to src from pre-sep tokens which are similair in numeric value, but especially so when
# they are larger.

#%% plotting dataset-avg attention pattern:

avg_attn_pattern = cache['pattern', 0].mean(dim=0)
cv.attention.attention_heads(
    avg_attn_pattern,
    attention_head_names=[f"head{i}.{j}" for i in range(n_layers) for j in range(n_heads)],
    tokens=dataset.str_toks[0],
)
# Both heads' average attention value is uniform across the pre-sep tokens.
# They mostly dont attend to post-sep tokens, with 2 exceptions. h0.0 strongly sources from early post-sep tokens,
# And both heads src from the start of the sorted list a bit when the current tokenis the last in the sorted list.

#%% sim matrix of all embeddings.

embed_normed = model.W_E / model.W_E.norm(dim=-1, keepdim=True)
imshow(einops.einsum(embed_normed, embed_normed, "voc1 d_model, voc2 d_model -> voc1 voc2"), title="Similarity matrix of embeddings")

#%% now we will take the embeddings and mul with the keys and queries, and make a K-Q sim matrix for the whole dictionary.
# Ahead of time predictions: Under the hypthesis that H0.1 srces from tokens which are near in value but larger than the
# current token, the shouild expect the sim matrix to be low on and below the main diagonal, and the first few diagonals
# above the main to have high value. (assuming we put keys on x axis and queries on y axis.)
# I wont include position embeddings at first just for simplicity, I expect this will mess things up. Will repeat with
# appropriate positional embeddings after.

Q = einops.einsum(model.W_E, model.W_Q.squeeze(), "d_vocab d_model, n_heads d_model d_head -> d_vocab n_heads d_head")
K = einops.einsum(model.W_E, model.W_K.squeeze(), "d_vocab d_model, n_heads d_model d_head -> d_vocab n_heads d_head")

Q_normed = Q / Q.norm(dim=-1, keepdim=True)
K_normed = K / K.norm(dim=-1, keepdim=True)

sim_matrix = einops.einsum(Q_normed, K_normed, "vocq n_heads d_head, vock n_heads d_head -> n_heads vocq vock")
imshow(sim_matrix[0], title="head 0.0 (W_E @ W_Q) @ (W_E @ W_K)^T (no positional embeddings)")
imshow(sim_matrix[1], title="head 0.1 (W_E @ W_Q) @ (W_E @ W_K)^T (no positional embeddings)")

# h0.1 looks like what i expected, with some extra structure i don't understand.

# reminder: x is the key, y is the query.
# sim[y][x] is tells us how much y wants to src from x.

# H0.1 pattern description. The pattern has 4 quarters. The top left is the same as the bottom right, and the top right
# is the same as the bottom left. Each quarter has a distinct pattern which splits it in half. The TL-BR quarters are all
# uniform-ush and positive above (not on) the main diagonal of that quarter. The TR-BL quarters are all uniform-ish and
# negative above the main diagonal of that quarter. the below-main-diagonal portion of all quartattend toers is basically 0.

# Hypothesizing about h0.1: it seems that tokens will attend to tokens which are larger than themselves, but only so much
# larger. This kind of makes sense: you dont usually want to attend to a much larger token than youraself, becuase there are
# likely some other tokens that should come before it.
# noticing confusion: why is it shaped like that though? shouldnt there be a gradual shift satarting from the main diagonal,
# descreasing based on dest-src distance? Instead it has clear zones of attend, dont attend, and attend against.

# This extra structure could be intended(?) for some reason I don't know, or it could be a result of the lack of positional
# embeddings. If we try to use the h0.1 pattern to make an adversarial example, it basically doesnt work, so either the structure of the
# patterns are not shortcomings, or if it is a shortcomings is is being accounted for.

# h0.0 is less regular than h0.1 and is a bit harder to describe. It sort of looks like h0.1 except the below main diagonal
# half of each quarter is filled in with the color of the quarter below it. So tl and br quarters are positive above the
# main diagonal and negative below it. The tr and bl quarters are negative above the main diagonal and positive below it.
# There are small 0 regions in between positive/negative sections, not a smooth gradient.

#%% now we add a certain sequence position's positional embedding to the queries, and another position's to the keys.
# This graph tells us how much does a y token at sequence position ypos, want to attend to an x token at xpos.

# resummarizing working hypothesis: h0.1 is attending to pre-sep tokens which are near in value but larger than the
# current token, incresing the odds of predicting that pre-sep token as being next. h0.0 is maybe just pushing down
# the logits of the tokens we have already seen after the sep-token, avoiding repeats.

# h0.0. predictions: we should see negative logits for any x, y when ypos and xpos are both after the sep token.
# h0.1 predictions: when ypos is post-sep, we should see the same positive above diagonal pattern when xpos is 
# pre-sep.

# I'm actually unsure about h0.0 now. Why would we need a post-sep repeat suppressor? h0.1 can just not attend to
# post sep tokens on its own, and only src values from pre-sep? If it can correctly src from only tokens
# larger than the current, it should never attend to a pre-sep token which is already in the sorted list portion.

Q_seq_pos = 11
K_seq_pos = 11
Q_plus_pos = einops.einsum(model.W_E + model.W_E_pos[Q_seq_pos], model.W_Q.squeeze(), "d_vocab d_model, n_heads d_model d_head -> d_vocab n_heads d_head")
K_plus_pos = einops.einsum(model.W_E + model.W_E_pos[K_seq_pos], model.W_K.squeeze(), "d_vocab d_model, n_heads d_model d_head -> d_vocab n_heads d_head")

Q_plus_pos_normed = Q_plus_pos / Q_plus_pos.norm(dim=-1, keepdim=True)
K_plus_pos_normed = K_plus_pos / K_plus_pos.norm(dim=-1, keepdim=True)

sim_matrix_with_pos = einops.einsum(Q_plus_pos_normed, K_plus_pos_normed, "vocq n_heads d_head, vock n_heads d_head -> n_heads vocq vock")
imshow(sim_matrix_with_pos[0], title=f"head 0.0 K-Q sim matrix<br>(where Q has posembed for seq pos {Q_seq_pos} and K has seq pos {K_seq_pos})")
imshow(sim_matrix_with_pos[1], title=f"head 0.1 K-Q sim matrix<br>(where Q has posembed for seq pos {Q_seq_pos} and K has seq pos {K_seq_pos})")

# these patterns really make no sense to me.
# they are generally positive ish below y = 38. There is a roundish positive patch for all pos values at 35, 35.
# Is there something important about token 38? There are also visible lines at x=38 and y=38 in the previous sim
# matrices, particularly for h0.0.

# I suspect im doing something out of distribution. idk where though.
# let's look at something other than attention patterns.

#%% unembedding function

def unembed(act, norm=True) -> t.Tensor:
    if norm:
        actmean = act.mean(dim=-1, keepdim=True)
        actvar = act.var(dim=-1, keepdim=True)
        act = (act - actmean) / t.sqrt(actvar + model.ln_final.eps)
        #act = act * model.ln_final.w + model.ln_final.b
    return einops.einsum(act, model.W_U, "... seq d_model, d_model vocab -> ... seq vocab") + model.b_U

#%% unembedding various stuff
# head output unembed predicitons. Assuming the heads are just boosting logits for repeating pre-sep tokens whcih are near in value but larger
# than the current, we should see decaying positive logits, starting on the correct token and falling off as the tokens get larger.
head = 0
dataset_idx = 123
show_logits(
    unembed(cache["blocks.0.attn.hook_result"][dataset_idx, :, head]),
    dataset.toks[dataset_idx],
    probs=False,
    title=f"h0.{head} output unembed on dataset_idx {dataset_idx}"
)

head = 1
show_logits(
    unembed(cache["blocks.0.attn.hook_result"][dataset_idx, :, head]),
    dataset.toks[dataset_idx],
    probs=False,
    title=f"h0.{head} output unembed on dataset_idx {dataset_idx}"
)

#show_logits(
#    unembed(model.blocks[0].attn.b_O.repeat(21, 1)),
#    dataset.toks[dataset_idx],
#    probs=False,
#    title=f"attn b_O unembedded"
#)

show_logits(
    unembed(cache['blocks.0.hook_resid_pre'][dataset_idx]),
    dataset.toks[dataset_idx],
    probs=False,
    title=f"pre-attn residual unembedded on dataset idx {dataset_idx}"
)
# This is very interesting. The main feature of unembeddings the pre-attention residual stream is that
# the logits for the previous correct token are strongly negative. So at each position s, the logits for
# the token at s-1 are strongly negative. how??????
# literally how can you do this without an attention layer.
show_logits(
    unembed(cache['blocks.0.hook_resid_pre'][dataset_idx] + cache['blocks.0.attn.hook_result'][dataset_idx, :, 0] + cache['blocks.0.attn.hook_result'][dataset_idx, :, 1] + model.blocks[0].attn.b_O),
    dataset.toks[dataset_idx],
    probs=False,
    title=f"all together now"
)
#%%

pre = cache['blocks.0.hook_resid_pre']
emb = cache['hook_embed']
posemb = cache['hook_pos_embed']
t.testing.assert_close(pre, emb + posemb)

#%%