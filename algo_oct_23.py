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
def show_probs(logits: t.Tensor, toks: t.Tensor):
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
# - We would expect the model to attend to all the post-sep tokens, as well as selectively attend to the pre-sep tokens based on their value and the current token's.
# - we would expect that when the head attends to post-sep tokens, that the logits are all negative, becuase the only reason it would attend there is to inhibit duplicates.
# - we would expect that when the head attends to pre-sep tokens, that the logits are all positive, because the only reason it would attend there is to boost the correct next token.
# - If we take the embeddings and/or the Q, K values for all the tokens and dot them together ot get a similaaity matrix, we should expect the values to be proportional (?) to the distance to the main diagonal

#%% attention patterns and logits on a sequence with a large jump
jump_seq = [1, 2, 3, 4, 5, 45, 46, 47, 48, 49]
random.shuffle(jump_seq)
jump_seq_toks = list_to_toks(jump_seq)
jump_seq_logits, jump_seq_cache = model.run_with_cache(jump_seq_toks)
show_probs(jump_seq_logits, jump_seq_toks)
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

# The pattern of h0.0 is quite different. attends strongly to 1 on sep and the first token of the sorted list. But from
# tokens [2-5] is attends strongly to the current token, decaying as we go backwards in the sequence. So mostly on
# the current token, some on the token behind, less on the token behind that.

# The patterns of h0.0 and h0.1 are visually quite different. After the sep token, H0.1 attends very little to post-sep tokens, whereas h0.0
# attends there strongly.
# H0.1 seems to be the 'what is next' head, and h0.0 seems to be the 'what isnt next' (as in what has already occurrsed in the sorted list) head.
#%%

embed_normed = model.W_E / model.W_E.norm(dim=-1, keepdim=True)
imshow(einops.einsum(embed_normed, embed_normed, "voc1 d_model, voc2 d_model -> voc1 voc2"), title="Similarity matrix of embeddings")

#%%

posembed_normed = model.W_E_pos / model.W_E_pos.norm(dim=-1, keepdim=True)
imshow(einops.einsum(posembed_normed, posembed_normed, "voc1 d_model, voc2 d_model -> voc1 voc2"), title="Similarity matrix of posembeddings")

#%%