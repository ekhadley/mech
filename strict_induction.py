#%%
from mechlibs import *
t.no_grad()
# in this dialogue,
# https://www.lesswrong.com/posts/tEPHGZAb63dfq2v8n/how-useful-is-mechanistic-interpretability
# A question gets brought up around induction heads. Buck suggests that 'this head does induction'
# captures around 1% of the important stuff that an induction head a head is doing even when it
# looks very inductiony. Neel responds with asking if he means that we replaced the inductiony
# head with python code that does strict induction that this would be about 1% as useful to the
# model. He says he expects it to be about 10-20% as useful to the model.
# Here i replace an induction head with python code and see how useful the head becomes.

device = t.device("cuda" if t.cuda.is_available() else "cpu")


model = HookedTransformer.from_pretrained(
    "gpt2-small",
    center_unembed=True,
    center_writing_weights=True,
    fold_ln=True,
    refactor_factored_attn_matrices=True,
    device=device
)

#%%

def gen_induction_test(length, batch_size=1):
    tokens = t.randint(0, model.cfg.d_vocab, (batch_size, length), device=device)
    tokens[..., 0] = model.tokenizer.bos_token_id # prepend with bos token
    tokens = tokens.repeat(1, 2)
    return tokens

@t.inference_mode()
def loss(logits, labels):
    return t.nn.functional.cross_entropy(logits.view(-1, logits.shape[-1]), labels.view(-1))
def induction_task_score(model):
    tokens = gen_induction_test(16, 16)
    logits = model.forward(tokens)
    return loss(logits, tokens)

# measures what percentage of the attention pattern of each token is on the previous token
def prev_token_head_score(cache, layer, head):
    patterns = cache["pattern", layer].mean(0)
    pattern = patterns[head]
    prev_token_diagonal = pattern.diagonal(-1)
    score = prev_token_diagonal.sum() / pattern.sum()
    return score
# measures, for the repeated random tokens task, what percentage of the attention pattern of the second half of the sequence is on the correct token.
def induction_head_score(cache, layer, head):
    patterns = cache["pattern", layer].mean(0)
    pattern = patterns[head]
    induction_diagonal = pattern.diagonal(-pattern.shape[0]//2+1)
    score = induction_diagonal.sum() / pattern.sum()
    return score
# returns two sorted lists of heads, one sorted by previous token head score, the other by induction head score
def find_induction_heads(cache):
    top_pheads, top_iheads = [], []
    for layer in range(model.cfg.n_layers):
        for head in range(model.cfg.n_heads):
            pscore = prev_token_head_score(cache, layer, head).item()
            iscore = induction_head_score(cache, layer, head).item()
            top_pheads.append((layer, head, pscore))
            top_iheads.append((layer, head, iscore))
    top_pheads.sort(key=lambda x: -x[2])
    top_iheads.sort(key=lambda x: -x[2])
    return top_pheads, top_iheads

def induction_score_map(model):
    tokens = gen_induction_test(16, 32)
    _, cache = model.run_with_cache(tokens)
    induction_scores = t.zeros((model.cfg.n_layers, model.cfg.n_heads), device=device)
    for layer in range(model.cfg.n_layers):
        for head in range(model.cfg.n_heads):
            induction_scores[layer, head] = induction_head_score(cache, layer, head)
    return (induction_scores - induction_scores.mean()) / induction_scores.std()

#%%

induction_scores = induction_score_map(model)
imshow(induction_scores, title="which heads attend most strongly to the correct token on the repeated random tokens task")
tokens = gen_induction_test(16)
logits, cache = model.run_with_cache(tokens)

print(f"model score on the repeated random tokens task: {bold}{induction_task_score(model):.3f}{endc}")

top_pheads, top_iheads = find_induction_heads(cache)
print(lime, f"top previous token heads: {top_pheads[:5]}", endc)
print(cyan, f"top induction heads: {top_iheads[:5]}", endc)
#%%
patterns = t.stack([cache["pattern", layer] for layer in range(model.cfg.n_layers)])[:, 0, :, :]
layer = 4
cv.attention.attention_patterns(
    attention = patterns[layer],
    tokens = model.to_str_tokens(tokens[0]),
    attention_head_names = [f"{layer}.{head}" for head in range(model.cfg.n_heads)],
)
print("our strongest prev token head is 4.11 with over 90% of its attn on the prev token. Second most has less than 50%. our induction-y heads on the repeated tokens task were 5.1, 7.2, 5.0, 6.9, 5.5, decreasing monotonically with 5.1 at about .37 iscore The fact that there are no standout inductiony heads lends credence to the idea that they are __literally__ just doing induction. probably there is some specialization in the heads, like diff types of induction, or totally different, non induction jobs on other prompts.")

#%%

# this replaces an attention pattern with one which attends entirely to the previous token and nothing else.
def strict_prev_token_head_hook(
        patterns: Float[Tensor, "batch seq seq"],
        hook: HookPoint,
        head: int
) -> Float[Tensor, "batch seq seq"]:
    batch_size, nhead, seq, _ = patterns.shape
    patterns[:, head] = t.zeros_like(patterns[:, head])
    for i in range(batch_size):
        t.ones((seq-1), out=patterns[i, head].diagonal(-1))
    patterns[..., 0, 0] = 1.0
    return patterns

print("Here we test our strict previous token head hook. It provides a very small but consistent improvement to performance on the repeated random tokens task.")
tokens = gen_induction_test(16, 32)
cleanlogits = model.forward(tokens)
print(lime, f"original model score on the repeated random tokens task: {bold}{loss(cleanlogits, tokens)}{endc}", endc)
layer, head = 4, 11
hook = partial(strict_prev_token_head_hook, head=head)
logits = model.run_with_hooks(tokens, fwd_hooks=[(lambda name: name==f"blocks.{layer}.attn.hook_pattern", hook)])
print(lime, f"model score on the repeated random tokens task with head {layer}.{head} replaced with a strict inductor: {bold}{loss(logits, tokens)}{endc}", endc)

#%%
# This is my take on an 'only induction head'.
# We first calculate the attention scores normally.
# then we go and find, for each sequence position, indices of duplicats of the 
# current token. We store the indices of the token after these duplicates.
# Then we go around recalculating attention scores based on if the destination
# token has a duplicate anywhere before it. If it does, we artificially set
# the key of that src token (the one after the previous self duplicate) to be
# identical to our current token's query vector. This emulates a particular token
# head looking for exactly the token that came after some previous duplicate of itself
# whihc is what induction does.
# We also exclude induction for periods and commas becuase is seems like the heads
# do that normally.
# Also if there are multiple self duplicates with different following tokens, we 
# maximize the key-query similarity for all of them and split up the attention.
# Destination tokens without a duplicate in the context are unmodified.
def strict_induction_head_hook(
        patterns: Float[Tensor, "batch seq seq"],
        cache: ActivationCache,
        hook: HookPoint,
        head: int,
        tokens: Int[Tensor, "batch seq"],
        vis_pattern = False,
        pattern_out = None
) -> Float[Tensor, "batch seq seq"]:
    hookname = f"blocks.{hook.layer()}.attn.hook_"
    q, k = cache[hookname+'q'], cache[hookname+'k']

    # for each current token, get a list of indices of previous identical tokens
    nexttoks = []
    for i, tok in enumerate(tokens[0]):
        current = tok.item()
        same = (tokens[0] == current).nonzero().squeeze().tolist()
        if current not in [11, 13] and isinstance(same, list) and len(same) > 1:
            same = [ti+1 for ti in same if ti < i]
            nexttoks.append(same)
        else:
            nexttoks.append([])

    # calculate attention scores
    attn_scores = einops.einsum(
        q, k,
        "batch posn_Q nheads d_head, batch posn_K nheads d_head -> batch nheads posn_Q posn_K", 
    )
    for i, nexttoks in enumerate(nexttoks): # for every token
        if len(nexttoks) != 0: # if it has duplicates
            for dupe_idx in nexttoks:
                # change our dest query to = src token (prev duplicate indices + 1) key and dot them together to get score
                attn_scores[:, head, i, dupe_idx] = einops.einsum(q[:, i, head], q[:, i, head], "batch d_head, batch d_head -> batch")

    # rest of normal attention
    attn_scores /= model.cfg.d_head ** 0.5
    all_ones = t.ones(attn_scores.size(-2), attn_scores.size(-1), device=attn_scores.device)
    mask = t.triu(all_ones, diagonal=1).bool()
    attn_scores.masked_fill_(mask, -1E5)
    attn_pattern = attn_scores.softmax(-1)
    if vis_pattern:
        cv.attention.attention_pattern(
            attention = attn_pattern[0, head],
            tokens = model.to_str_tokens(tokens.squeeze()),
        ).show()
    if pattern_out is not None: pattern_out = attn_pattern[0, head]
    return attn_pattern

def attn_out_random_ablation_hook(
        attn_out: Float[Tensor, "batch seq d_model"],
        hook: HookPoint,
        head: int,
) -> Float[Tensor, "batch seq d_model"]:
    mean, var = attn_out[:, head].mean(), attn_out[:, head].var()
    attn_out[:, head] = var*t.rand(attn_out[:, head].shape, device=attn_out.device) + mean
    return attn_out

def replace_heads_with_strict_induction(model,  heads, cache, tokens):
    for layer, head in heads:
        hook = partial(strict_induction_head_hook, head=head, tokens=tokens, cache=cache)
        logits = model.run_with_hooks(tokens, fwd_hooks=[(f"blocks.{layer}.attn.hook_pattern", hook)])
    return logits

@t.inference_mode()
def strict_induction_ablation_map(model, clean_cache, tokens, normalize=False):
    clean_loss = loss(model.forward(tokens), tokens)
    model.reset_hooks()
    induction_scores = t.zeros((model.cfg.n_layers, model.cfg.n_heads), device=device)
    for layer in trange(model.cfg.n_layers, ncols=100, desc="mapping strict induction ablations. . ."):
        for head in range(model.cfg.n_heads):
            t.cuda.empty_cache()
            hook = partial(strict_induction_head_hook, head=head, tokens=tokens, cache=clean_cache)
            logits = model.run_with_hooks(tokens, fwd_hooks=[(f"blocks.{layer}.attn.hook_pattern", hook)])
            induction_scores[layer, head] = loss(logits, tokens) - clean_loss
    if normalize: return (induction_scores - induction_scores.mean()) / induction_scores.std()
    return induction_scores

@t.inference_mode()
def random_ablation_map(model, tokens, start_layer=0, normalize=False):
    model.reset_hooks()
    clean_loss = loss(model.forward(tokens), tokens)
    scores = t.zeros((model.cfg.n_layers-start_layer, model.cfg.n_heads), device=device)
    for layer in trange(start_layer, model.cfg.n_layers, ncols=100, desc="mapping noise ablations. . ."):
        for head in range(model.cfg.n_heads):
            t.cuda.empty_cache()
            hook = partial(attn_out_random_ablation_hook, head=head)
            logits = model.run_with_hooks(tokens, fwd_hooks=[(f"blocks.{layer}.hook_attn_out", hook)])
            scores[layer-start_layer, head] = loss(logits, tokens) - clean_loss

    if normalize: return (scores - scores.mean()) / scores.std()
    return scores

#%%
print("pattern for 5.1 (an induction head) on harry potter")
hptext = "Mr and Mrs Dursley, of number four, Privet Drive, were proud to say that they were perfectly normal, thank you very much. They were the last people you'd expect to be involved in anything strange or mysterious, because they just didn't hold with such nonsense. Mr Dursley was the director of a firm called Grunnings, which made drills."
hptoks = model.tokenizer(hptext, return_tensors='pt', add_special_tokens=True)['input_ids'].to(device)
_, hp_cache = model.run_with_cache(hptoks)
str_tokens = model.to_str_tokens(hptoks.squeeze())
cv.attention.attention_pattern(
    attention = hp_cache["pattern", 5][0, 1],
    tokens = str_tokens,
)
#%%
print("the pattern of a strict induction head on the same text")
model.reset_hooks()
hook = partial(strict_induction_head_hook, head=5, tokens=hptoks, cache=hp_cache)
#_, strict_hp_cache = model.run_with_hooks(hptoks, fwd_hooks=[("blocks.5.attn.hook_pattern", hook)])
model.add_hook("blocks.5.attn.hook_pattern", hook)
_, strict_hp_cache = model.run_with_cache(hptoks)
cv.attention.attention_pattern(
    attention = strict_hp_cache["pattern", 5][0, 5],
    tokens = str_tokens,
)
model.reset_hooks()

#%%
#text = "Mr and Mrs Dursley, of number four, Privet Drive, were proud to say that they were perfectly normal, thank you very much. They were the last people you'd expect to be involved in anything strange or mysterious, because they just didn't hold with such nonsense. Mr Dursley was the director of a firm called Grunnings, which made drills. He was a big, beefy man with hardly any neck, although he did have a very large moustache. Mrs Dursley was thin and blonde and had nearly twice the usual amount of neck, which came in very useful as she spent so much of her time craning over garden fences, spying on the neighbours. The Dursleys had a small son called Dudley and in their opinion there was no finer boy anywhere. The Dursleys had everything they wanted, but they also had a secret, and their greatest fear was that somebody would discover it. They didn't think they could bear it if anyone found out about the Potters. Mrs Potter was Mrs Dursley's sister, but they hadn't met for several years; in fact, Mrs Dursley pretended she didn't have a sister, because her sister and her good- for-nothing husband were as unDursleyish as it was possible to be. The Dursleys shuddered to think what the neighbours would say if the Potters arrived in the street. The Dursleys knew that the Potters had a small son, too, but they had never even seen him. This boy was another good reason for keeping the Potters away; they didn't want Dudley mixing with a child like that."
text = 'George Washington (February 22, 1732 - December 14, 1799) was an American Founding Father, military officer, and politician who served as the first president of the United States from 1789 to 1797. Appointed by the Second Continental Congress as commander of the Continental Army in 1775, Washington led Patriot forces to victory in the American Revolutionary War and then served as president of the Constitutional Convention in 1787, which drafted the current Constitution of the United States. Washington has thus become commonly known as the "Father of his Country".'
toks = model.tokenizer(text, return_tensors='pt', add_special_tokens=True)['input_ids'].to(device)

print("here we test our strict induction head hook by observing loss differences on wikipedia text by ablating each head one at a time.")
logits, strict_induction_cache = model.run_with_cache(toks)
induction_ablation_map = strict_induction_ablation_map(model, strict_induction_cache, toks)
imshow(induction_ablation_map, title="loss difference when ablating each head with a strict induction head")
#%%

random_ablation_scores = random_ablation_map(model, toks, start_layer=1)
imshow(random_ablation_scores, title="loss difference when ablating each head with normal noise")

# The strict induction ablation map shows very strong degradation on ~5 heads in layers 0-3, and
# the impacts of random ablation are very large for all first layer heads, and matter
# little elsewhere.
# oddly, ablating the layer one heads with strict induction has very little impact,
# (compared to ablating with random noise, which basically destroys the model.)

# Ablating with strict induction is particularly harmful for ~5 heads in layers 0-3, and have
# little impact elsewhere.
# There appears to be no significant correlation between the effect of ablating with random noise
# and ablating with strict induction.

# Results: ?
# basically no relationship was found between mean ablation score difference and strict induction
# score difference. This was only tested on a single sequence, the first paragraph of harry potter,
# so theres a potential weakness. We also tried replacing the 5 most inductiony heads (on the repeated
# random tokens task) with struct inductors and find zero loss degradation. Replacing any number of
# random heads with strict inductors actually makes little difference, even if they werent inductiony
# before.
# The lack of found correlation could be due to backups in the model. Perhaps that is why ablating
# any single head with noise or strict induction results in little loss difference, for most heads.
# (meaning something like backup name mover heads. Another head could be picking up slack during
# ablation and hiding  performance drops)

# Potential future directions:
# we only evaluated performance on a single paragrah of fiction. To evaluate wether any heads that
# seem inductiony are doing pure induction, it would probably be necessary to examine performance
# many more tokens to try and find if there are any where the heads are failing to perform their
# backup (or primary who knows) job (on ANY sequence, not averages).

# For reasons stated above, one could investigate the impact of ablating groups of heads with strict
# induction and comparing the loss differences to ablating those same heads with noise.
# %%
