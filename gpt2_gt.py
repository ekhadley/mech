# This is my attempt at reverse engineering GPT-2's greater than circuit, as described
# in the intro of "How does GPT-2 compute greater than?" (https://arxiv.org/pdf/2305.00586)
#%%
from mechlibs import *
t.set_grad_enabled(False);
t.manual_seed(42)
#%%

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
nlayer = model.cfg.n_layers
nhead = model.cfg.n_heads
d_model = model.cfg.d_model
d_head = model.cfg.d_head

#%%
example_prompt = "The war lasted from 1783 to 17"
print(f"On the example prompt {repr(example_prompt)}, the model's top 10 predictions are all after 83, except for the 9th whcih is also 83:")
example_logits = model(example_prompt)
top5 = t.topk(example_logits[0, -1], 10).indices
top5_tokens = model.tokenizer.convert_ids_to_tokens(top5)
print(top5_tokens)
print("so gpt-2 seems to do pretty well at this task when framed in this way.")

#%%
nouns = [ "empire", "dynasty", "war", "occupation", "crisis", "conflict", "expedition", "campaign", "siege", "alliance", "treaty", "reform", "plague", "famine", "uprising", "prohibition", "movement", "migration", "construction", "blockade"]
print("our duration nouns, courtesy of sonnet-3.5:")
print(nouns)

#%%

#constructing our dataset class containing the nouns, a list of the prompts, a single tensor of all the tokenized prompts, and a list of the 
class DurationDataset:
    def __init__(self, nouns, start_century=11, end_century=17, start_year=2, end_year=99, prompt_format=None):
        self.prompt_format = prompt_format if prompt_format is not None else "The {noun} lasted from {firstyear} to {century}"
        self.nouns = nouns
        #we create a prompt for every noun-century combination with centuries 11-17
        # 'the war lasted from XXYY to XX'
        self.prompts = []
        self.XX, self.YY = {i:[] for i in range(start_century, end_century)}, {i: [] for i in range(start_year, end_year)}
        #for noun in nouns[:5]:
        noun = 'war'
        for century in range(start_century, end_century):
            for y in range(start_year, end_year):
                firstyear = f"{century}{'0' + str(y) if y < 10 else str(y)}"
                toks = model.tokenizer(f" {firstyear}", return_tensors="pt")['input_ids'].to(device)
                str_toks = model.to_str_tokens(toks)
                if str_toks[0] == f' {century}':
                    #self.prompts.append(f"The {noun} lasted from the year {firstyear} to the year {century}")
                    prompt = self.prompt_format.format(noun=noun, firstyear=firstyear, century=century)
                    self.prompts.append(prompt)
                    self.XX[century].append(len(self.prompts)-1)
                    self.YY[y].append(len(self.prompts)-1)

        self.tokens = model.tokenizer(self.prompts, return_tensors="pt")['input_ids'].to(device)
        self.tokens = t.cat([t.full((self.tokens.shape[0], 1), 50256, device=device), self.tokens], dim=1)

    def __len__(self): return self.tokens.shape[0]
    def __getitem__(self, i): return self.prompts[i]

dataset = DurationDataset(nouns, start_century=11, end_century=17, start_year=2, end_year=99)
num_examples = len(dataset)
other_dataset = DurationDataset(nouns, start_century=11, end_century=17, start_year=1, end_year=2)


#%%
print(f"we have {len(dataset)} prompts in our dataset. A random selection:")
for i in np.random.randint(0, len(dataset), 5):
    print(model.to_str_tokens(dataset.tokens[i]))

#%%
print("getting the logits for the whole dataset")
logits, clean_cache = model.run_with_cache(dataset.tokens)
print(pink, f"{logits.shape=}", endc)
other_logits, other_cache = model.run_with_cache(other_dataset.tokens)

#%%
yearstrs = ['0' + str(y) if y < 10 else str(y) for y in range(2, 99)]
yeartoks = model.tokenizer(yearstrs, return_tensors='pt')['input_ids'].to(device).squeeze()
print(yearstrs)
print(yeartoks)

# %%

# this function generates a logits/probability distribution over valid year completions.
# for a year in the dataset XXYY, we have several occurrences of each YY with a different XX.
# this function averages YY-YY associations for each XX, and returns a distribution over YY pairs.
def year_dist(logits, probs, year=None, dataset=dataset, seq_pos=-1):
    if year is not None: return logits[dataset.YY[year], -1].mean(0)[yeartoks]
    nyears = len(dataset.YY)
    out = t.empty((nyears, nyears), device=device)
    for y in dataset.YY.keys():
        avg_over_XX = logits[dataset.YY[y], seq_pos].mean(0)[yeartoks] # average last-token-prediction logits for 11YY, 12YY, etc.
        if probs: avg_over_XX = t.softmax(avg_over_XX, dim=-1)
        out[y-2] = avg_over_XX
    return out

def score_dist(dist):
    assert dist.shape == (97, 97)
    return (dist[*t.triu_indices(97, 97, offset=1)].sum() / dist.sum()).item()

def plot_dist(dist, title=""):
    imshow(dist, title=title, x=yearstrs, y=yearstrs, labels={'x':'to XX', 'y':'The war lasted from XX'})

#%%    
clean_year_probs = year_dist(logits, probs=True)
clean_year_probs_score = score_dist(clean_year_probs)
clean_year_logits = year_dist(logits, probs=False)
clean_year_logits_score = score_dist(clean_year_logits)
plot_dist(clean_year_probs)
print(f"clean performance score: {clean_year_probs_score}")
print(f"so on average, when given XXYY, {clean_year_probs_score:.4f} of the model's probability mass (specifically over years, excluding other types of tokens) is on years after XXYY")

#%%

def act_patch(orig_act: Tensor, hook: HookPoint, new_act: Tensor, head=None):
    if head is not None:
        orig_act[:,:, head] = new_act
    else:
        #assert new_act.shape == orig_act.shape, f"new_act_out shape {new_act.shape} does not match orig_act shape {orig_act.shape}"
        orig_act[:] = new_act
    return orig_act

def head_mean_ablation_hook(orig_z:t.Tensor, hook: HookPoint, z_mean: t.Tensor, head:int):
    orig_z[:,:, head] = z_mean
    return orig_z

def head_noise_ablation_hook(orig_z:t.Tensor, hook: HookPoint, head:int):
    z = orig_z[:, -1, head]
    noise = (t.randn_like(z) * z.std()) + z.mean()
    orig_z[:, -1, head] = noise
    return orig_z

def attn_ablation_map(ablation_type, probs=True):
    global ballz
    out = t.empty((nlayer, nhead), device=device)
    if ablation_type == 'mean': z_means = [clean_cache['z', layer].mean(0) for layer in range(nlayer)]
    for layer in trange(nlayer):
        for head in range(nhead):
            hookname = f'blocks.{layer}.attn.hook_z'
            #if ablation_type == 'mean': hook = partial(head_mean_ablation_hook, head=head, z_mean=z_means[layer][:, head])
            #elif ablation_type == 'noise': hook = partial(head_noise_ablation_hook, head=head)
            if ablation_type == 'mean': hook = partial(act_patch, new_act=z_means[layer][:, head], head=head)
            elif ablation_type == 'noise': hook = partial(head_noise_ablation_hook, head=head)
            ablated_logits = model.run_with_hooks(dataset.tokens, fwd_hooks=[(hookname, hook)])
            ablated_year_dist = year_dist(ablated_logits, probs=probs)
            out[layer, head] = score_dist(ablated_year_dist)
            t.cuda.empty_cache()
    return out - (clean_year_probs_score if probs else clean_year_logits_score)

def mlp_mean_ablation_hook(orig_mlp_out:t.Tensor, hook: HookPoint, mlp_mean: t.Tensor):
    orig_mlp_out = mlp_mean
    return orig_mlp_out

def mlp_noise_ablation_hook(orig_mlp_out:t.Tensor, hook: HookPoint):
    orig_mlp_out = (t.randn_like(orig_mlp_out) * orig_mlp_out.std()) + orig_mlp_out.mean()
    return orig_mlp_out

def mlp_ablation_map(ablation_type, probs=True):
    global boobz
    out = t.empty((1, nlayer), device=device)
    if ablation_type == 'mean': mlp_means = [clean_cache['mlp_out', layer].mean(0) for layer in range(nlayer)]
    for layer in trange(nlayer):
        hookname = f'blocks.{layer}.hook_mlp_out'
        if ablation_type == 'mean': hook = partial(act_patch, new_act=mlp_means[layer])
        elif ablation_type == 'noise': hook = mlp_noise_ablation_hook
        ablated_logits = model.run_with_hooks(dataset.tokens, fwd_hooks=[(hookname, hook)])
        ablated_year_dist = year_dist(ablated_logits, probs=probs)
        out[0, layer] = score_dist(ablated_year_dist)
        t.cuda.empty_cache()
    return out - (clean_year_probs_score if probs else clean_year_logits_score)

def mapshow(map, title=""):
    imshow(map, x=[f'head{i}' for i in range(nhead)], y=[f'layer{i}' for i in range(nlayer)], title=title)

#%%
attn_mean_ablation_scores = attn_ablation_map('mean', probs=True)
mapshow(attn_mean_ablation_scores, title="model score when ablating each head with its average over the whole dataset (probs, not logits)")
print("we see that the model is robust under all mean ablations, with the largest hit being -0.06 on h9.1")
attn_noise_ablation_scores = attn_ablation_map('noise', probs=True)
mapshow(attn_noise_ablation_scores, title="model score when ablating each head with gaussian noise (probs, not logits)")
print("However under noise ablation of h1.10 kills performance (-0.53), and h0.9:-0.25, with a few other around -0.12 and the rest negligible.")

# this is fucking weird! 
# there is one head whose mean ablation results in a score difference of -0.06 and the rest basically dont matter.
# mean ablating mlps 9 and 10 both result in about -0.09. 
# do ablations just not harm model performance in a meaningful way or are my metrics bad or do i have a bug? 
# when we look at early unembedding, we go from very little performance to normal performance around layers 9/10.
# so our ablation doesnt seem to be literally just noise, but who fucking knows.

# noise ablations seem to do the trick: ablating h1.10 totally kills performance, and head 0.9 is about half as
# important. noise ablating mlps [0-4] totally kills performance, and ablating 5, 9, and 10 also do some damage.
# 

# theories for why single-component mean ablations dont seem to harm performance:
# - the model's solution to the task is just really distributed. no one component is that important
# - taking the mean output over the whole dataset is not wiping much of the necessary information for performing the task (for heads and mlps both!!)
# - there is a bug
# - the solution is in fact sparse, but there are 'backup' components for (every!) important component
#%% 
mlp_mean_ablation_scores = mlp_ablation_map('mean', probs=True)
imshow(mlp_mean_ablation_scores, title="model score when ablating each layer's mlp output with its average over the whole dataset (probs, not logits)")
print("a similair story for mlps. mean ablation is  ineffective. We see ~-0.08 in 9 and 10, and little impact elsewhere.")
mlp_noise_ablation_scores = mlp_ablation_map('noise', probs=True)
imshow(mlp_noise_ablation_scores, title="model score when ablating each layer's mlp output with gaussian noise (probs, not logits)")
print("but noise ablation is effective. ablating 0-4 totally kills performance, and 5, 9, and 10 do some damage.")
print("some similarity to the patterns seen in the ROME paper, with the two effective regions for injection being right at the start and near the end, but not the very end. \shrug")

#%%

def unembed(resid: Float[Tensor, "... seq d_model"]):
    return einops.einsum(resid, model.W_U, '... seq d_model, d_model d_vocab -> ... seq d_vocab') + model.b_U

def score_early_unembeds(tokens=dataset.tokens, hook:Union[Callable, None]=None, hookname:Union[str, Callable, None]=None, probs=True):
    assert (hook is None) == (hookname is None), f"If you want to use a hook, provide the function and the hookpoint."
    out = t.empty((2*nlayer), device=device)
    distns = t.empty((2*nlayer, 97, 97), device=device)
    if hook is not None:
        model.reset_hooks()
        model.add_hook(hookname, hook)
        _, cache = model.run_with_cache(tokens)
    elif tokens is not dataset.tokens:
        _, cache = model.run_with_cache(tokens)
    else:
        cache = clean_cache

    for layer in trange(nlayer):
        resid_pre = cache[f'blocks.{layer}.hook_resid_pre'] # what goes into ln + attn
        resid_mid = cache[f'blocks.{layer}.hook_resid_mid'] # what goes into ln + mlp
        resid_pre_logits, resid_mid_logits  = unembed(resid_pre), unembed(resid_mid)
        resid_pre_dist, resid_mid_dist  = year_dist(resid_pre_logits, probs=probs), year_dist(resid_mid_logits, probs=probs)
        out[2*layer], out[2*layer+1] = score_dist(resid_pre_dist), score_dist(resid_mid_dist)
        distns[2*layer], distns[2*layer+1] = year_dist(resid_pre_logits, probs=True), year_dist(resid_mid_logits, probs=True)
    model.reset_hooks()
    return out.tolist(), distns

#%%

early_unembed_map, early_unembed_distns = score_early_unembeds(probs=True)
early_unembed_names = []
[early_unembed_names.extend([f'pre_attn{i}', f'pre_mlp{i}']) for i in range(nlayer)]
line(y=early_unembed_map, x=early_unembed_names, title="model score when unembedding each layer's pre-attention and pre-mlp output (after layernorm) (probs, not logits)")
print("we see that model performance hovers around 0.4 (worse than random) up until layer 9, which gives the greatest performance jump to about 0.75, and then attn10 and mlp10 bring it up to max score. ~90")
#%%

def attn_head_act_patch(orig_z: Tensor, hook: HookPoint, head: int, new_z: Tensor):
    orig_z[:, :, head] = new_z
    return orig_z

def attn_head_patch_map(probs=True):
    out = t.empty((nlayer, nhead), device=device)
    for layer in trange(nlayer):
        for head in range(nhead):
            hookname = f'blocks.{layer}.attn.hook_z'
            hook = partial(attn_head_act_patch, head=head, new_z=other_cache['z', layer][:, :, head])
            ablated_logits = model.run_with_hooks(dataset.tokens, fwd_hooks=[(hookname, hook)])
            ablated_year_dist = year_dist(ablated_logits, probs=probs)
            out[layer, head] = score_dist(ablated_year_dist)
            t.cuda.empty_cache()
    return out - (clean_year_probs_score if probs else clean_year_logits_score)

#%%
layer = 10
direct_logits = unembed(clean_cache[f'blocks.{layer}.hook_mlp_out'])
dla_year_dist = year_dist(direct_logits, probs=False)
dla_year_score = score_dist(year_dist(direct_logits, probs=True))
plot_dist(dla_year_dist - dla_year_dist.mean(), title=f"mlp{layer} direct logit attribution (mean-normalized). score: {dla_year_score:.4f}")
print("Here we see from the direct logit attribution of mlps 9 and 10 that they clearly are boosting the correct years for each example.")
print(f"The score of the direct logit attribution itself is {dla_year_score:.4f}")

#%%
model.reset_hooks()
layer = 10
mean_mlp_in = clean_cache[f'blocks.{layer}.ln2.hook_normalized'].mean(0)
model.add_hook(f'blocks.{layer}.ln2.hook_normalized', partial(act_patch, new_act=mean_mlp_in))
mean_mlp_in_ablation_logits, mean_mlp_in_ablation_cache = model.run_with_cache(dataset.tokens)
mean_mlp_in_dla = unembed(mean_mlp_in_ablation_cache[f'blocks.{layer}.hook_mlp_out'])
mean_mlp_in_year_dist = year_dist(mean_mlp_in_dla, probs=False)
mean_mlp_in_year_probs = year_dist(mean_mlp_in_dla, probs=True)
mean_mlp_in_year_score = score_dist(mean_mlp_in_year_probs)
plot_dist(mean_mlp_in_year_dist - mean_mlp_in_year_dist.mean(), title=f"mlp{layer} dla when its input is averaged over the whole dataset. score: {mean_mlp_in_year_score:.4f}")
mean_mlp_in_final_dist = year_dist(mean_mlp_in_ablation_logits, probs=True)
mean_mlp_in_final_score = score_dist(mean_mlp_in_final_dist)
plot_dist(mean_mlp_in_final_dist, title=f"the model's final distribution with above ablation. score: {mean_mlp_in_final_score:.4f} ({mean_mlp_in_final_score - clean_year_probs_score:.4f})")
model.reset_hooks()
print("Here we average the output of the layernorm before the mlp over the whole dataset and give that to the mlp.")
print("We see that this makes the mlp's output invariant over input values of YY, but NOT over next year predictions.")
print(f"The mlps 9 and 10 just boost late years and inhibit early ones. The score of the direct logit attribution for mlp{layer} is {mean_mlp_in_year_score:.4f}")
# This seems to validate the observation that the vip mlps (9 and 10 it seems) actually just have good performance even without specific YY info
# The question is how? It is obvious that what the mlps are doing under mean ablation is the best thing they can do without YY info, but why do they
# perform well at all on that task? The model cant be implementing backup logic like 'if this previous part seems to have failed, just resort to
# promoting late years and inhibiting early ones', (i have a proof of this but its too long to put here) even though that is the behavior we see.
# Roughly speaking, we see that the mlp's normally are just boosting the correct years (that is, they boost years after the start YY and inhibit others),
# but when we mean ablate their input, they output the correct years for start YY=50.

# I find this strange and not obvious. It seems trivial that the average of the answers (the output of the mlp) should just be the average answer,
# but it seems weird that the output of the average input is the average output. This is because the average of the inputs shouldnt just be the 
# average input. The average of the residual streams at layer 10 for all sequences for 'the war lasted from XXYY to XX' for YY=range(2, 99) shouldnt
#  be equal to the residual for 'the war lasted from XX50 to XX' just because we can average over YY and get 50. right??????????

# we investigate below and in fact find that the mlp's output for the average input is NOT in general the average output, this
# just happens to be true on the whole dataset.

#%%

model.reset_hooks()
layer = 10
other_mlp_in = clean_cache[f'blocks.{layer}.ln2.hook_normalized'][:len(dataset)//2].mean(0)
model.add_hook(f'blocks.{layer}.ln2.hook_normalized', partial(act_patch, new_act=other_mlp_in))
other_mlp_in_ablation_logits, other_mlp_in_ablation_cache = model.run_with_cache(dataset.tokens)
other_mlp_in_dla = unembed(other_mlp_in_ablation_cache[f'blocks.{layer}.hook_mlp_out'])
other_mlp_in_year_dist = year_dist(other_mlp_in_dla, probs=False)
other_mlp_in_year_probs = year_dist(other_mlp_in_dla, probs=True)
other_mlp_in_year_score = score_dist(other_mlp_in_year_probs)
plot_dist(other_mlp_in_year_dist - other_mlp_in_year_dist.mean(), title=f"mlp{layer} dla when its input is averaged over the first half of the dataset. score: {other_mlp_in_year_score:.4f}")
other_mlp_in_final_dist = year_dist(other_mlp_in_ablation_logits, probs=True)
other_mlp_in_final_score = score_dist(other_mlp_in_final_dist)
plot_dist(other_mlp_in_final_dist, title=f"the model's final distribution with above ablation. score: {other_mlp_in_final_score:.4f} ({other_mlp_in_final_score - clean_year_probs_score:.4f})")
model.reset_hooks()
print("Here we give the mlp its input averaged over only the first half of the dataset. It appears identical to when we average over all examples.")
# If the mlp actually was just outputting the the average answer of the inputs that were averaged over, then we would see the mlp
# outputting the correct years for the average year of the first half of the dataset (the correct years for YY=25).

#%%
model.reset_hooks()
layer = 10
other_mlp_in = other_cache[f'blocks.{layer}.ln2.hook_normalized'].mean(0)
model.add_hook(f'blocks.{layer}.ln2.hook_normalized', partial(act_patch, new_act=other_mlp_in))
other_mlp_in_ablation_logits, other_mlp_in_ablation_cache = model.run_with_cache(dataset.tokens)
other_mlp_in_dla = unembed(other_mlp_in_ablation_cache[f'blocks.{layer}.hook_mlp_out'])
other_mlp_in_year_dist = year_dist(other_mlp_in_dla, probs=False)
other_mlp_in_year_probs = year_dist(other_mlp_in_dla, probs=True)
other_mlp_in_year_score = score_dist(other_mlp_in_year_probs)
plot_dist(other_mlp_in_year_dist - other_mlp_in_year_dist.mean(), title=f"mlp{layer} dla (on the original dataset) when its input is patched in with its averaged value on the whole other_dataset. score: {other_mlp_in_year_score:.4f}")
other_mlp_in_final_dist = year_dist(other_mlp_in_ablation_logits, probs=True)
other_mlp_in_final_score = score_dist(other_mlp_in_final_dist)
plot_dist(other_mlp_in_final_dist, title=f"the model's final distribution with above ablation. score: {other_mlp_in_final_score:.4f} ({other_mlp_in_final_score - clean_year_probs_score:.4f})")
model.reset_hooks()
print("When we instead patch into the mlp's input the values from the other_dataset (where YY is only 2), we see the mlp fails to output useful logits.")
print("(it looks about the same when we patch in a single activation from the other_dataset as opposed to the mean, so the mean here doesnt seem v important)")
# It seems most likely to me that averaging over the whole dataset really is erasing some relevant YY information.
# But averaged input should still contain information like 'im talking about a war' and 'i should be predicting a year following another year',
# and for that task it is useful to learn the inductive bias of 'later years are more likely to be valid than early ones'. This explains why
# the mlp shows this behavior (presumably) in absence of specific YY info. And in fact we see this bias in the normal dla pattern; the actual 
# performance on early years is poorer by some metrics (as in if YY=05, the model doesnt like to just predict 06, 10, 15, or other smallish numbers
# that are close to 05, and has a very wide distribution.)

# We investigate this apparent inductive bias below

#%%

layer = 10
first_year_pos_unembed = unembed(clean_cache[f'blocks.{layer}.hook_mlp_out'])
# This is basically what mlp{layer} is contributing to the model's guess about what the first year is. As in what this mlp boosts for the YY in 'the war lasted from XXYY to XX'.
first_year_unembed_year_logits = year_dist(first_year_pos_unembed, probs=False, seq_pos=5)
first_yeat_unembed_year_probs = year_dist(first_year_pos_unembed, probs=True, seq_pos=5)
plot_dist(first_year_unembed_year_logits - first_year_unembed_year_logits.mean(), title=f"mlp{layer} dla for the YY token in the normal dataset. score: {score_dist(first_yeat_unembed_year_probs):.4f}")
print("""we see that even when the model just doing a general 'predict year completion' task, it has some bias towards predicting
years which are later in the century. although the pattern is noisier and a bit weaker than for the specific next-year-prediction
task.""")

pos_5_to_all_hook = functools.partial(act_patch, new_act=clean_cache[f'blocks.{layer}.hook_mlp_out'][:, 5].unsqueeze(1))
pos_5_to_all_logits = model.run_with_hooks(dataset.tokens, fwd_hooks=[(f'blocks.{layer}.hook_mlp_out', pos_5_to_all_hook)])
pos_5_to_all_out_logits = year_dist(pos_5_to_all_logits, probs=False)
pos_5_to_all_out_probs = year_dist(pos_5_to_all_logits, probs=True)
plot_dist(pos_5_to_all_out_logits - pos_5_to_all_out_logits.mean(), title=f"output logits where mlp{layer}'s final output is its output at seq=5. score: {score_dist(pos_5_to_all_out_probs):.4f}")
plot_dist(pos_5_to_all_out_probs, title=f"the model's final distribution with above ablation. score: {score_dist(pos_5_to_all_out_probs):.4f} ({score_dist(pos_5_to_all_out_probs) - clean_year_probs_score:.4f})")
print("""to demonstrate, for mlp10 and 9, by patching in mlp_out[:, 5] -> mlp[out:, :] during the forward pass, we see this bias accounts for more
than half (~-0.15 change) of the mlp's performance. I also tried giving the models just the year tokens, as in ['<|endoftext|>', ' 11', '45'] and
checked the mlp9,10 and they do NOT exhibit the same later-years-more bias as we see here. This is strongly suggestive that these MLPS are actually
doing stuff specific to year completion, not general number completion.""")

# A few speculations about this before moving on to other parts of the circuit.
# potentially the later numbers are just actually globally more common and this mlp is contributing to that
# potentially people just talk about years later in the century more than earlier ones
# potentially the model actually has learned this bias just for next-year-completion but it accounts
# for the i-am-doing-next-year-completion detection circuit being faulty so it just sprinkles in some of that bias to be safe and hedge its bets

# interpretability is dark and full of terrors

#%%

print("""
So what have we learned and what is next?
 - mlps 9 and 10 are doing almost all of the correct logit contributions.
 - most of their contributions are static (independent of YY), and consists of always boosting YYs late in the century.
 - The mlps only exhibit this bias when they are predicting years, not 4-digit numbers in general.
 - The MLP does actually have a YY-dependent component, like its obvious from the normal DLA that the mlp is learning a crisp
 rule and is doing work besides the static bias.
 - But a large part of the performance IS actually static.
 - So now I am kind of meh on the idea of continuing becuase it all seems less clean?
    - like much of the performance is just *there* all the time.
        - which means there is some 'detect that we are doing x task and respond appropriately' circuit but it will all just be a bit weaker
    - but i also think that mechinterp on very clean algorithmic capabilities is easy but not as productive as fuzzy, fucked up capabilities.
    - but also I am a beginner and clean capabilities probably ARE what i should focus on to learn the ropes
ehhhh?
""")

#%%