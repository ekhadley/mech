from mechlibs import *
device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")
MAIN = __name__ == "__main__"
t.set_grad_enabled(False);
t.manual_seed(42)
#%%

model = HookedTransformer.from_pretrained(
    "gpt2-small",
    center_unembed=True,
    center_writing_weights=True,
    fold_ln=True,
    refactor_factored_attn_matrices=True,
)
print(bold, green, model, endc)

#%%

def sample(model: HookedTransformer, prompt: str, ntok=50, show=True, showall=False) -> Union[Tensor, str]:
    toks = t.tensor(model.tokenizer(prompt)['input_ids'], device=device)
    out = t.tensor([], device=device)
    for n in range(ntok):
        preds = model(toks).argmax(-1)
        nextok = preds[:,-1]
        toks = t.cat([toks, nextok])
        out = t.cat([out, nextok])
        if showall: print(bold, cyan, model.tokenizer.decode(toks), endc)
    if showall or show: print(bold, cyan, model.tokenizer.decode(toks), endc)
    return out

def show_top(logits, k=5):
    #print(f"{blue} getting top {k} predictions for next token in: {model.to_str_tokens(prompt)}{endc}")
    #logits = model(prompt).squeeze()[-1,:]
    logits = logits.squeeze()[-1,:]
    topv, topi = t.topk(logits, k, dim=-1)
    print(cyan, {model.tokenizer.decode(e): round(logits[e].item(), 3) for e in topi}, endc)

#%%
############################################## dataset & modeling

days = [' Monday', ' Tuesday', ' Wednesday', ' Thursday', ' Friday', ' Saturday', ' Sunday']
daytokens = {day: model.tokenizer(day)['input_ids'][0] for day in days}

class weekday_datapoint:
    def __init__(self, day: int):
        self.day = days[day]
        self.correct_str = days[(day+1)%7]
        self.correct = daytokens[self.correct_str]
        self.prompt = f"If today is{self.day}, tomorrow is"
    
    def __repr__(self):
        return f"weekday_datapoint({self.day})"

class weekday_dataset:
    def __init__(self):
        self.data = [weekday_datapoint(i) for i in range(len(days))]
        self.correct = t.tensor([e.correct for e in self.data], device=device)
        self.prompts = [e.prompt for e in self.data]
        self.toks = t.tensor(model.tokenizer(self.prompts)['input_ids'], device=device)
    
    def __getitem__(self, i):
        return self.data[i]

weekdays = weekday_dataset()

def score_logits(logits, dataset=weekdays):
    assert logits.ndim == 3, f"expected 3d logits, got {logits.shape}"
    #maxlogits = logits[:,-1,:].max(dim=-1).values
    correctlogits = logits[:,-1,:][t.arange(7),dataset.correct]
    return correctlogits.mean().item()
    #return (correctlogits - maxlogits).mean().item()

#%%

clean_logits, clean_cache = model.run_with_cache(weekdays.prompts)
print(f"{lime}clean model score: {score_logits(clean_logits)}{endc}")

#%%

def patch_attn_head_out_to_avg(
        orig_z: Float[Tensor, "batch seq nhead dhead"],
        head_idx: int,
        clean_cache: ActivationCache,
        hook: HookPoint
) -> Float[Tensor, "batch seq nhead dhead"]:
    orig_z[:,:,head_idx] = clean_cache['z', hook.layer()].mean(0, keepdim=True)[:, :, head_idx]
    return orig_z

def patch_attn_head_out_to_zero(
        orig_z: Float[Tensor, "batch seq nhead dhead"],
        head_idx: int,
        clean_cache: ActivationCache,
        hook: HookPoint
) -> Float[Tensor, "batch seq nhead dhead"]:
    orig_z[:,:,head_idx,:] *= 0
    return orig_z

def corrupt_attn_head_out(
        orig_z: Float[Tensor, "batch seq nhead dhead"],
        head_idx: int,
        clean_cache: ActivationCache,
        hook: HookPoint
) -> Float[Tensor, "batch seq nhead dhead"]:
    for i in range(7):
        orig_z[i,:,head_idx,:] = clean_cache[hook.name][(i+2)%7,:,head_idx,:]
    return orig_z

def attn_head_mean_ablation(model=model, clean_cache=clean_cache, metric=score_logits):
    nlayer, nhead = model.cfg.n_layers, model.cfg.n_heads

    out = t.empty((nlayer, nhead), device=device)
    for layer in trange(nlayer):
        for head in range(nhead):
            hookname = f"blocks.{layer}.attn.hook_z"
            hook = partial(patch_attn_head_out_to_avg, head_idx=head, clean_cache=clean_cache)
            #hook = partial(corrupt_attn_head_out, head_idx=head, clean_cache=clean_cache)
            #hook = partial(patch_attn_head_out_to_zero, head_idx=head, clean_cache=clean_cache)
            logits = model.run_with_hooks(weekdays.prompts, fwd_hooks=[(hookname, hook)])
            out[layer, head] = metric(logits)
    out = (out - out.mean()) / out.std()
    return out

#%%

mean_ablation = attn_head_mean_ablation()
imshow(
    mean_ablation, 
    labels={"y": "Layer", "x": "Head"}, 
    title="dataset attention corrupted ablation",
    width=600
)

#%%

#promptidx = 2
#headlayer = 10
#headidx = 7

#pattern = clean_cache[f'blocks.{10}.attn.hook_pattern'][promptidx,7]
#cv.attention.attention_pattern(
#    attention = pattern,
#    tokens = model.to_str_tokens(weekdays.prompts[promptidx]),
#).show()


#rand_prompt = "The day after Monday is"
#_, random_cache = model.run_with_cache(rand_prompt)
#pattern = random_cache[f'blocks.{headlayer}.attn.hook_pattern'][0,headidx]
#cv.attention.attention_pattern(
#    attention = pattern,
#    tokens = model.to_str_tokens(rand_prompt),
#).show()


# when ablating heads with their average activation over all 7 prompts, head 9.1 causes the largest drop in performance (and 10.7 doesnt matter at all)
    # on the 'is' token, head 9.1 attends to the starting <endoftext> and the day of the week (about 80:20)
# when ablating heads with the activation they had on a different prompt, head 10.7 causes the largest drop in performance (and 9.1 is a bit less important than itself)
    # on the 'is' token, head 10.7 attends mostly to 'tomorrow' and the day of the week
