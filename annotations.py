from mechlibs import *
"""
This is skeleton code for the common transformer layers, with intermediate
results annotated with their names in the ActivationCache object.
"""

def embed_forward(self, tokens: Int[Tensor, "batch position"]) -> Float[Tensor, "batch position d_model"]:
    return self.W_E[tokens] # blocks.x.hook_embed

def pos_embed_forward(self, tokens: Int[Tensor, "batch position"]) -> Float[Tensor, "batch position d_model"]:
    return self.W_pos[t.arange(tokens.shape[-1]).unsqueeze(0)] # blocks.x.hook_pos_embed

def ln_forward(self, resid: Float[Tensor, "batch posn d_model"]) -> Float[Tensor, "batch posn d_model"]:
    # the values are different if we use fold_ln
    mean, var = resid.mean(dim=-1, keepdims=True), resid.var(dim=-1, keepdims=True, unbiased=False)
    normalized = (resid - mean) / (var + self.cfg.layer_norm_eps).sqrt() # blocks.x.lnz.hook_normalized
    return normalized * self.w + self.b # blocks.x.ln.hook_scale


def attention_forward(self, resid: Float[Tensor, "batch posn d_model"]) -> Float[Tensor, "batch posn d_model"]:
    posn = resid.shape[1]
    Q = einops.einsum(resid, self.W_Q, "batch posn d_model, n_heads d_model d_head -> batch posn n_heads d_head") + self.b_Q # blocks.x.attn.hook_q
    K = einops.einsum(resid, self.W_K, "batch posn d_model, n_heads d_model d_head -> batch posn n_heads d_head") + self.b_K # blocks.x.attn.hook_k
    V = einops.einsum(resid, self.W_V, "batch posn d_model, n_heads d_model d_head -> batch posn n_heads d_head") + self.b_V # blocks.x.attn.hook_v
    QK = einops.einsum(Q, K, "batch qposn n_heads d_head, batch kposn n_heads d_head -> batch n_heads qposn kposn") / np.sqrt(self.cfg.d_head) 
    QK[:,:,*t.triu_indices(posn, posn, offset=1)] = self.IGNORE                                                              # blocks.x.attn.hook_attn_scores
    A = QK.softmax(dim=-1)                                                                                                   # blocks.x.attn.hook_pattern
    z = einops.einsum(A, V, "batch n_heads qposn kposn, batch kposn n_heads d_head -> batch qposn n_heads d_head")           # blocks.x.attn.hook_z
    result = einops.einsum(z, self.W_O, "batch qposn n_heads d_head, n_heads d_head d_model -> batch qposn d_model")         # blocks.x.attn.hook_result
    out = result.sum(dim=2) + self.b_O                                                                                       # blocks.x.hook_attn_out
    return out

def mlp_forward(self, normalized_resid_mid: Float[Tensor, "batch posn d_model"]) -> Float[Tensor, "batch posn d_model"]:
    acts_pre = normalized_resid_mid @ self.W_in + self.b_in # blocks.x.mlp.hook_pre
    acts_post = self.activation(acts_pre) # blocks.x.mlp.hook_post
    return acts_post @ self.W_out + self.b_out # blocks.x.hook_mlp_out

def unembed_forward(self, normalized_resid_final: Float[Tensor, "batch position d_model"]) -> Float[Tensor, "batch position d_vocab"]:
    return normalized_resid_final @ self.W_U + self.b_U # logits

# 'hook_embed',
# 'hook_pos_embed',
# 'blocks.0.hook_resid_pre',
# 'blocks.0.ln1.hook_scale',
# 'blocks.0.ln1.hook_normalized',
# 'blocks.0.attn.hook_q',
# 'blocks.0.attn.hook_k',
# 'blocks.0.attn.hook_v',
# 'blocks.0.attn.hook_attn_scores',
# 'blocks.0.attn.hook_pattern',
# 'blocks.0.attn.hook_z',
# 'blocks.0.hook_attn_out',
# 'blocks.0.hook_resid_mid',
# 'blocks.0.ln2.hook_scale',
# 'blocks.0.ln2.hook_normalized',
# 'blocks.0.mlp.hook_pre',
# 'blocks.0.mlp.hook_post',
# 'blocks.0.hook_mlp_out',
# 'blocks.0.hook_resid_post',
# . . .
# 'blocks.n.hook_resid_post',
# 'ln_final.hook_scale',
# 'ln_final.hook_normalized'