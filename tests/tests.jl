using PyCall

torch = pyimport("torch")

include("../app/train_gpt2.jl")

py"""

import os
import math
import glob
import struct
import inspect
from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
# import torch._inductor.config as config
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.distributed.optim import ZeroRedundancyOptimizer
import torch.distributed as dist

# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the GPT-2 model

class NewGELU(nn.Module):
    def forward(self, input):
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))

# using a global to toggle flash-attention
FLASH = 0

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.LLMC_RESIDUAL_SCALE_FLAG = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # not really a 'bias', more of a mask, but following the OpenAI/HF naming though
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        if FLASH:
            # flashattention
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            # manual implementation of attention
            # this materializes the large (T,T) matrix for all the queries and keys
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = NewGELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.LLMC_RESIDUAL_SCALE_FLAG = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

# -----------------------------------------------------------------------------
# The main GPT-2 model

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.LLMC_SKIP_INIT = 1 # don't init this one, we will tie weights
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights, use a torch rng object to be very careful
        self.init_rng = torch.Generator()
        self.init_rng.manual_seed(42)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # apply special scaled init to the residual projections, per GPT-2 paper
            std = 0.02 if not hasattr(module, 'LLMC_RESIDUAL_SCALE_FLAG') else 0.02/math.sqrt(2 * self.config.n_layer)
            # we want to skip initializing lm_head, which shares parameters with wte
            # and wte was already initialized down below during the Embedding init
            if not hasattr(module, 'LLMC_SKIP_INIT'):
                torch.nn.init.normal_(module.weight, mean=0.0, std=std, generator=self.init_rng)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02, generator=self.init_rng)

    def forward(self, idx, targets=None, return_logits=True):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = tok_emb + pos_emb

        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        # there are performance reasons why not returning logits is prudent, if not needed
        if not return_logits:
            logits = None

        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        # Loads pretrained GPT-2 model weights from huggingface
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type, zero_stage):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print0(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print0(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        print0(f"using fused AdamW: {use_fused}")
        if zero_stage == 1:
            print0("using ZeroRedundancyOptimizer")
            optimizer = ZeroRedundancyOptimizer(**optim_groups[0], optimizer_class=torch.optim.AdamW,
                                                lr=learning_rate, betas=betas, fused=use_fused)
            optimizer.add_param_group(optim_groups[1])
        else:
            print0("using regular AdamW")
            optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, fused=use_fused)
        return optimizer

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):

        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
"""

using Test
## test NewGELU


# Create a random tensor (uniformly distributed between 0 and 1)
random_tensor_th = torch.rand(1, 3, 4).float()  # Creates a 3x4 tensor
println("Random Tensor (Uniform):")
println(random_tensor_th)
randmon_julia_array = random_tensor_th.numpy()
println(randmon_julia_array)

gelu_py = py"NewGELU"()

gelu_py_res = gelu_py(random_tensor_th)
println("Python results:")
println(gelu_py_res)

gelu_jl_res = Array{Float32, 3}(undef, 1, 3, 4)

gelu_forward(gelu_jl_res, randmon_julia_array, 12)

println("Julia results:")
println(gelu_jl_res)

# println(gelu_jl_res - gelu_py_res.numpy())

@test gelu_jl_res ≈ gelu_py_res.numpy() atol=1e-4

## test ------------------------------ matmul_forward ------------------------------------
block_size=1024
vocab_size=50257
padded_vocab_size=
n_layer=12
n_head=12
n_embd=768
C=n_embd

B2 = 4
T2 = 64

# B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
x2_py = torch.rand(B2, T2, n_embd).float()
x2_py.requires_grad_()

c_attn_py = py"nn.Linear"(n_embd, 3 * n_embd)
# py"""
# Initialize weights using Xavier uniform initialization
py"nn.init.xavier_uniform_"(c_attn_py.weight)
# Initialize biases to zeros
py"nn.init.zeros_"(c_attn_py.bias)
py"nn.init.normal_"(c_attn_py.bias, mean=0.0, std=1.0)
# # Print initialized weights and biases
# println("Weights:", c_attn_py.weight)
# println("Biases:", c_attn_py.bias)
y2_py = c_attn_py(x2_py)
y2_loss_py = torch.sum(y2_py)
y2_loss_py.backward()
# println("x2 py: ")
# println(y2_py)

#--
x2_jl = x2_py.detach().numpy()
y2_jl = zeros(Float32, y2_py.size())

matmul_forward(y2_jl, x2_jl, c_attn_py.weight.detach().numpy(), Int32(B2), Int32(T2), Int32(n_embd), Int32(3 * n_embd), bias=c_attn_py.bias.detach().numpy())
# println("julia res: ")
# println(y2_jl)

@test y2_jl ≈ y2_py.detach().numpy() atol=1e-3

### ========================== test matmul_backward ================================================

dx2_jl = zeros(Float32, (B2, T2, n_embd))
dw_jl = zeros(Float32, c_attn_py.weight.size())
db_jl = zeros(Float32, c_attn_py.bias.size())
dy2_jl = ones(Float32, size(y2_jl))


matmul_backward(dx2_jl, dw_jl, dy2_jl, x2_jl,  c_attn_py.weight.detach().numpy(), Int32(B2), Int32(T2), Int32(n_embd), Int32(3 * n_embd), dbias=db_jl);

@test dw_jl ≈ c_attn_py.weight.grad.detach().numpy() atol=1e-10
@test db_jl ≈ c_attn_py.bias.grad.detach().numpy() atol=1e-10
@test dx2_jl ≈ x2_py.grad.detach().numpy() atol=1e-3

## ------------------- check attention_forward ----------------------

py"""

# FLASH = False
def attention_forward_py(qkv, B, T, C, n_embd, n_head, block_size):
    # qkv = self.c_attn(x)
    q, k, v = qkv.split(n_embd, dim=2)
    k = k.view(B, T, n_head, C // n_head).transpose(1, 2) # (B, nh, T, hs)
    q = q.view(B, T, n_head, C // n_head).transpose(1, 2) # (B, nh, T, hs)
    v = v.view(B, T, n_head, C // n_head).transpose(1, 2) # (B, nh, T, hs)
    if FLASH:
        # flashattention
        # y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        pass
    else:
        self_bias = torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
        # manual implementation of attention
        # this materializes the large (T,T) matrix for all the queries and keys
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self_bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
    y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
    return y, att
"""



block_size = 1024
vocab_size = 50257
n_layer = 12
n_head = 12


# B2 = 1
# T2 = 3
# C = n_embd = 4
# n_head = 2
# block_size = 8

# float* l_qkv = acts.qkv + l * B * T * 3*C;
x3_qkv_py = torch.rand(B2, T2, 3 * n_embd).float()
x3_qkv_py.requires_grad_()

y3_py, att_py = py"attention_forward_py"(x3_qkv_py, B2, T2, C, n_embd, n_head, block_size)
att_py.retain_grad()

x3_qkv_jl = x3_qkv_py.detach().numpy()

y3_jl = zeros(Float32, y3_py.size())

preatt = Array{Float32, 4}(undef, B2, n_head, T2, T2) #// preatt
att = Array{Float32, 4}(undef, B2, n_head, T2, T2) #// att


attention_forward(y3_jl, preatt, att, x3_qkv_jl, Int32(B2), Int32(T2), Int32(C), Int32(n_head))

@test att ≈ att_py.detach().numpy() atol=1e-2

@test y3_jl ≈ y3_py.detach().numpy() atol=1e-1

## test ============== attention backward ============================
attn_y3_py_loss = torch.sum(y3_py)
attn_y3_py_loss.backward()

d_y3_jl = ones(Float32, size(y3_jl))

dx3_qkv_jl = zeros(Float32, x3_qkv_py.grad.detach().size())
dpreatt_jl = zeros(Float32, B2, n_head, T2, T2)
datt_jl = zeros(Float32, att_py.detach().size())

attention_backward(dx3_qkv_jl, dpreatt_jl, datt_jl, d_y3_jl, x3_qkv_jl, att, Int32(B2), Int32(T2), Int32(C), Int32(n_head))
@test datt_jl ≈ att_py.grad.detach().numpy() atol=1e-2
@test dx3_qkv_jl ≈ x3_qkv_py.grad.detach().numpy() atol=1e-2


## test ------------ layer norm forward ---------------------------

x4_py = torch.rand(B2, T2, n_embd).float() * 5 + 1.3
x4_py.requires_grad_()

ln_py = py"nn.LayerNorm"(n_embd)
py"nn.init.constant_"(ln_py.weight, 1/3.0)  # Higher scaling factor"
# ln_py.weight.grad.zero_()
# ln_py.bias.grad.zero_()


y4_py = ln_py(x4_py)
# print(y4_py)

ln_weight = ln_py.weight.detach().numpy()
ln_bias = ln_py.bias.detach().numpy()

x4_jl = x4_py.detach().numpy()
y4_jl = zeros(Float32, y4_py.size())
 # float* l_ln1_mean = acts.ln1_mean + l * B * T;
 # float* l_ln1_rstd = acts.ln1_rstd + l * B * T;
ln_mean = zeros(Float32, B2, T2)
ln_rstd = zeros(Float32, B2, T2)
layernorm_forward(y4_jl, ln_mean, ln_rstd, x4_jl, ln_weight, ln_bias, Int32(B2), Int32(T2), Int32(n_embd))

# # print(y4_jl)
@test y4_jl ≈ y4_py.detach().numpy() atol=1e-2
@test ln_mean ≈ x4_py.detach().mean(dim=-1).detach().numpy()

var_py = x4_py.detach().var(dim=-1, unbiased=false, keepdim=true)

# Calculate reciprocal of standard deviation (rstd)
epsilon = 1e-5  # Small constant for numerical stability
rstd_py = torch.rsqrt(var_py + epsilon)
@test ln_rstd ≈ rstd_py.detach().numpy()

## test =======================layernorm_backward============================
ln_loss = torch.sum(y4_py)
ln_loss.backward()

# println(ln_py.weight.grad)

dln_out = ones(Float32, size(y4_jl))

dx4_jl = zeros(Float32, size(x4_jl))
dln_w = zeros(Float32, ln_py.weight.size())
dln_b = zeros(Float32, ln_py.bias.size())
layernorm_backward(dx4_jl, dln_w, dln_b, dln_out, x4_jl, ln_weight, ln_mean, ln_rstd, Int32(B2), Int32(T2), Int32(n_embd))

@test dx4_jl ≈ x4_py.grad.detach().numpy() atol=1e-2
@test dln_w ≈ ln_py.weight.grad.detach().numpy() atol=1e-3
@test dln_b ≈ ln_py.bias.grad.detach().numpy() atol=1e-2


## test ------------ softmax_forward ----------------------

logits_py = torch.rand(B2, T2, vocab_size).float()
probs_py = py"F.softmax"(logits_py, dim=-1)
# print(probs_py)

logits_jl = logits_py.detach().numpy()
probs_jl = zeros(Float32, probs_py.size())

softmax_forward(probs_jl, logits_jl, Int32(B2), Int32(T2), Int32(vocab_size), Int32(vocab_size))

# print(probs_jl)

@test probs_jl ≈ probs_py.detach().numpy() atol=1e-2

##---------------- test crossentropy_forward -----------------------------------------------
#         crossentropy_forward(model.acts.losses, model.acts.probs, targets, B, T, Vp);
println(probs_py.size())
targets_py = py"torch.randint"(0, vocab_size, (B2, T2))

log_probs_py = py"torch.log"(probs_py)
nll_loss_py = py"F.nll_loss"(log_probs_py.view(B2 * T2, vocab_size), targets_py.view(-1), reduce=false)
println(nll_loss_py)
# loss = py"F.cross_entropy"(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

losses_jl = zeros(Float32, (B2, T2))#nll_loss_py.size())
# probs_jl = probs_py.detach().numpy()
crossentropy_forward(losses_jl, probs_jl, targets_py.detach().to(dtype=torch.int32).numpy(), Int32(B2), Int32(T2), Vp);
println(losses_jl)

@test  losses_jl ≈ nll_loss_py.reshape((B2, T2)).detach().numpy() atol=1e-2

## --------------- test crossentropy_softmax_backward ---------------------------------------
logits_2_py = torch.rand(B2, T2, vocab_size).float()
logits_2_py.requires_grad_()

targets_2_py = py"torch.randint"(0, vocab_size, (B2, T2))


loss_2_py = py"F.cross_entropy"(logits_2_py.view(-1, logits_2_py.size(-1)), targets_2_py.view(-1), ignore_index=-1)
println(loss_2_py)


# Perform backward pass
loss_2_py.backward()  # Compute gradients

# Check gradients for logits
println("Gradients for logits (example):", logits_2_py.grad) 


dlogits_2_jl = zeros(Float32, (B2, T2, vocab_size))
dloss_mean_jl::Float32 = 1.0f0 / (B2*T2);
d_loss_jl = losses = Matrix{Float32}(undef, B2, T2)
d_loss_jl .= dloss_mean_jl
softmax_forward(probs_jl, logits_2_py.detach().numpy(), Int32(B2), Int32(T2), Int32(vocab_size), Int32(vocab_size))
crossentropy_softmax_backward(dlogits_2_jl, d_loss_jl, probs_jl, targets_2_py.detach().to(dtype=torch.int32).numpy(), Int32(B2), Int32(T2), Int32(vocab_size), Int32(vocab_size))

@test  dlogits_2_jl ≈ logits_2_py.grad.detach().numpy() atol=1e-9

#------------------------------- 