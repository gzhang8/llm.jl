using StaticArrays
using Statistics
using Base.Threads

const Tensor{T, N} = Union{Array{T, N}, SubArray{T, N}}

# ----------------------------------------------------------------------------
# all the individual layers' forward and backward passes
# B = batch_size, T = sequence_length, C = channels, V = vocab_size
# ===========================================
# this is the adding of token embeddings and positional embedding
@inbounds function encoder_forward(
    out::Array{Float32, 3},
    inp::Matrix{Int32}, wte::Matrix{Float32}, wpe::Matrix{Float32},
    B::Int32, T::Int32, C::Int32)
    # out is (B,T,C). At each position (b,t), a C-dimensional vector summarizing token & position
    # inp is (B,T) of integers, holding the token ids at each (b,t) position TODO: is Int32 enough for it: depend on token size
    # wte is (V,C) of token embeddings, short for "weight token embeddings"
    # wpe is (maxT,C) of position embeddings, short for "weight positional embedding"
    Threads.@threads for b in 1:B
        for t in 1:T
            # seek to the output position in out[b,t,:]
            # get the index of the token at inp[t, b]
            ix::Int32 = inp[t, b] + 1; # input has zeros indexing TODOinp1
            # seek to the position in wte corresponding to the token
            # seek to the position in wpe corresponding to the position
            # add the two vectors and store the result in out[b,t,:]
            for i in 1:C
                out[i, t, b] = wte[i, ix] + wpe[i, t];
            end
        end
    end
end

@inbounds function encoder_backward(
    dwte::Tensor{Float32, 2}, dwpe::Tensor{Float32, 2},
    dout::Tensor{Float32, 3}, inp::Tensor{Int32, 2},
    B::Int32, T::Int32, C::Int32)
    Threads.@threads for b in 1:B
        for t in 1:T
            # input has zeros indexing
            ix::Int32 = inp[t, b] + 1;
            for i in 1:C 
                d::Float32 = dout[i, t, b];
                dwte[i, ix] += d;
                dwpe[i, t] += d;
            end
        end
    end
end

@inbounds function layernorm_forward(
    out::Tensor{Float32, 3}, mean::Tensor{Float32, 2}, rstd::Tensor{Float32, 2},
    inp::Tensor{Float32, 3}, weight::Tensor{Float32, 1}, bias::Tensor{Float32, 1},
    B::Int32, T::Int32, C::Int32)
    # // reference: https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
    # // both inp and out are (B,T,C) of the activations
    # // mean and rstd are (B,T) buffers, to be used later in backward pass
    # // at each position (b,t) of the input, the C-dimensional vector
    # // of activations gets normalized, then scaled and shifted
    eps::Float32 = 1e-5;
    for b in 1:B
        for t in 1:T
            # // seek to the input position inp[b,t,:]
            # // calculate the mean
            m::Float32 = 0.0f0;
            for i in 1:C
                m += inp[i, t, b] #x[i];
            end
            m = m/C;
            # // calculate the variance (without any bias correction)
            v::Float32 = 0.0f0;
            for i in 1:C 
                xshift::Float32 = inp[i, t, b] - m;
                v += xshift * xshift;
            end
            v = v/C;
            # // calculate the rstd (reciprocal standard deviation)
            s::Float32 = 1.0f0 / sqrt(v + eps);
            # // seek to the output position in out[b,t,:]
            for i in 1:C 
                n::Float32 = (s * (inp[i, t, b] - m)); #// normalize
                o::Float32 = n * weight[i] + bias[i]; #// scale and shift
                out[i, t, b] = o; #// write
            end
            # // cache the mean and rstd for the backward pass later
            mean[t, b] = m;
            rstd[t, b] = s;
        end
    end
end


@inbounds function layernorm_backward(
    dinp::Tensor{Float32, 3}, dweight::Tensor{Float32, 1}, dbias::Tensor{Float32, 1},
    dout::Tensor{Float32, 3}, inp::Tensor{Float32, 3}, weight::Tensor{Float32, 1}, mean::Tensor{Float32, 2}, rstd::Tensor{Float32, 2},
    B::Int32, T::Int32, C::Int32)
    # calucated grad wrt inp, weight, bias, given output grad
    for b in 1:B
        for t in 1:T
            # // first: two reduce operations
            dnorm_mean::Float32 = 0.0f0;
            dnorm_norm_mean::Float32 = 0.0f0;
            for i in 1:C
                norm_bti::Float32 = (inp[i, t, b] - mean[t, b]) * rstd[t, b];
                dnorm_i::Float32 = weight[i] * dout[i, t, b];
                dnorm_mean += dnorm_i;
                dnorm_norm_mean += dnorm_i * norm_bti;
            end
            dnorm_mean = dnorm_mean / C;
            dnorm_norm_mean = dnorm_norm_mean / C;

            # // now iterate again and accumulate all the gradients
            for i in 1:C
                norm_bti::Float32 = (inp[i, t, b] - mean[t, b]) * rstd[t, b];
                dnorm_i::Float32 = weight[i] * dout[i, t, b] #dout_bt[i];
                # // gradient contribution to bias
                # same as dout, but accumulate over B and T
                dbias[i] += dout[i, t, b];
                # // gradient contribution to weight
                dweight[i] += norm_bti * dout[i, t, b];
                # // gradient contribution to input
                dval::Float64 = 0.0f0;
                dval += dnorm_i; # // term 1
                dval -= dnorm_mean; #// term 2
                dval -= norm_bti * dnorm_norm_mean; #// term 3
                dval *= rstd[t, b]; # // final scale
                dinp[i, t, b] += Float32(dval);
            end
        end
    end
end

@inbounds function matmul_forward_naive(
    out::Tensor{Float32, 3}, # (B, T, OC)
    inp::Tensor{Float32, 3}, weight::Tensor{Float32, 2},
    B::Int32, T::Int32, C::Int32, OC::Int32; bias::Tensor{Float32, 1}=Float32[])
    # // the most naive implementation of matrix multiplication
    # // this serves as an algorithmic reference, and as a fallback for
    # // unfriendly input shapes inside matmul_forward(), below.
    #pragma omp parallel for collapse(2)
    Threads.@threads for b in 1:B 
        for t in 1:T 
            for o in 1:OC 
                val::Float32 = if !isempty(bias)
                    bias[o]
                else
                    0.0f0
                end
                for i in 1:C
                    val += inp[i, t, b] * weight[i, o];
                end
                out[o, t, b] = val;
            end
        end
    end
end

const LOOP_UNROLL::Int32 = 8;

@inbounds function matmul_forward(
    out::Tensor{Float32, 3}, # (B, T, OC)
    inp::Tensor{Float32, 3}, weight::Tensor{Float32, 2},
    B::Int32, T::Int32, C::Int32, OC::Int32; bias::Tensor{Float32, 1}=Float32[])
    # // most of the running time is spent here and in matmul_backward
    # // therefore, the implementation below is very mildly optimized
    # // this function is otherwise identical to that of matmul_forward_naive()
    # // OC is short for "output channels"
    # // inp is (B,T,C), weight is (OC, C), bias is (OC)
    # // out will be (B,T,OC)

    # // make sure the tiled loop will be correct or fallback to naive version
    if (B*T % LOOP_UNROLL != 0)
        matmul_forward_naive(out, inp, weight, B, T, C, OC, bias=bias);
        return;
    end

    # // collapse the B and T loops into one and turn it into a strided loop.
    # // then we can tile the inner loop, and reuse the loaded weight LOOP_UNROLL many times
    #pragma omp parallel for
    Threads.@threads for obt in 1:LOOP_UNROLL:B * T #(int obt = 0; obt < B * T; obt += LOOP_UNROLL) {
        for o in 1:OC
            # // we'll keep LOOP_UNROLL many results in registers
            # float result[LOOP_UNROLL];
            result = @MVector zeros(Float32, Int64(LOOP_UNROLL))
            # // initialize the bias, if it exists
            for ibt in 1:LOOP_UNROLL
                result[ibt] = if !isempty(bias)
                    bias[o]
                else
                    0.0f0
                end
            end
            # // inner loops. Because we do LOOP_UNROLL steps of inner bt, we can cache
            # // the value of weight[i + o * C] and reuse it.
            # // we compile with -Ofast, so the compiler will turn the inner loop into FMAs
            for i in 1:C
                w::Float32 = weight[i, o]#weight[i + o * C];
                for ibt in 1:LOOP_UNROLL
                    bt = obt + ibt - 1;
                    result[ibt] += inp[(bt-1) * C + i] * w #inp[bt * C + i] * w;
                end
            end
            # // write back results to main memory
            for ibt in 1:LOOP_UNROLL
                bt = obt + ibt - 1;
                out[(bt-1) * OC + o] = result[ibt];
            end
        end
    end
end

@inbounds function matmul_backward(
    dinp::Tensor{Float32, 3}, dweight::Tensor{Float32, 2},
    dout::Tensor{Float32, 3}, inp::Tensor{Float32, 3}, weight::Tensor{Float32, 2},
    B::Int32, T::Int32, C::Int32, OC::Int32; dbias::Tensor{Float32, 1}=Vector{Float32}(undef, 0))
    # // most of the running time is spent here and in matmul_forward
    # // this backward could be done in a single "round" of loops
    # // but that doesn't afford an efficient parallelization strategy

    # // backward into inp first, parallelize over B,T
    #pragma omp parallel for collapse(2)
    Threads.@sync for b in 1:B
        for t in 1:T
            Threads.@spawn for o in 1:OC 
                d::Float32 = dout[o, t, b]
                for i in 1:C
                    dinp[i, t, b] += weight[i, o] * d
                end
            end
        end
    end
    # // backward into weight/bias, parallelize over output channels OC
    #pragma omp parallel for
    Threads.@threads for o in 1:OC
        for b in 1:B
            for t in 1:T
                d::Float32 = dout[o, t, b] # dout_bt[o];
                if !isempty(dbias) dbias[o] += d end
                for i in 1:C
                    dweight[i, o] += inp[i, t, b] * d;
                end
            end
        end
    end
end


@inbounds function attention_forward(
    out::Tensor{Float32, 3}, preatt::Tensor{Float32, 4}, att::Tensor{Float32, 4},
    inp::Tensor{Float32, 3},
    B::Int32, T::Int32, C::Int32, NH::Int32)
    # // input is (B, T, 3C) holding the query, key, value (Q, K, V) vectors
    #    for 3C in input, it is | Q | K | V | where Q contains all the q for NH heads
    # // preatt, att are (B, NH, T, T). NH = number of heads, T = sequence length
    # // that holds the pre-attention and post-attention scores (used in backward)
    # // output is (B, T, C)
    # // attention is the only layer that mixes information across time
    # // every other operation is applied at every (b,t) position independently
    # // (and of course, no layer mixes information across batch)
    C3::Int32 = C*3;
    hs::Int32 = C / NH; # // head size
    scale::Float32 = 1.0 / sqrt(hs);

    #pragma omp parallel for collapse(3)
    Threads.@sync for b in 1:B
        for t in 1:T
            for h in 1:NH
                Threads.@spawn begin
                    query_t = @view inp[((h-1) * hs + 1):((h-1) * hs + hs) , t, b]

                    # // pass 1: calculate query dot key and maxval
                    maxval::Float32 = -Inf
                    for t2 in 1:t
                        key_t2 = @view inp[((h-1) * hs + C + 1):((h-1) * hs + hs + C) , t2, b]

                        # // (query_t) dot (key_t2)
                        val::Float32 = 0.0f0;
                        for i in 1:hs 
                            val += query_t[i] * key_t2[i];
                        end
                        val *= scale;
                        if val > maxval
                            maxval = val;
                        end

                        preatt[t2, t, h, b] = val
                    end

                    # // pass 2: calculate the exp and keep track of sum
                    # // maxval is being calculated and subtracted only for numerical stability
                    expsum::Float32 = 0.0f0;
                    for t2 in 1:t
                        expv::Float32 = exp(preatt[t2, t, h, b] - maxval);
                        expsum += expv;
                        att[t2, t, h, b] = expv;
                    end
                    expsum_inv::Float32 = if expsum == 0.0f0
                        0.0f0
                    else
                        1.0f0 / expsum
                    end

                    # // pass 3: normalize to get the softmax
                    for t2 in 1:T 
                        if t2 <= t
                            att[t2, t, h, b] *= expsum_inv;
                        else
                            # // causal attention mask. not strictly necessary to set to zero here
                            # // only doing this explicitly for debugging and checking to PyTorch
                            att[t2, t, h, b] = 0.0f0;
                        end
                    end

                    # // pass 4: accumulate weighted values into the output of attention
                    for i in 1:hs
                        out[(h - 1) * hs + i, t, b] = 0.0f0
                    end
                    for t2 in 1:t
                        att_btht2::Float32 = att[t2, t, h, b]
                        for i in 1:hs
                            out[(h - 1) * hs + i, t, b] += att_btht2 * inp[(h - 1) * hs + C*2 + i, t2, b]#value_t2[i];
                        end
                    end
                end
            end
        end
    end
end



@inbounds function attention_backward(
    dinp::Tensor{Float32, 3}, dpreatt::Tensor{Float32, 4}, datt::Tensor{Float32, 4},
    dout::Tensor{Float32, 3}, inp::Tensor{Float32, 3}, att::Tensor{Float32, 4},
    B::Int32, T::Int32, C::Int32, NH::Int32)
    # // inp/dinp are (B, T, 3C) Q,K,V
    # // att/datt/dpreatt are (B, NH, T, T)
    # // dout is (B, T, C)
    C3::Int32 = Int32(C*3);
    hs::Int32 = Int32(div(C, NH)); # // head size, integer division
    scale::Float32 = 1.f0 / sqrt(hs);

    for b in 1:B
        for t in 1:T
            for h in 1:NH

                # // backward pass 4, through the value accumulation
                for t2 in 1:t
                    for i in 1:hs
                        # // in the forward pass this was:
                        # // out_bth[i] += att_bth[t2] * value_t2[i];
                        # // so now we have:
                        datt[t2, t, h, b] += inp[(h-1) * hs + C*2 + i, t2, b] * dout[(h-1) * hs + i, t, b];
                        dinp[(h-1) * hs + C*2 + i, t2, b] += att[t2, t, h, b] * dout[(h-1) * hs + i, t, b];
                    end
                end

                # // backward pass 2 & 3, the softmax
                # // note that softmax (like e.g. tanh) doesn't need the input (preatt) to backward
                for t2 in 1:t
                    for t3 in 1:t
                        indicator::Float32 = if t2 == t3
                            1.0f0
                        else
                            0.0f0
                        end
                        local_derivative::Float32 = att[t2, t, h, b] * (indicator - att[t3, t, h, b]);
                        dpreatt[t3, t, h, b] += local_derivative * datt[t2, t, h, b];
                    end
                end

                # // backward pass 1, the query @ key matmul
                for t2 in 1:t
                    for i in 1:hs
                        # // in the forward pass this was:
                        # // preatt_bth[t2] += (query_t[i] * key_t2[i]) * scale;
                        # // so now we have:
                        dinp[(h-1) * hs + i, t, b] += inp[(h-1) * hs + C + i, t2, b] * dpreatt[t2, t, h, b] * scale;
                        dinp[(h-1) * hs + C + i, t2, b] += inp[(h-1) * hs + i, t, b] * dpreatt[t2, t, h, b] * scale;
                    end
                end
            end
        end
    end
end

#define GELU_SCALING_FACTOR sqrtf(2.0f / M_PI)
@inline function GELU_SCALING_FACTOR() 
    return sqrt(2.0f0 / π)
end

@inbounds function gelu_forward(out::Tensor{Float32, 3}, inp::Tensor{Float32, 3}, N::Int64)
    # // (approximate) GeLU elementwise non-linearity in the MLP block of Transformer
    Threads.@threads for i in 1:N
        x::Float32 = inp[i];
        cube::Float32 = 0.044715f0 * x * x * x;
        out[i] = 0.5f0 * x * (1.0f0 + tanh(GELU_SCALING_FACTOR() * (x + cube)));
    end
end

@inbounds function gelu_backward(dinp::Tensor{Float32, 3}, inp::Tensor{Float32, 3}, dout::Tensor{Float32, 3}, N::Int64)
    Threads.@threads for i in 1:N
        x::Float32 = inp[i];
        cube::Float32 = 0.044715f0 * x * x * x;
        tanh_arg::Float32 = GELU_SCALING_FACTOR() * (x + cube);
        tanh_out::Float32 = tanh(tanh_arg);
        coshf_out::Float32 = cosh(tanh_arg);
        sech_out::Float32 = 1.0f0 / (coshf_out * coshf_out);
        local_grad::Float32 = 0.5f0 * (1.0f0 + tanh_out) + x * 0.5f0 * sech_out * GELU_SCALING_FACTOR() * (1.0f0 + 3.0f0 * 0.044715f0 * x * x);
        dinp[i] += local_grad * dout[i];
    end
end

function residual_forward(out::Tensor{Float32, 3}, inp1::Tensor{Float32, 3}, inp2::Tensor{Float32, 3})
    out .= inp1 .+ inp2;
end

function residual_backward(dinp1::Tensor{Float32, 3}, dinp2::Tensor{Float32, 3}, dout::Tensor{Float32, 3})
    dinp1 .+= dout;
    dinp2 .+= dout; # TODO why +=
end

@inbounds function softmax_forward(probs::Tensor{Float32, 3}, logits::Tensor{Float32, 3}, B::Int32, T::Int32, V::Int32, Vp::Int32)
    # // output: probs are (B,T,Vp) of the probabilities (sums to 1.0 in each b,t position)
    # // input: logits is (B,T,Vp) of the unnormalized log probabilities
    # // Vp is the padded vocab size (for efficiency), V is the "real" vocab size
    # // example: Vp is 50304 and V is 50257
    #pragma omp parallel for collapse(2)
    Threads.@sync for b in 1:B
        for t in 1:T
            Threads.@spawn begin
                # // probs <- softmax(logits)

                # // maxval is only calculated and subtracted for numerical stability
                maxval::Float32 = -Inf
                for i in 1:V
                    if (logits[i, t, b] > maxval) 
                        maxval = logits[i, t, b];
                    end
                end
                sum::Float32 = 0.0f0;
                for i in 1:V
                    probs[i, t, b] = exp(logits[i, t, b] - maxval);
                    sum += probs[i, t, b];
                end
                # // note we only loop to V, leaving the padded dimensions
                for i in 1:V
                    probs[i, t, b] /= sum;
                end
                # // for extra super safety we may wish to include this too,
                # // forcing the probabilities here to be zero, but it shouldn't matter
                for i in V+1:Vp
                    probs[i, t, b] = 0.0f0;
                end
            end
        end
    end
end

@inbounds function crossentropy_forward(
    losses::Tensor{Float32, 2},
    probs::Tensor{Float32, 3}, targets::Tensor{Int32, 2},
    B::Int32, T::Int32, Vp::Int32) 
    # // output: losses is (B,T) of the individual losses at each position
    # // input: probs are (B,T,Vp) of the probabilities
    # // input: targets is (B,T) of integers giving the correct index in logits
    Threads.@threads for b in 1:B
        for t in 1:T
            # // loss = -log(probs[target])
            ix::Int32 = targets[t, b] + 1; # target is 0 indexed
            losses[t, b] = -log(probs[ix, t, b]);
        end
    end
end

@inbounds function crossentropy_softmax_backward(
    dlogits::Tensor{Float32, 3},
    dlosses::Tensor{Float32, 2}, probs::Tensor{Float32, 3}, targets::Tensor{Int32, 2},
    B::Int32, T::Int32, V::Int32, Vp::Int32)
    # // backwards through both softmax and crossentropy
    for b in 1:B
        for t in 1:T
            dloss::Float32 = dlosses[t, b];
            ix::Int32 = targets[t, b];
            # // note we only loop to V, leaving the padded dimensions
            # // of dlogits untouched, so gradient there stays at zero
            for i in 1:V
                p::Float32 = probs[i, t, b];
                indicator::Float32 = if i == ix + 1 # Note julia is 1 indexed
                    1.0f0
                else
                    0.0f0
                end
                dlogits[i, t, b] += (p - indicator) * dloss;
            end
        end
    end
end

# // ----------------------------------------------------------------------------
# // GPT-2 model definition

struct GPT2Config
    max_seq_len::Int32  #; // max sequence length, e.g. 1024
    vocab_size::Int32   #; // vocab size, e.g. 50257
    padded_vocab_size::Int32  #; // padded to e.g. %128==0, 50304
    num_layers::Int32  #; // number of layers, e.g. 12
    num_heads::Int32   #; // number of heads in attention, e.g. 12
    channels::Int32    #; // number of channels, e.g. 768
end

# // the parameters of the model
#define NUM_PARAMETER_TENSORS 16
struct ParameterTensors
    wte::Matrix{Float32}  #; // (V, C)
    wpe::Matrix{Float32}  #; // (maxT, C)
    ln1w::Matrix{Float32}  #; // (L, C)
    ln1b::Matrix{Float32}  #; // (L, C)
    qkvw::Array{Float32, 3}  #; // (L, 3*C, C)
    qkvb::Matrix{Float32}  #; // (L, 3*C)
    attprojw::Array{Float32, 3}  #; // (L, C, C)
    attprojb::Matrix{Float32}  #; // (L, C)
    ln2w::Matrix{Float32}  #; // (L, C)
    ln2b::Matrix{Float32}  #; // (L, C)
    fcw::Array{Float32, 3}  #; // (L, 4*C, C)
    fcb::Matrix{Float32}  #; // (L, 4*C)
    fcprojw::Array{Float32, 3}  #; // (L, C, 4*C)
    fcprojb::Matrix{Float32}  #; // (L, C)
    lnfw::Vector{Float32}  #; // (C)
    lnfb::Vector{Float32}  #; // (C)
    NUM_PARAMETER_TENSORS::Int32
end

function get_tensors(params::ParameterTensors)::Vector{AbstractArray}
    res = [
        params.wte, params.wpe,
        params.ln1w, params.ln1b,
        params.qkvw, params.qkvb,
        params.attprojw, params.attprojb,
        params.ln2w, params.ln2b,
        params.fcw, params.fcb,
        params.fcprojw, params.fcprojb,
        params.lnfw, params.lnfb
    ]
end

function fill_in_parameter_sizes(param_sizes::Vector{Int64}, config::GPT2Config)
    Vp::Int64 = config.padded_vocab_size;
    C::Int64 = config.channels;
    maxT::Int64 = config.max_seq_len;
    L::Int64 = config.num_layers;
    param_sizes[0] = Vp * C; #// wte
    param_sizes[1] = maxT * C; #// wpe
    param_sizes[2] = L * C; #// ln1w
    param_sizes[3] = L * C; #// ln1b
    param_sizes[4] = L * (3 * C) * C; #// qkvw
    param_sizes[5] = L * (3 * C); #// qkvb
    param_sizes[6] = L * C * C; #// attprojw
    param_sizes[7] = L * C; #// attprojb
    param_sizes[8] = L * C; #// ln2w
    param_sizes[9] = L * C; #// ln2b
    param_sizes[10] = L * (4 * C) * C; #// fcw
    param_sizes[11] = L * (4 * C); #// fcb
    param_sizes[12] = L * C * (4 * C); #// fcprojw
    param_sizes[13] = L * C; #// fcprojb
    param_sizes[14] = C; #// lnfw
    param_sizes[15] = C; #// lnfb
end


function malloc_and_point_parameters(config::GPT2Config)::ParameterTensors
    Vp::Int64 = config.padded_vocab_size;
    C::Int64 = config.channels;
    maxT::Int64 = config.max_seq_len;
    L::Int64 = config.num_layers;
    wte = zeros(Float32, C, Vp) #// wte
    wpe = zeros(Float32, C, maxT) #// wpe
    ln1w = zeros(Float32, C, L) #// ln1w
    ln1b = zeros(Float32, C, L) #// ln1b
    qkvw = zeros(Float32, C, (3 * C), L) #// qkvw TODO check correctness
    qkvb = zeros(Float32, 3 * C, L) #// qkvb
    attprojw = zeros(Float32, C, C, L) #// attprojw
    attprojb = zeros(Float32, C, L) #// attprojb
    ln2w = zeros(Float32, C, L) #// ln2w
    ln2b = zeros(Float32, C, L) #// ln2b
    fcw = zeros(Float32, C, (4 * C), L) #// fcw
    fcb = zeros(Float32, 4 * C, L) #// fcb
    fcprojw = zeros(Float32, 4 * C, C, L) #// fcprojw
    fcprojb = zeros(Float32, C, L) #// fcprojb
    lnfw = zeros(Float32, C) #// lnfw
    lnfb = zeros(Float32, C) #// lnfb

    NUM_PARAMETER_TENSORS::Int32 = 16
    ParameterTensors(
        wte,
        wpe,
        ln1w,
        ln1b,
        qkvw,
        qkvb,
        attprojw,
        attprojb,
        ln2w,
        ln2b,
        fcw,
        fcb,
        fcprojw,
        fcprojb,
        lnfw,
        lnfb,
        NUM_PARAMETER_TENSORS
    )
end


#define NUM_ACTIVATION_TENSORS 23
struct ActivationTensors
    encoded::Array{Float32, 3} # // (B, T, C)
    ln1::Array{Float32, 4} #// (L, B, T, C)
    ln1_mean::Array{Float32, 3} #// (L, B, T)
    ln1_rstd::Array{Float32, 3} #// (L, B, T)
    qkv::Array{Float32, 4} #// (L, B, T, 3*C)
    atty::Array{Float32, 4} #// (L, B, T, C)
    preatt::Array{Float32, 5} #// (L, B, NH, T, T)
    att::Array{Float32, 5} #// (L, B, NH, T, T)
    attproj::Array{Float32, 4} #// (L, B, T, C)
    residual2::Array{Float32, 4} #// (L, B, T, C)
    ln2::Array{Float32, 4} #// (L, B, T, C)
    ln2_mean::Array{Float32, 3} #// (L, B, T)
    ln2_rstd::Array{Float32, 3} #// (L, B, T)
    fch::Array{Float32, 4} #// (L, B, T, 4*C)
    fch_gelu::Array{Float32, 4} #// (L, B, T, 4*C)
    fcproj::Array{Float32, 4} #// (L, B, T, C)
    residual3::Array{Float32, 4} #// (L, B, T, C)
    lnf::Array{Float32, 3} #// (B, T, C)
    lnf_mean::Matrix{Float32} #// (B, T)
    lnf_rstd::Matrix{Float32} #// (B, T)
    logits::Array{Float32, 3} #// (B, T, V)
    probs::Array{Float32, 3} #// (B, T, V)
    losses::Matrix{Float32} #// (B, T)
    NUM_ACTIVATION_TENSORS::Int32
end

function get_tensors(acts::ActivationTensors)::Vector{AbstractArray}
    res = [
        acts.encoded,
        acts.ln1, acts.ln1_mean,
        acts.ln1_rstd, acts.qkv,
        acts.atty, acts.preatt,
        acts.att, acts.attproj,
        acts.residual2, acts.ln2,
        acts.ln2_mean, acts.ln2_rstd,
        acts.fch, acts.fch_gelu,
        acts.fcproj, acts.residual3,
        acts.lnf, acts.lnf_mean,
        acts.lnf_rstd, acts.logits,
        acts.probs, acts.losses,
    ]
end

function malloc_activations(config::GPT2Config, B::Int32, T::Int32)
    C::Int64 = config.channels;
    NH::Int64 = config.num_heads;
    L::Int64 = config.num_layers;
    Vp::Int64 = config.padded_vocab_size;
    encoded = zeros(Float32, C, T, B) #// encoded
    ln1 = zeros(Float32, C, T, B, L) #// ln1
    ln1_mean = zeros(Float32, T, B, L) #// ln1_mean
    ln1_rstd = zeros(Float32, T, B, L) #// ln1_rstd
    qkv = zeros(Float32, 3 * C, T, B, L) #// qkv
    atty = zeros(Float32, C, T, B, L) #// atty
    preatt = zeros(Float32, T, T, NH, B, L) #// preatt
    att = zeros(Float32, T, T, NH, B, L) #// att
    attproj = zeros(Float32, C, T, B, L) #// attproj
    residual2 = zeros(Float32, C, T, B, L) #// residual2
    ln2 = zeros(Float32, C, T, B, L) #// ln2
    ln2_mean = zeros(Float32, T, B, L) #// ln2_mean
    ln2_rstd = zeros(Float32, T, B, L) #// ln2_rstd
    fch = zeros(Float32, 4 * C, T, B, L) #// fch
    fch_gelu = zeros(Float32, 4 * C, T, B, L) #// fch_gelu
    fcproj = zeros(Float32, C, T, B, L) #// fcproj
    residual3 = zeros(Float32, C, T, B, L) #// residual3
    lnf = zeros(Float32, C, T, B) #// lnf
    lnf_mean = zeros(Float32, T, B) #// lnf_mean
    lnf_rstd = zeros(Float32, T, B) #// lnf_rstd
    logits = zeros(Float32, Vp, T, B) #// logits
    probs = zeros(Float32, Vp, T, B) #// probs
    losses = zeros(Float32, T, B) #// losses
    NUM_ACTIVATION_TENSORS::Int32 = 23

    ActivationTensors(
        encoded,
        ln1,
        ln1_mean,
        ln1_rstd,
        qkv,
        atty,
        preatt,
        att,
        attproj,
        residual2,
        ln2,
        ln2_mean,
        ln2_rstd,
        fch,
        fch_gelu,
        fcproj,
        residual3,
        lnf,
        lnf_mean,
        lnf_rstd,
        logits,
        probs,
        losses,
        NUM_ACTIVATION_TENSORS
    )
end



struct GPT2
    config::GPT2Config
    # // the weights (parameters) of the model, and their sizes
    params::ParameterTensors
    params_memory::Vector{AbstractArray}
    # // gradients of the weights
    grads::ParameterTensors
    grads_memory::Vector{AbstractArray}
    # // buffers for the AdamW optimizer
    m_memory::Vector{AbstractArray} # ParameterTensors
    v_memory::Vector{AbstractArray} # ParameterTensors
    # // the activations of the model, and their sizes
    acts::ActivationTensors
    num_activations::Int64
    # // gradients of the activations
    grads_acts::ActivationTensors
    grads_acts_memory::Vector{AbstractArray}
    # // other run state configuration
    batch_size::Int32 #// the batch size (B) of current forward pass
    seq_len::Int32 #// the sequence length (T) of current forward pass
    inputs::Matrix{Int32} #// the input tokens for the current forward pass
    targets::Matrix{Int32} #// the target tokens for the current forward pass
    mean_loss::Vector{Float32} # one element // after a forward pass with targets, will be populated with the mean loss
end

function fread_into_param_tensor(checkpoint_path::String, params_tensor::ParameterTensors, file_header_size::Int64, config::GPT2Config)

    C = config.channels;
    Vp = config.padded_vocab_size;
    maxT = config.max_seq_len;
    L = config.num_layers;

    open(checkpoint_path, "r") do io
        # Seek to the desired byte position (e.g., byte 100)
        seek(io, file_header_size)
        
        # Julia use column major
        read!(io, params_tensor.wte)
        read!(io, params_tensor.wpe)
        read!(io, params_tensor.ln1w)
        read!(io, params_tensor.ln1b)
        read!(io, params_tensor.qkvw)
        read!(io, params_tensor.qkvb)
        read!(io, params_tensor.attprojw)
        read!(io, params_tensor.attprojb)
        read!(io, params_tensor.ln2w)
        read!(io, params_tensor.ln2b)
        read!(io, params_tensor.fcw)
        read!(io, params_tensor.fcb)
        read!(io, params_tensor.fcprojw)
        read!(io, params_tensor.fcprojb)
        read!(io, params_tensor.lnfw)
        read!(io, params_tensor.lnfb)

    end
end

function gpt2_build_from_checkpoint(checkpoint_path::String; B::Int32=1, T::Int32=0)::GPT2

    # # // read in model from a checkpoint file
    file_header_size = 256

    model_header = Vector{Int32}(undef, file_header_size)
    # Open the file in read-only binary mode
     open(checkpoint_path, "r") do io
        read!(io, model_header)
    end

    if (model_header[1] != 20240326)
        println("Bad magic model file")
        exit(1)
    end
    if (model_header[2] != 3)
        println("Bad version in model file");
        println("---> HINT: try to re-run `python train_gpt2.py`");
        exit(1);
    end

    # // read in hyperparameters
    maxT = model_header[3];
    V = model_header[4];
    L = model_header[5];
    NH = model_header[6];
    C = model_header[7];
    Vp = model_header[8];

    println("[GPT-2]");
    println("max_seq_len: $maxT");
    println("vocab_size: $V");
    println("padded_vocab_size: $Vp");
    println("num_layers: $L");
    println("num_heads: $NH");
    println("channels: $C");

    gpt2_config = GPT2Config(
        maxT, # max sequence length
        V,    # vocab size, e.g. 50257
        Vp,   # padded to e.g. %128==0, 50304
        L,    # number of layers, e.g. 12
        NH,   # number of heads in attention, e.g. 12
        C     # number of channels, e.g. 768
    )

    params_tensor = malloc_and_point_parameters(gpt2_config)
    params_memory = get_tensors(params_tensor)

    fread_into_param_tensor(checkpoint_path, params_tensor, file_header_size * sizeof(Int32), gpt2_config)


    # // other inits
    batch_size = B;
    seq_len = if T != 0 T else maxT end;
    mean_loss = [-1.0f0]; #// -1.0f will designate no loss

    grads = malloc_and_point_parameters(gpt2_config)
    grads_memory = get_tensors(grads)
    m_memory = get_tensors(malloc_and_point_parameters(gpt2_config))
    for arr in m_memory
        arr .= 0
    end
    v_memory = get_tensors(malloc_and_point_parameters(gpt2_config))
    for arr in v_memory
        arr .= 0
    end


    acts = malloc_activations(gpt2_config, B, T)
    grads_acts = malloc_activations(gpt2_config, B, T)
    grads_acts_memory = get_tensors(grads_acts)

    inputs = Matrix{Int32}(undef, T, B) #// the input tokens for the current forward pass
    targets = Matrix{Int32}(undef, T, B) #// the target tokens for the current forward pass

    GPT2(
        gpt2_config,
        params_tensor,
        params_memory,
        grads,
        grads_memory,
        m_memory,
        v_memory,
        acts,
        0, #num_activations::Int64 TODO
        # // gradients of the activations
        grads_acts,
        grads_acts_memory,
        # // other run state configuration
        batch_size, #batch_size::Int32 #// the batch size (B) of current forward pass TODO
        seq_len, #// the sequence length (T) of current forward pass
        inputs, #// the input tokens for the current forward pass
        targets, #// the target tokens for the current forward pass
        mean_loss #// after a forward pass with targets, will be populated with the mean loss
    )
end

function gpt2_forward(model::GPT2, inputs::Array{Int32, 2}, B::Int32, T::Int32; targets::Array{Int32, 2}=Array{Int32, 2}(undef, 0, 0))
    # // targets are optional and could be NULL

    # // convenience parameters (size_t to help prevent int overflow)
    V = model.config.vocab_size;
    Vp = model.config.padded_vocab_size;
    L = model.config.num_layers;
    NH = model.config.num_heads;
    C = model.config.channels;

    # // validate inputs, all indices must be in the range [0, V)
    for idx in CartesianIndices(inputs)
        @assert 0 <= inputs[idx] && inputs[idx] < V "input must be in the range [0, $V)"
        if !isempty(targets)
            @assert 0 <= targets[idx] && targets[idx] < V "targets must be in the range [0, $V)"
        end
    end

    if (B != model.batch_size || T != model.seq_len)
        println("Model: B=$(model.batch_size) T=$(model.seq_len), Desired: B=$B T=$T")
        EXIT_FAILURE = -10
        exit(EXIT_FAILURE);
    end
    # end

    # // cache the inputs/targets
    model.inputs .= inputs
    if !isempty(targets)
        model.targets .= targets
    end

    # // forward pass
    params::ParameterTensors = model.params; #// for brevity
    acts::ActivationTensors = model.acts;
    # float* residual; # TODO
    encoder_forward(acts.encoded, inputs, params.wte, params.wpe, B, T, C); #// encoding goes into residual[0]
    for l in 1:L

        residual = if l == 1
            acts.encoded
        else
            acts.residual3[:, :, :, l-1] # TODO l or l - 1
        end

        # // get the pointers of the weights for this layer
        l_ln1w = @view params.ln1w[:, l]
        l_ln1b = @view params.ln1b[:, l]
        l_qkvw = @view params.qkvw[:, :, l]
        l_qkvb = @view params.qkvb[:, l]
        l_attprojw = @view params.attprojw[:, :, l]
        l_attprojb = @view params.attprojb[:, l]
        l_ln2w = @view params.ln2w[:, l]
        l_ln2b = @view params.ln2b[:, l]
        l_fcw = @view params.fcw[:, :, l]
        l_fcb = @view params.fcb[:, l]
        l_fcprojw = @view params.fcprojw[:, :, l]
        l_fcprojb = @view params.fcprojb[:, l]

        # // get the pointers of the activations for this layer
        l_ln1 = @view acts.ln1[:, :, :, l]
        l_ln1_mean = @view acts.ln1_mean[:, :, l]
        l_ln1_rstd = @view acts.ln1_rstd[:, :, l]
        l_qkv = @view acts.qkv[:, :, :, l]
        l_atty = @view acts.atty[:, :, :, l]
        l_preatt = @view acts.preatt[:, :, :, :, l]
        l_att = @view acts.att[:, :, :, :, l]
        l_attproj = @view acts.attproj[:, :, :, l]
        l_residual2 = @view acts.residual2[:, :, :, l]
        l_ln2 = @view acts.ln2[:, :, :, l]
        l_ln2_mean = @view acts.ln2_mean[:, :, l]
        l_ln2_rstd = @view acts.ln2_rstd[:, :, l]
        l_fch = @view acts.fch[:, :, :, l]
        l_fch_gelu = @view acts.fch_gelu[:, :, :, l]
        l_fcproj = @view acts.fcproj[:, :, :, l]
        l_residual3 = @view acts.residual3[:, :, :, l]

        # // now do the forward pass
        layernorm_forward(l_ln1, l_ln1_mean, l_ln1_rstd, residual, l_ln1w, l_ln1b, B, T, C);
        matmul_forward(l_qkv, l_ln1, l_qkvw, B, T, C, Int32(3*C), bias=l_qkvb);
        attention_forward(l_atty, l_preatt, l_att, l_qkv, B, T, C, NH);
        matmul_forward(l_attproj, l_atty, l_attprojw, B, T, C, C, bias=l_attprojb);
        residual_forward(l_residual2, residual, l_attproj);
        layernorm_forward(l_ln2, l_ln2_mean, l_ln2_rstd, l_residual2, l_ln2w, l_ln2b, B, T, C);
        matmul_forward(l_fch, l_ln2, l_fcw, B, T, C, Int32(4*C), bias=l_fcb);
        gelu_forward(l_fch_gelu, l_fch, B*T*4*C);
        matmul_forward(l_fcproj, l_fch_gelu, l_fcprojw, B, T, Int32(4*C), C, bias=l_fcprojb);
        residual_forward(l_residual3, l_residual2, l_fcproj);
    end
    residual = acts.residual3[:, :, :, L] #// last residual is in residual3

    layernorm_forward(acts.lnf, acts.lnf_mean, acts.lnf_rstd, residual, params.lnfw, params.lnfb, B, T, C);
    matmul_forward(acts.logits, acts.lnf, params.wte, B, T, C, Vp);
    softmax_forward(acts.probs, acts.logits, B, T, V, Vp);

    # // also forward the cross-entropy loss function if we have the targets
    if !isempty(targets)
        crossentropy_forward(model.acts.losses, model.acts.probs, targets, B, T, Vp);
        mean_loss = mean(model.acts.losses)
        model.mean_loss[1] = mean_loss
    else
        # // if we don't have targets, we don't have a loss
        model.mean_loss[1] = -1.0f0;
    end
end

function gpt2_zero_grad(model::GPT2)
    for arr in model.grads_memory
        arr .= 0
    end

    for arr in model.grads_acts_memory
        arr .= 0
    end

end

function gpt2_backward(model::GPT2)

    # // double check we forwarded previously, with targets
    if (model.mean_loss[1] == -1.0f0)
        println("Error: must forward with targets before backward");
        exit(1);
    end

    # // convenience shortcuts (and size_t to help prevent int overflow)
    B = model.batch_size;
    T = model.seq_len;
    V = model.config.vocab_size;
    Vp = model.config.padded_vocab_size;
    L = model.config.num_layers;
    NH = model.config.num_heads;
    C = model.config.channels;

    # // backward pass: go in the reverse order of the forward pass, and call backward() functions
    params = model.params; # // for brevity
    grads = model.grads;
    acts = model.acts;
    grads_acts = model.grads_acts;

    # // we kick off the chain rule by filling in dlosses with 1.0f/(B*T)
    # // technically this is a small, inline backward() pass of calculating
    # // total, final loss as the mean over all losses over all (B,T) positions in the batch
    dloss_mean::Float32 = 1.0f0 / (B*T);
    grads_acts.losses .= dloss_mean;

    crossentropy_softmax_backward(grads_acts.logits, grads_acts.losses, acts.probs, model.targets, B, T, V, Vp);
    matmul_backward(grads_acts.lnf, grads.wte, grads_acts.logits, acts.lnf, params.wte, B, T, C, Vp);
    residual = @view acts.residual3[:, :, :, L];
    dresidual = @view grads_acts.residual3[:, :, :, L]
    layernorm_backward(dresidual, grads.lnfw, grads.lnfb, grads_acts.lnf, residual, params.lnfw, acts.lnf_mean, acts.lnf_rstd, B, T, C);

    for l in L:-1:1
        residual = if l == 1
            acts.encoded
        else
            @view acts.residual3[:, :, :, l-1]
        end
        dresidual = if l == 1
            grads_acts.encoded
        else
            @view grads_acts.residual3[:, :, :, l-1]
        end

        # Get the views of the weights for this layer
        l_ln1w = @view params.ln1w[:, l]
        l_qkvw = @view params.qkvw[:, :, l]
        l_attprojw = @view params.attprojw[:, :, l]
        l_ln2w = @view params.ln2w[:, l]
        l_fcw = @view params.fcw[:, :, l]
        l_fcprojw = @view params.fcprojw[:, :, l]

        # Get the views of the gradients of the weights for this layer
        dl_ln1w = @view grads.ln1w[:, l]
        dl_ln1b = @view grads.ln1b[:, l]
        dl_qkvw = @view grads.qkvw[:, :, l]
        dl_qkvb = @view grads.qkvb[:, l]
        dl_attprojw = @view grads.attprojw[:, :, l]
        dl_attprojb = @view grads.attprojb[:, l]
        dl_ln2w = @view grads.ln2w[:, l]
        dl_ln2b = @view grads.ln2b[:, l]
        dl_fcw = @view grads.fcw[:, :, l]
        dl_fcb = @view grads.fcb[:, l]
        dl_fcprojw = @view grads.fcprojw[:, :, l]
        dl_fcprojb = @view grads.fcprojb[:, l]

        # Get the views of the activations for this layer
        l_ln1 = @view acts.ln1[:, :, :, l]
        l_ln1_mean = @view acts.ln1_mean[:, :, l]
        l_ln1_rstd = @view acts.ln1_rstd[:, :, l]
        l_qkv = @view acts.qkv[:, :, :, l]
        l_atty = @view acts.atty[:, :, :, l]
        l_att = @view acts.att[:, :, :, :, l]
        l_residual2 = @view acts.residual2[:, :, :, l]
        l_ln2 = @view acts.ln2[:, :, :, l]
        l_ln2_mean = @view acts.ln2_mean[:, :, l]
        l_ln2_rstd = @view acts.ln2_rstd[:, :, l]
        l_fch = @view acts.fch[:, :, :, l]
        l_fch_gelu = @view acts.fch_gelu[:, :, :, l]

        # Get the views of the gradients of the activations for this layer
        dl_ln1 = @view grads_acts.ln1[:, :, :, l]
        dl_qkv = @view grads_acts.qkv[:, :, :, l]
        dl_atty = @view grads_acts.atty[:, :, :, l]
        dl_preatt = @view grads_acts.preatt[:, :, :, :, l]
        dl_att = @view grads_acts.att[:, :, :, :, l]
        dl_attproj = @view grads_acts.attproj[:, :, :, l]
        dl_residual2 = @view grads_acts.residual2[:, :, :, l]
        dl_ln2 = @view grads_acts.ln2[:, :, :, l]
        dl_fch = @view grads_acts.fch[:, :, :, l]
        dl_fch_gelu = @view grads_acts.fch_gelu[:, :, :, l]
        dl_fcproj = @view grads_acts.fcproj[:, :, :, l]
        dl_residual3 = @view grads_acts.residual3[:, :, :, l]


        # // backprop this layer
        residual_backward(dl_residual2, dl_fcproj, dl_residual3);
        matmul_backward(dl_fch_gelu, dl_fcprojw, dl_fcproj, l_fch_gelu, l_fcprojw, B, T, Int32(4*C), C, dbias=dl_fcprojb);
        gelu_backward(dl_fch, l_fch, dl_fch_gelu, B*T*4*C);
        matmul_backward(dl_ln2, dl_fcw, dl_fch, l_ln2, l_fcw, B, T, C, Int32(4*C), dbias=dl_fcb);
        layernorm_backward(dl_residual2, dl_ln2w, dl_ln2b, dl_ln2, l_residual2, l_ln2w, l_ln2_mean, l_ln2_rstd, B, T, C);
        residual_backward(dresidual, dl_attproj, dl_residual2);
        matmul_backward(dl_atty, dl_attprojw, dl_attproj, l_atty, l_attprojw, B, T, C, C, dbias=dl_attprojb);
        attention_backward(dl_qkv, dl_preatt, dl_att, dl_atty, l_qkv, l_att, B, T, C, NH);
        matmul_backward(dl_ln1, dl_qkvw, dl_qkv, l_ln1, l_qkvw, B, T, C, Int32(3*C), dbias=dl_qkvb);
        layernorm_backward(dresidual, dl_ln1w, dl_ln1b, dl_ln1, residual, l_ln1w, l_ln1_mean, l_ln1_rstd, B, T, C);
    end
    encoder_backward(grads.wte, grads.wpe, grads_acts.encoded, model.inputs, B, T, C);
end

function gpt2_update(model::GPT2, learning_rate::Float32, beta1::Float32, beta2::Float32, eps::Float32, weight_decay::Float32, t::Int64)
    # // reference: https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html

    arr_num = length(model.params_memory)

    for i in 1:arr_num
        param = model.params_memory[i]
        grad = model.grads_memory[i]

        # // update the first moment (momentum)
        m = beta1 * model.m_memory[i] + (1.0f0 - beta1) * grad;
        # // update the second moment (RMSprop)
        v = beta2 * model.v_memory[i] + (1.0f0 - beta2) .* grad .* grad;
        # // bias-correct both moments
        m_hat = m ./ (1.0f0 - beta1.^t);
        v_hat = v ./ (1.0f0 - beta2.^t);

        # // update
        model.m_memory[i] .= m;
        model.v_memory[i] .= v;
        model.params_memory[i] .-= learning_rate * (m_hat ./ (sqrt.(v_hat) .+ eps) + weight_decay * param);
    end
end
