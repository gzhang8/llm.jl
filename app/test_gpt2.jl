include("train_gpt2.jl")

function check_tensor(a::AbstractArray{T}, b::AbstractArray{T}, label::String; tol=2e-2)::Bool where T <: Real
    print_upto = 5;
    ok = true;
    println(label);
    for i in 1:print_upto
        println("$(a[i]), $(b[i])");
    end

    abs_diff_array = abs.(a - b)
    maxdiff = maximum(abs_diff_array)
    ok = maxdiff <= tol

    # // print the final result for this tensor
    if (ok)
        println("TENSOR OK, maxdiff = $maxdiff");
    else
        println("TENSOR NOT OK, maxdiff = $maxdiff");
    end
    return ok;
end




# // load additional information that we will use for debugging and error checking

file_header_size = 256
debug_checkpoint_path = "ckpt/gpt2_124M_debug_state.bin"

state_header = Vector{Int32}(undef, file_header_size)
# Open the file in read-only binary mode
 open(debug_checkpoint_path, "r") do io
    read!(io, state_header)
end

if (state_header[1] != 20240327)
    println("Bad magic model file")
    exit(1)
end
if (state_header[2] != 2)
    println("Bad version in model file");
    println("---> HINT: try to re-run `python train_gpt2.py`");
    exit(1);
end

B = state_header[3]; # // batch size, e.g. 4
T = state_header[4]; # // time / sequence length (e.g. 64, up to maxT)
println("[State]");
println("batch_size: $B");
println("seq_len: $T");


# // build the GPT-2 model from a checkpoint
model::GPT2 = gpt2_build_from_checkpoint("ckpt/gpt2_124M.bin", B=B, T=T);

C = model.config.channels;
V = model.config.vocab_size;
NH = model.config.num_heads
Vp = model.config.padded_vocab_size;
maxT = model.config.max_seq_len;
L = model.config.num_layers;


# ParameterTensors expected_grads;
expected_grads::ParameterTensors = malloc_and_point_parameters(model.config)


# // inputs and expected outputs, only used for error checking
x = Matrix{Int32}(undef, T, B)
y = Matrix{Int32}(undef, T, B)

# note: this tensor is not fully saved: a BxTxVp size tensor dumped into BxTxV in c order
expected_logits_tp = Array{Float32, 3}(undef, V, T, B)
expected_loss = Vector{Float32}(undef, 1)
expected_loss[1] = 0.0f0
println(expected_loss[1])


open(debug_checkpoint_path, "r") do io
    # Seek to the desired byte position (e.g., byte 100)
    seek(io, file_header_size * sizeof(Int32))
    
    read!(io, x)
    println("x: ")
    println(x)

    read!(io, y)
    read!(io, expected_logits_tp)
    read!(io, expected_loss)


end

fread_into_param_tensor(debug_checkpoint_path, expected_grads, 
    file_header_size * sizeof(Int32) + sizeof(x) + sizeof(y) + sizeof(expected_logits_tp) + sizeof(expected_loss),
    model.config)


# // overall OK signal for the test
allok = true;

# // let's do 10 training iterations, following the pytorch code
expected_losses::Vector{Float32} = [
    5.270007133483887f0,
    4.059706687927246f0,
    3.3751230239868164f0,
    2.8007826805114746f0,
    2.315382242202759f0,
    1.8490285873413086f0,
    1.3946564197540283f0,
    0.9991465210914612f0,
    0.6240804195404053f0,
    0.37651097774505615f0
];

for step in 1:10
    println("start loop $step")

    # struct timespec start, end;
    # clock_gettime(CLOCK_MONOTONIC, &start);
    start_time = time()
    println("vect params the same arr? 1 at $step: ",  model.params_memory[1] === model.params.wte)
    gpt2_forward(model, x, B, T, targets=y);
    println("vect params the same arr? 2 at $step: ",  model.params_memory[1] === model.params.wte)

    gpt2_zero_grad(model);
    println("vect params the same arr? 3 at $step: ",  model.params_memory[1] === model.params.wte)

    gpt2_backward(model);
    println("vect params the same arr? 4 at $step: ",  model.params_memory[1] === model.params.wte)

    end_time = time()
    # clock_gettime(CLOCK_MONOTONIC, &end);
    time_elapsed_s = end_time - start_time

    if step == 1
        # // error checking at step 0 for reference activations/gradients
        # // at this point, target should be equal to expected_logits, let's compare
        logits_ok = true
        calculated_logits = model.acts.logits;
        max_diff::Float32 = 0.0f0;
        for bt in 1:B*T
            for v in 1:V
                i = (bt - 1) * Vp + v; # // linearized index, using Vp
                if (i < 10) 
                    println("$(expected_logits_tp[i]), $(calculated_logits[i])");
                end
                diff = abs(expected_logits_tp[i] - calculated_logits[i]);
                max_diff = max(max_diff, diff);
                if (diff >= 1e-2)
                    print("MISMATCH AT INDEX $bt, $v: ");
                    println("$(expected_logits_tp[i]) $(calculated_logits[i])");
                    logits_ok = false;
                    break;
                end
            end
            if logits_ok
                break
            end
        end
        if(!logits_ok)
            print("NOT ");
        end
        println("OK (LOGITS), max_diff = $max_diff");
        global allok = allok && logits_ok;

        # // compare the achieved loss
        global expected_loss
        if abs(model.mean_loss[1] - expected_loss[1]) >= 1e-2
            println("LOSS MISMATCH: $(model.mean_loss[1]) $(expected_loss[1])");
            allok = false;
        else 
            println("LOSS OK: $(model.mean_loss[1]) $(expected_loss[1])");
        end

        # // finally check all the gradients
        gradoks::Vector{Bool} = fill(false, 16)
        grads::ParameterTensors = model.grads;
        gradoks[16] = check_tensor(grads.wte, expected_grads.wte, "dwte");
        gradoks[1] = check_tensor(grads.wpe, expected_grads.wpe, "dwpe");
        gradoks[2] = check_tensor(grads.ln1w, expected_grads.ln1w, "dln1w");
        gradoks[3] = check_tensor(grads.ln1b, expected_grads.ln1b, "dln1b");
        gradoks[4] = check_tensor(grads.qkvw, expected_grads.qkvw, "dqkvw");
        gradoks[5] = check_tensor(grads.qkvb, expected_grads.qkvb, "dqkvb");
        gradoks[6] = check_tensor(grads.attprojw, expected_grads.attprojw, "dattprojw");
        gradoks[7] = check_tensor(grads.attprojb, expected_grads.attprojb, "dattprojb");
        gradoks[8] = check_tensor(grads.ln2w, expected_grads.ln2w, "dln2w");
        gradoks[9] = check_tensor(grads.ln2b, expected_grads.ln2b, "dln2b");
        gradoks[10] = check_tensor(grads.fcw, expected_grads.fcw, "dfcw");
        gradoks[11] = check_tensor(grads.fcb, expected_grads.fcb, "dfcb");
        gradoks[12] = check_tensor(grads.fcprojw, expected_grads.fcprojw, "dfcprojw");
        gradoks[13] = check_tensor(grads.fcprojb, expected_grads.fcprojb, "dfcprojb");
        gradoks[14] = check_tensor(grads.lnfw, expected_grads.lnfw, "dlnfw");
        gradoks[15] = check_tensor(grads.lnfb, expected_grads.lnfb, "dlnfb");
        for i in 1:16
            allok = allok && gradoks[i]
        end
    end
    println("vect params the same arr? 5 at $step: ",  model.params_memory[1] === model.params.wte)
    gpt2_update(model, Float32(1e-4), 0.9f0, 0.999f0, Float32(1e-8), 0.01f0, step);
    println("vect params the same arr? 6 at $step: ",  model.params_memory[1] === model.params.wte)

    # // compare the losses
    expected_loss_float = expected_losses[step];
    actual_loss = model.mean_loss[1];
    step_loss_ok = abs(expected_loss_float - actual_loss) < 1e-2;
    allok = allok && step_loss_ok;

    # // print the timing information at the end
    println("step $step: loss $(model.mean_loss) (took $(time_elapsed_s * 1000) ms) OK = $step_loss_ok");
end

# // final judgement
println("overall okay: ", allok);
