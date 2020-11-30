
module PoleeClassifier

using Polee
using Polee.PoleeModel
using Flux
using HDF5
import CUDA
import ProgressMeter

const device = Flux.gpu
# const device = Flux.cpu

const batchsize = 100
# const nepochs = 1000
# const nepochs = 25
const nepochs = 250
# const nepochs = 500

"""
Generic transformation applid to expression vectors. This seems to make
training easier.
"""
# expr_trans(x) = log.(x) .- log(1f0/size(x, 1))

function clr(x)
    x_log = log.(x)
    return x_log .- CUDA.mean(x_log)
end
const expr_trans = clr

sqnorm(ws) = sum(abs2, ws)

# Different ways of approaching quantification each with a different take
# on training the clasifier
abstract type QuantMethod end

"""
Simple nn classifier that tries to learn some feature (e.g. tissue) from
expression, where how quantification is handled depends on QuantMethod.
"""
mutable struct Classifier{T <: QuantMethod}
    quant::T
    layers::Union{Chain, Nothing}
    factor::String
    classes::Vector{String}
end


"""
Plain old point estimates.
"""
struct PointEstimate <: QuantMethod
    ts::Polee.Transcripts
    ts_metadata::Polee.TranscriptsMetadata
    point_estimate::String
end


"""
Collect an array of unique classes from the specification for the given factor.
"""
function get_classes(spec::Dict, factor::String)
    classes = Set{String}()
    for sample in spec["samples"]
        push!(classes, sample["factors"][factor])
    end
    return collect(classes)
end


"""
Construct what will be our standard classifier nn given input and output size.
"""
function build_model(n_in::Int, n_out::Int)
    initW = (dims...) -> 1e-3 * randn(Float32, dims...)

    # M = 50
    # return Chain(
    #     Dense(n_in, M, leakyrelu, initW=initW),
    #     Dropout(0.25, M),
    #     Dense(M, M, leakyrelu, initW=initW),
    #     Dropout(0.25, M),
    #     Dense(M, M, leakyrelu, initW=initW),
    #     Dense(M, n_out, initW=initW))

    # logistic regression
    return Chain(Dense(n_in, n_out, initW=initW))
end


function fit!(model::Classifier{PointEstimate}, train_spec::Dict)

    train_data = load_point_estimates_from_specification(
        train_spec,
        model.quant.ts,
        model.quant.ts_metadata,
        model.quant.point_estimate)

    model.classes = get_classes(train_spec, model.factor)
    model.layers = device(build_model(length(model.quant.ts), length(model.classes)))

    num_samples, n = size(train_data.x0_values)
    train_expr = expr_trans(Array(transpose(train_data.x0_values)))
    train_classes = hcat(
        [Flux.onehot(sample["factors"][model.factor], model.classes)
         for sample in train_spec["samples"]]...)

    train_data_loader = device.(Flux.Data.DataLoader(
        train_expr, train_classes,
        batchsize=batchsize, shuffle=true))

    function total_loss()
        l = 0f0
        for (x, y) in train_data_loader
            l += Flux.Losses.logitcrossentropy(model.layers(x), y)
        end
        return l / length(train_data_loader)
    end

    loss = (x, y) -> Flux.Losses.logitcrossentropy(model.layers(x), y)

    opt = ADAM()
    prog = ProgressMeter.Progress(nepochs, desc="training: ")
    for epoch in 1:nepochs
        ps = params(model.layers)
        for (x, y) in train_data_loader
            gs = gradient(ps) do
                return loss(x, y)
            end
            Flux.update!(opt, ps, gs)
        end
        ProgressMeter.next!(prog, showvalues = [(:loss, total_loss())])
    end
end


function eval(model::Classifier{PointEstimate}, eval_spec::Dict)
    Flux.testmode!(model.layers)

    eval_data = load_point_estimates_from_specification(
        eval_spec,
        model.quant.ts,
        model.quant.ts_metadata,
        model.quant.point_estimate)

    num_samples, n = size(eval_data.x0_values)
    eval_expr = expr_trans(Array(transpose(eval_data.x0_values)))
    eval_classes = hcat(
        [Flux.onehot(sample["factors"][model.factor], model.classes)
         for sample in eval_spec["samples"]]...)

    eval_data_loader = Flux.Data.DataLoader(
        device(eval_expr), eval_classes,
        batchsize=batchsize)

    acc = 0.0
    for (x, y) in eval_data_loader
        pred = cpu(model.layers(x))
        acc += sum(Flux.onecold(pred) .== Flux.onecold(y))
    end
    acc /= num_samples
    return acc
end


struct KallistoBootstrap <: QuantMethod
    ts::Polee.Transcripts
    ts_metadata::Polee.TranscriptsMetadata
    pseudocount::Float32
    max_boot_samples::Union{Nothing, Int}
end


function load_kallisto_bootstap(model::Classifier{KallistoBootstrap}, spec::Dict)

    transcript_idx = Dict{String, Int}()
    for (j, t) in enumerate(model.quant.ts)
        transcript_idx[t.metadata.name] = j
    end
    n = length(transcript_idx)

    expr_data_vecs = Array{Float32, 2}[]
    class_data_vecs = Flux.OneHotVector[]

    for sample in spec["samples"]
        filename = sample["kallisto"]
        input = h5open(filename)

        transcript_ids = read(input["aux"]["ids"])
        efflens = read(input["aux"]["eff_lengths"])

        label = sample["factors"][model.factor]
        label_onehot = Flux.onehot(label, model.classes)

        boot_count = 0
        for dataset in input["bootstrap"]
            if model.quant.max_boot_samples !== nothing && boot_count >= model.quant.max_boot_samples
                break
            end

            bootstrap_counts = Vector{Float32}(read(dataset))
            bs = Polee.PoleeModel.kallisto_counts_to_proportions(
                bootstrap_counts, efflens, model.quant.pseudocount,
                transcript_ids, transcript_idx)
            push!(expr_data_vecs, bs)
            push!(class_data_vecs, label_onehot)
            boot_count += 1
        end

        close(input)
    end

    expr_data = expr_trans(Array(transpose(vcat(expr_data_vecs...))))
    class_data_vecs = hcat(class_data_vecs...)

    return expr_data, class_data_vecs
end


function fit!(model::Classifier{KallistoBootstrap}, train_spec::Dict)
    model.classes = get_classes(train_spec, model.factor)
    model.layers = device(build_model(length(model.quant.ts), length(model.classes)))

    train_expr, train_classes = load_kallisto_bootstap(model, train_spec)

    train_data_loader = device.(Flux.Data.DataLoader(
        train_expr, train_classes,
        batchsize=batchsize, shuffle=true))

    function total_loss()
        l = 0f0
        for (x, y) in train_data_loader
            l += Flux.Losses.logitcrossentropy(model.layers(x), y)
        end
        return l / length(train_data_loader)
    end

    opt = ADAM()
    Flux.@epochs nepochs Flux.Optimise.train!(
            (x, y) -> Flux.Losses.logitcrossentropy(model.layers(x), y),
            params(model.layers),
            train_data_loader,
            opt,
            cb=() -> @show total_loss())
end


function eval(model::Classifier{KallistoBootstrap}, eval_spec::Dict)
    Flux.testmode!(model.layers)

    transcript_idx = Dict{String, Int}()
    for (j, t) in enumerate(model.quant.ts)
        transcript_idx[t.metadata.name] = j
    end
    n = length(transcript_idx)

    acc = 0.0
    for sample in eval_spec["samples"]
        filename = sample["kallisto"]
        input = h5open(filename)

        transcript_ids = read(input["aux"]["ids"])
        efflens = read(input["aux"]["eff_lengths"])

        label = sample["factors"][model.factor]
        label_onehot = Flux.onehot(label, model.classes)

        pred = zeros(Float32, length(model.classes))
        boot_count = 0

        # TODO: Actually we should be averaging after softmax, not prior
        for dataset in input["bootstrap"]
            if model.quant.max_boot_samples !== nothing && boot_count >= model.quant.max_boot_samples
                break
            end

            bootstrap_counts = Vector{Float32}(read(dataset))
            bs = Polee.PoleeModel.kallisto_counts_to_proportions(
                bootstrap_counts, efflens, model.quant.pseudocount,
                transcript_ids, transcript_idx)

            pred .+= cpu(model.layers(device(expr_trans(bs[1,:]))))
            boot_count += 1
        end
        pred ./= boot_count
        acc += Flux.onecold(pred) .== Flux.onecold(label_onehot)
        close(input)
    end
    acc /= length(eval_spec["samples"])

    return acc
end


struct PTTBetaParams <: QuantMethod
end


function load_beta_params(model::Classifier{PTTBetaParams}, spec::Dict)
    xs_vecs = Vector{Float32}[]
    for sample in spec["samples"]
        input = h5open(sample["file"])
        x = vcat(read(input["alpha"]), read(input["beta"]))
        push!(xs_vecs, x)
        close(input)
    end

    train_approx = log.(hcat(xs_vecs...))
    train_classes = hcat(
        [Flux.onehot(sample["factors"][model.factor], model.classes)
         for sample in spec["samples"]]...)

    return train_approx, train_classes
end


function fit!(model::Classifier{PTTBetaParams}, train_spec::Dict)
    # TODO: one issue with this approach is that we have no way of using
    # effective length.

    model.classes = get_classes(train_spec, model.factor)
    train_approx, train_classes = load_beta_params(model, train_spec)

    train_data_loader = device.(Flux.Data.DataLoader(
        train_approx, train_classes,
        batchsize=batchsize, shuffle=true))

    n_in = size(train_approx, 1)
    n_out = length(model.classes)
    model.layers = device(build_model(n_in, n_out))

    function total_loss()
        l = 0f0
        for (x, y) in train_data_loader
            l += Flux.Losses.logitcrossentropy(model.layers(x), y)
        end
        return l / length(train_data_loader)
    end

    opt = ADAM()
    Flux.@epochs nepochs Flux.Optimise.train!(
            (x, y) -> Flux.Losses.logitcrossentropy(model.layers(x), y),
            params(model.layers),
            train_data_loader,
            opt,
            cb=() -> @show total_loss())
end


function eval(model::Classifier{PTTBetaParams}, eval_spec::Dict)
    Flux.testmode!(model.layers)

    eval_approx, eval_classes = load_beta_params(model, eval_spec)

    n_in, num_samples = size(eval_approx)

    acc = 0.0
    for i in 1:num_samples
        pred = cpu(model.layers(device(eval_approx[:,i])))
        acc += eval_spec["samples"][i]["factors"][model.factor] == model.classes[Flux.onecold(pred)]
    end
    acc /= num_samples
    return acc
end


struct PTTLatentExpr <: QuantMethod
    t::Polee.PolyaTreeTransform
    neval_samples::Int
end


function PTTLatentExpr(ptt_filename::String, neval_samples::Int)
    input = h5open(ptt_filename)
    t = Polee.PolyaTreeTransform(
        read(input["node_parent_idxs"]),
        read(input["node_js"]))
    close(input)
    return PTTLatentExpr(t, neval_samples)
end


function load_pttlatent_data(model::Classifier{PTTLatentExpr}, spec::Dict)
    μ_vecs = Vector{Float32}[]
    σ_vecs = Vector{Float32}[]
    α_vecs = Vector{Float32}[]

    for sample in spec["samples"]
        input = h5open(sample["file"])
        push!(μ_vecs, read(input["mu"]))
        push!(σ_vecs, exp.(read(input["omega"])))
        push!(α_vecs, read(input["alpha"]))
        close(input)
    end

    train_classes = hcat(
        [Flux.onehot(sample["factors"][model.factor], model.classes)
         for sample in spec["samples"]]...)

    μs = hcat(μ_vecs...)
    σs = hcat(σ_vecs...)
    αs = hcat(α_vecs...)

    return μs, σs, αs, train_classes
end


function cuda_approx_sample(t::Polee.PolyaTreeTransform, μ, σ, α)
    n = size(μ, 1) + 1
    m = size(μ, 2)
    z0 = CUDA.randn(size(μ)...)
    z = CUDA.sinh.(CUDA.asinh.(z0) .+ α) # sinh-asinh transform
    y = CUDA.inv.(CUDA.exp.(.-(μ .+ z .* σ)) .+ 1f0) # logit normal transform

    # polya tree transformation

    # TODO: does seem like the part below is the really slow part
    # Not really sure how to make it faster. In the past we tried a series
    # of sparse matrix multiplies. Would that be better? Is there a way to
    # avoid allocating so many intermediate arrays?

    # u = CUDA.zeros(Float64, 2*n-1, m) # intermediate values

    # this way we can avoid copying rows
    us = [CUDA.zeros(Float64, m) for _ in 1:2*n-1]
    # us = Array{CUDA.CuVector{Float64}}(undef, 2*n-1)
    us[1] .= CUDA.ones(Float64, m)

    x = CUDA.zeros(n, m) # output values
    k = 1 # internal node count
    for i in 1:2n-1
        @show i
        u_i = us[i]
        output_idx = t.index[1, i]
        if output_idx != 0
            # TODO: would it just be faster to do a big hcat?
            x[output_idx,:] .= u_i
            continue
        end

        left_idx = t.index[2, i]
        right_idx = t.index[3, i]

        v = y[k,:] .* u_i

        # TODO: Can we drop these max statements?
        us[left_idx] .= max.(v, 1e-16)
        us[right_idx] .= max.(u_i .- v, 1e-16)

        k += 1
    end

    return x
end


function approx_sample(t::Polee.PolyaTreeTransform, als::Vector{Polee.ApproxLikelihoodSampler}, μ, σ, α)
    n = size(μ, 1) + 1
    x = Array{Float32}(undef, (n, size(μ, 2)))
    # TODO: oof this is totally fucked because t holds intermediate state that gets overwritten
    Threads.@threads for i in 1:size(μ, 2)
        Polee.set_transform!(als[i], t, μ[:,i], σ[:,i], α[:,i])
        Polee.rand!(als[i], @view x[:,i])
    end
    return x
end


# TODO: Let's just do a vectorized and mulithreaded cpu implementation. From
# there we could probably write a custom cuda version.

"""
Approx likelihood sampler operating on multiple samples at once.
"""
struct VecApproxLikelihoodSampler
    ys::Array{Float64, 2}
    us::Array{Float64, 2}
    t::Polee.PolyaTreeTransform
end


function VecApproxLikelihoodSampler(t::Polee.PolyaTreeTransform, n::Int, m::Int)
    return VecApproxLikelihoodSampler(
        Array{Float64}(undef, n-1, m),
        Array{Float64}(undef, 2*n-1, m),
        t)
end


function logistic(x::T) where {T}
    return inv(one(T) + exp(-x))
end


"""
Generate multiple samples from likelihood approximations.
"""
function Polee.rand!(
    sampler::VecApproxLikelihoodSampler, μ, σ, α, x)

    ys = sampler.ys
    us = sampler.us
    t = sampler.t

    n, m = size(x)

    # randn -> sinh-asinh -> logit-normal
    Threads.@threads for i in 1:(n-1)*m
        ys[i] = logistic(sinh(asinh(randn(Float32)) + α[i]) * σ[i] + μ[i])
    end

    # polya tree
    us[1,:] .= 1.0f0
    k = 1
    for i in 1:2n-1
        output_idx = t.index[1, i]
        if output_idx != 0
            for j in 1:m
                x[output_idx, j] = Float32(us[i, j])
            end

            # have to cast, so can't do this
            # unsafe_copyto!(
            #     pointer(@view x[output_idx, 1]),
            #     pointer(@view us[i, 1]), m)
            continue
        end

        left_idx = t.index[2, i]
        right_idx = t.index[3, i]

        # TODO: could use simd and/or threads for this part.
        for j in 1:m
            v = ys[k,j] * us[i,j]
            us[left_idx, j] = max(v, 1e-16)
            us[right_idx, j] = max(us[i,j] - v, 1e-16)
        end

        k += 1
    end
end



function fit!(model::Classifier{PTTLatentExpr}, train_spec::Dict)
    # Just going to assume logit-skew-normal approx here. May need
    # to generalize in the future if we start using beta approx.

    model.classes = get_classes(train_spec, model.factor)
    μs, σs, αs, train_classes = load_pttlatent_data(model, train_spec)

    train_data_loader = Flux.Data.DataLoader(
        μs, σs, αs,
        device(train_classes),
        batchsize=batchsize, shuffle=true)

    n = size(μs, 1) + 1
    n_out = length(model.classes)
    model.layers = device(build_model(n, n_out))

    sampler = VecApproxLikelihoodSampler(model.quant.t, n, batchsize)

    function total_loss()
        l = 0f0
        for (μ, σ, α, y) in train_data_loader
            x = Array{Float32}(undef, n, size(μ, 2))
            Polee.rand!(sampler, μ, σ, α, x)
            x_gpu = expr_trans(device(x))
            # x = expr_trans(device(approx_sample(model.quant.t, als, μ, σ, α)))
            l += Flux.Losses.logitcrossentropy(
                model.layers(x_gpu), y)
        end
        return l / length(train_data_loader)
    end

    loss = (x, y) -> Flux.Losses.logitcrossentropy(model.layers(x), y)

    # custom training loop that handles drawing samples
    opt = ADAM()
    prog = ProgressMeter.Progress(nepochs, desc="training: ")
    for epoch in 1:nepochs
        ps = params(model.layers)
        for (μ, σ, α, y) in train_data_loader
            # x = expr_trans(device(approx_sample(model.quant.t, als, μ, σ, α)))

            x = Array{Float32}(undef, n, size(μ, 2))
            Polee.rand!(sampler, μ, σ, α, x)
            x_gpu = expr_trans(device(x))

            gs = gradient(ps) do
                return loss(x_gpu, y)
            end
            Flux.update!(opt, ps, gs)
        end
        ProgressMeter.next!(prog, showvalues = [(:loss, total_loss())])
    end
end


function eval(model::Classifier{PTTLatentExpr}, eval_spec::Dict)
    Flux.testmode!(model.layers)

    μs, σs, αs, eval_classes = load_pttlatent_data(model, eval_spec)
    eval_data_loader = Flux.Data.DataLoader(
        μs, σs, αs, eval_classes, batchsize=batchsize)

    n = size(μs, 1) + 1
    m = size(μs, 2)
    sampler = VecApproxLikelihoodSampler(model.quant.t, n, batchsize)

    acc = 0.0
    total_count = 0
    for (μ, σ, α, y) in eval_data_loader
        pred = zeros(Float32, length(model.classes), size(μ, 2))
        for i in 1:model.quant.neval_samples

            x = Array{Float32}(undef, n, size(μ, 2))
            Polee.rand!(sampler, μ, σ, α, x)
            x_gpu = expr_trans(device(x))
            pred .+= Flux.softmax(cpu(model.layers(x_gpu)))
        end
        pred ./= model.quant.neval_samples

        acc += sum(Flux.onecold(pred) .== Flux.onecold(y))
    end

    acc /= m
    return acc
end


end # module PoleeClassifier