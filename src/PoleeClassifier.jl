
module PoleeClassifier

using Polee
using Polee.PoleeModel
using Flux
using DelimitedFiles
using LinearAlgebra
using SIMD
import Distributions: Beta, rand
using HDF5
import CUDA
import ProgressMeter
import Zygote

include("distance.jl")

const device = Flux.gpu
# const device = Flux.cpu

const batchsize = 100
# const nepochs = 1000
# const nepochs = 25
# const nepochs = 250
const nepochs = 500

"""
Generic transformation applid to expression vectors. This seems to make
training easier.
"""
# expr_trans(x) = log.(x) .- log(1f0/size(x, 1))

function clr(x)
    x_log = log.(x)
    return x_log .- CUDA.mean(x_log, dims=1)
end

# const expr_trans = identity
const expr_trans(x) = log.(1f0 .+ 1f6.*x)
# const expr_trans = clr
# const expr_trans(x) = log.(x)


l1(ws) = sum(abs, ws)
l2(ws) = sum(abs2, ws)

function make_loss(model)
    # return (x, y) ->
    #     sum(l2, Flux.params(model.layers)) +
    #     Flux.Losses.logitcrossentropy(model.layers(x), y)

    # return (x, y) ->
    #     Flux.Losses.logitcrossentropy(model.layers(x), y)

    return (x, y) ->
        Flux.Losses.logitcrossentropy(model.layers(x), y) +
        10f0 * sum(l2, model.layers(x)[1])
end


function make_optimizer()
    return Flux.Optimiser(ExpDecay(1.0, 0.995, 1), ADAM())
end


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
    classes::Vector
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
    classes = Set()
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

    M = 50
    # return Chain(
    #     # Dropout(0.5, n_in),
    #     Dense(n_in, M, leakyrelu, initW=initW),
    #     # Dropout(0.25, M),
    #     Dense(M, M, leakyrelu, initW=initW),
    #     # Dropout(0.25, M),
    #     Dense(M, M, leakyrelu, initW=initW),
    #     Dense(M, n_out, initW=initW))

    # TODO: I want the classifier to fit small differences, which I don't think
    # it's really doing, leading to point estimates being just as good as
    # 

    return Chain(
        # Dropout(0.5, n_in),
        Dense(n_in, M, Flux.elu),
        # Dropout(0.25, M),
        Dense(M, M, Flux.elu),
        # Dropout(0.25, M),
        Dense(M, M, Flux.elu),
        Dense(M, n_out))

    # logistic regression
    # return Chain(Dense(n_in, n_out, initW=initW))
end


function fit!(
        model::Classifier{PointEstimate}, train_spec::Dict,
        modelfn::Function=build_model, nepochs::Int=nepochs)

    # TODO: can I do this without needing transcripts?
    train_data = load_point_estimates_from_specification(
        train_spec,
        model.quant.ts,
        model.quant.ts_metadata,
        model.quant.point_estimate)

    fit!(model, train_spec, train_data, modelfn)
end

function fit!(
        model::Classifier{PointEstimate}, train_spec::Dict,
        train_data::Polee.LoadedSamples, modelfn::Function=build_model,
        nepochs::Int=nepochs)

    model.classes = get_classes(train_spec, model.factor)
    model.layers = device(modelfn(length(model.quant.ts), length(model.classes)))

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

    loss = make_loss(model)

    opt = make_optimizer()
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

    eval_data = load_point_estimates_from_specification(
        eval_spec,
        model.quant.ts,
        model.quant.ts_metadata,
        model.quant.point_estimate)

    return eval(model, eval_spec, eval_data)
end


function eval(model::Classifier{PointEstimate}, eval_spec::Dict, eval_data::Polee.LoadedSamples)
    Flux.testmode!(model.layers)

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


function fit_and_monitor!(
        model::Classifier{PointEstimate},
        train_spec::Dict, eval_spec::Dict,
        train_data, eval_data,
        modelfn::Function, nepochs::Int,
        output::IO, report_gap::Int=25)

    model.classes = get_classes(train_spec, model.factor)
    model.layers = device(modelfn(length(model.quant.ts), length(model.classes)))

    # train data

    num_train_samples, n = size(train_data.x0_values)
    train_expr = expr_trans(Array(transpose(train_data.x0_values)))
    train_classes = hcat(
        [Flux.onehot(sample["factors"][model.factor], model.classes)
         for sample in train_spec["samples"]]...)

    train_data_loader = device.(Flux.Data.DataLoader(
        train_expr, train_classes,
        batchsize=batchsize, shuffle=true))

    num_train_samples = size(train_data.x0_values, 1)

    # eval data

    eval_expr = expr_trans(Array(transpose(eval_data.x0_values)))
    eval_classes = hcat(
        [Flux.onehot(sample["factors"][model.factor], model.classes)
         for sample in eval_spec["samples"]]...)

    eval_data_loader = Flux.Data.DataLoader(
        device(eval_expr), eval_classes,
        batchsize=batchsize)

    num_eval_samples = size(eval_data.x0_values, 1)

    function eval_accuracy()
        Flux.testmode!(model.layers)
        acc = 0.0
        for (x, y) in eval_data_loader
            pred = cpu(model.layers(x))
            acc += sum(Flux.onecold(pred) .== Flux.onecold(y))
        end
        Flux.trainmode!(model.layers)
        acc /= num_eval_samples
        return acc
    end

    loss = make_loss(model)

    function total_loss()
        l = 0f0
        for (x, y) in train_data_loader
            l += loss(x, y)
        end
        return l / length(train_data_loader)
    end

    method_name = "point_$(model.quant.point_estimate)"

    opt = make_optimizer()
    prog = ProgressMeter.Progress(nepochs, desc="training: ")
    for epoch in 1:nepochs
        if (epoch - 1) % report_gap == 0
            println(
                output,
                method_name, ',',
                epoch - 1, ',',
                num_train_samples, ',',
                num_eval_samples, ',',
                eval_accuracy())
            flush(output)
        end

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

    opt = make_optimizer()
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

    opt = make_optimizer()
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
    el_vecs = Vector{Float32}[]

    for sample in spec["samples"]
        input = h5open(sample["file"])
        push!(μ_vecs, read(input["mu"]))
        push!(σ_vecs, exp.(read(input["omega"])))
        push!(α_vecs, read(input["alpha"]))
        push!(el_vecs, read(input["effective_lengths"]))
        close(input)
    end

    train_classes = hcat(
        [Flux.onehot(sample["factors"][model.factor], model.classes)
         for sample in spec["samples"]]...)

    μs = hcat(μ_vecs...)
    σs = hcat(σ_vecs...)
    αs = hcat(α_vecs...)
    els = hcat(el_vecs...)

    return μs, σs, αs, els, train_classes
end


function load_beta_pttlatent_data(model::Classifier{PTTLatentExpr}, spec::Dict)
    α_vecs = Vector{Float32}[]
    β_vecs = Vector{Float32}[]
    el_vecs = Vector{Float32}[]

    for sample in spec["samples"]
        input = h5open(sample["file"])
        push!(α_vecs, read(input["alpha"]))
        push!(β_vecs, read(input["beta"]))
        push!(el_vecs, read(input["effective_lengths"]))
        close(input)
    end

    train_classes = hcat(
        [Flux.onehot(sample["factors"][model.factor], model.classes)
         for sample in spec["samples"]]...)

    αs = hcat(α_vecs...)
    βs = hcat(β_vecs...)
    els = hcat(el_vecs...)

    return αs, βs, els, train_classes
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


@enum Approximation begin
    LogitSkewNormalApprox
    BetaApprox
end


"""
Approx likelihood sampler operating on multiple samples at once.
"""
struct VecApproxLikelihoodSampler{T}
    ys::Array{Float64, 2}
    us::Array{Float64, 2}
    t::Polee.PolyaTreeTransform
end


function VecApproxLikelihoodSampler{T}(t::Polee.PolyaTreeTransform, n::Int, m::Int) where {T}
    return VecApproxLikelihoodSampler{T}(
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
    sampler::VecApproxLikelihoodSampler{LogitSkewNormalApprox}, μ, σ, α, els, x)

    ys = sampler.ys
    us = sampler.us
    t = sampler.t

    n = size(μ, 1) + 1
    m = size(μ, 2)

    # randn -> sinh-asinh -> logit-normal
    eps = 1e-14
    Threads.@threads for i in 1:(n-1)*m
        ys[i] = clamp(logistic(sinh(asinh(randn(Float32)) + α[i]) * σ[i] + μ[i]), eps, 1-eps)
    end

    ptt_transform!(t, x, us, ys, els)
end


function Polee.rand!(
    sampler::VecApproxLikelihoodSampler{BetaApprox}, α, β, els, x)

    ys = sampler.ys
    us = sampler.us
    t = sampler.t

    n, m = size(x)

    # rand beta
    eps = 1e-14
    Threads.@threads for i in 1:(n-1)*m
        ys[i] = clamp(rand(Beta(Float64(α[i]), Float64(β[i]))), eps, 1-eps)
    end

    ptt_transform!(t, x, us, ys, els)
end


function ptt_transform!(t::Polee.PolyaTreeTransform, x, us, ys, els)
    n, m = size(x)
    eps = 1e-14
    epsf = 1f-14

    # polya tree
    us[1,:] .= 1.0

    chunks = collect(Iterators.partition(1:m, 8))
    Threads.@threads for chunk in chunks
        k = 1
        for i in 1:2n-1
            output_idx = t.index[1, i]
            if output_idx != 0
                for j in chunk
                    x[output_idx, j] = clamp(Float32(us[i, j]), epsf, 1f0 - epsf)
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
            for j in chunk
                v = ys[k,j] * us[i,j]
                us[left_idx, j] = max(v, eps)
                us[right_idx, j] = max(us[i,j] - v, eps)
            end

            k += 1
        end
    end

    # effective length transformation
    x ./= els
    x ./= sum(x, dims=1)
end



# Alternative fitting scheme where we just sample a big bunch all at once
# so we don't have to copy back and forth from the gpu.
function fit_samples!(
        model::Classifier{PTTLatentExpr}, train_spec::Dict, nsamples::Int,
        modelfn::Function=build_model, nepochs::Int=nepochs)

    model.classes = get_classes(train_spec, model.factor)
    μs, σs, αs, train_classes = load_pttlatent_data(model, train_spec)

    fit_samples!(
        model, train_spec, μs, σs, αs, train_classes, nsamples,
        modelfn, nepochs)
end


function fit_samples!(
        model::Classifier{PTTLatentExpr}, train_spec::Dict,
        μs, σs, αs, train_classes, nsamples::Int,
        modelfn::Function=build_model, nepochs::Int=nepochs)

    model.classes = get_classes(train_spec, model.factor)

    # generate data
    n = size(μs, 1) + 1
    m = size(μs, 2)
    sampler = VecApproxLikelihoodSampler{LogitSkewNormalApprox}(model.quant.t, n, m)
    expanded_train_expr = Array{Float32}(undef, n, nsamples*m)
    for i in 1:nsamples
        Polee.rand!(sampler, μs, σs, αs, @view expanded_train_expr[:,((i-1)*m+1):(i*m)])
    end

    expanded_train_classes = hcat([train_classes for _ in 1:nsamples]...)

    train_data_loader = device.(Flux.Data.DataLoader(
        expr_trans(expanded_train_expr), expanded_train_classes,
        batchsize=batchsize, shuffle=true))

    # train as if we had point estimates
    n_out = length(model.classes)
    model.layers = device(modelfn(n, n_out))

    function total_loss()
        l = 0f0
        for (x, y) in train_data_loader
            l += Flux.Losses.logitcrossentropy(model.layers(x), y)
        end
        return l / length(train_data_loader)
    end

    loss = make_loss(model)

    opt = make_optimizer()
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


function fit!(
        model::Classifier{PTTLatentExpr}, train_spec::Dict,
        modelfn::Function=build_model, nepochs::Int=nepochs)
    # Just going to assume logit-skew-normal approx here. May need
    # to generalize in the future if we start using beta approx.

    model.classes = get_classes(train_spec, model.factor)
    μs, σs, αs, train_classes = load_pttlatent_data(model, train_spec)

    return fit!(model, train_spec, μs, σs, αs, train_classes, modelfn, nepochs)
end


function fit!(
        model::Classifier{PTTLatentExpr}, train_spec::Dict,
        μs, σs, αs, train_classes,
        modelfn::Function=build_model, nepochs::Int=nepochs)

    model.classes = get_classes(train_spec, model.factor)

    train_data_loader = Flux.Data.DataLoader(
        μs, σs, αs,
        device(train_classes),
        batchsize=batchsize, shuffle=true)

    n = size(μs, 1) + 1
    n_out = length(model.classes)
    model.layers = device(modelfn(n, n_out))

    sampler = VecApproxLikelihoodSampler{LogitSkewNormalApprox}(model.quant.t, n, batchsize)

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

    loss = make_loss(model)

    # custom training loop that handles drawing samples
    opt = make_optimizer()
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
    μs, σs, αs, eval_classes = load_pttlatent_data(model, eval_spec)
    eval(model, eval_spec, μs, σs, αs, eval_classes)
end


function eval(model::Classifier{PTTLatentExpr}, eval_spec::Dict, μs, σs, αs, eval_classes)
    Flux.testmode!(model.layers)

    eval_data_loader = Flux.Data.DataLoader(
        μs, σs, αs, eval_classes, batchsize=batchsize)

    n = size(μs, 1) + 1
    m = size(μs, 2)
    sampler = VecApproxLikelihoodSampler{LogitSkewNormal}(model.quant.t, n, batchsize)

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


function fit_and_monitor!(
        model::Classifier{PTTLatentExpr},
        train_spec::Dict, eval_spec::Dict,
        μs_train, σs_train, αs_train, els_train, train_classes,
        μs_eval, σs_eval, αs_eval, els_eval, eval_classes,
        modelfn::Function, nepochs::Int,
        output::IO, report_gap::Int=25)

    model.classes = get_classes(train_spec, model.factor)
    num_train_samples = size(μs_train, 2)

    train_data_loader = Flux.Data.DataLoader(
        μs_train, σs_train, αs_train, els_train,
        device(train_classes),
        batchsize=batchsize, shuffle=true)

    n = size(μs_train, 1) + 1
    n_out = length(model.classes)
    model.layers = device(modelfn(n, n_out))

    num_eval_samples = size(μs_eval, 2)

    eval_data_loader = Flux.Data.DataLoader(
        μs_eval, σs_eval, αs_eval, els_eval, eval_classes, batchsize=batchsize)

    sampler = VecApproxLikelihoodSampler{LogitSkewNormalApprox}(model.quant.t, n, batchsize)

    function eval_accuracy()
        Flux.testmode!(model.layers)
        acc = 0.0
        for (μ, σ, α, el, y) in eval_data_loader
            pred = zeros(Float32, length(model.classes), size(μ, 2))
            for i in 1:model.quant.neval_samples

                x = Array{Float32}(undef, n, size(μ, 2))
                Polee.rand!(sampler, μ, σ, α, el, x)
                x_gpu = expr_trans(device(x))
                pred .+= Flux.softmax(cpu(model.layers(x_gpu)))
            end
            pred ./= model.quant.neval_samples

            acc += sum(Flux.onecold(pred) .== Flux.onecold(y))
        end
        Flux.trainmode!(model.layers)

        acc /= num_eval_samples
        return acc
    end

    loss = make_loss(model)

    function total_loss()
        l = 0f0
        for (μ, σ, α, el, y) in train_data_loader
            x = Array{Float32}(undef, n, size(μ, 2))
            Polee.rand!(sampler, μ, σ, α, el, x)
            x_gpu = expr_trans(device(x))
            l += loss(x_gpu, y)
        end
        return l / length(train_data_loader)
    end

    opt = make_optimizer()
    prog = ProgressMeter.Progress(nepochs, desc="training: ")
    last_total_loss = Inf
    n_grad_samples = 3
    for epoch in 1:nepochs
        # This is expensive, so let's only do it so often
        if (epoch - 1) % report_gap == 0
            last_total_loss = total_loss()
        end

        if (epoch - 1) % report_gap == 0
            println(
                output,
                "approx_likelihood", ',',
                epoch - 1, ',',
                num_train_samples, ',',
                num_eval_samples, ',',
                eval_accuracy())
            flush(output)
        end

        ps = params(model.layers)
        for (μ, σ, α, el, y) in train_data_loader
            x = Array{Float32}(undef, n, size(μ, 2))
            # Polee.rand!(sampler, μ, σ, α, el, x)
            # x_gpu = expr_trans(device(x))

            gs = gradient(ps) do
                mean_loss = 0.f0
                for i in 1:n_grad_samples
                    x_gpu = Zygote.ignore() do
                        Polee.rand!(sampler, μ, σ, α, el, x)
                        return expr_trans(device(x))
                    end
                    mean_loss += loss(x_gpu, y)
                end
                mean_loss /= n_grad_samples
            end

            # gs = gradient(ps) do
            #     return loss(x_gpu, y)
            # end

            Flux.update!(opt, ps, gs)
        end
        ProgressMeter.next!(prog, showvalues = [(:loss, last_total_loss)])
    end
end


function fit_and_monitor_beta!(
        model::Classifier{PTTLatentExpr},
        train_spec::Dict, eval_spec::Dict,
        αs_train, βs_train, els_train, train_classes,
        αs_eval, βs_eval, els_eval, eval_classes,
        modelfn::Function, nepochs::Int,
        output::IO, report_gap::Int=25)

    model.classes = get_classes(train_spec, model.factor)
    num_train_samples = size(αs_train, 2)

    train_data_loader = Flux.Data.DataLoader(
        αs_train, βs_train, els_train,
        device(train_classes),
        batchsize=batchsize, shuffle=true)

    n = size(αs_train, 1) + 1
    n_out = length(model.classes)
    model.layers = device(modelfn(n, n_out))

    num_eval_samples = size(αs_eval, 2)

    eval_data_loader = Flux.Data.DataLoader(
        αs_eval, βs_eval, els_eval, eval_classes, batchsize=batchsize)

    sampler = VecApproxLikelihoodSampler{BetaApprox}(model.quant.t, n, batchsize)

    function eval_accuracy()
        Flux.testmode!(model.layers)
        acc = 0.0
        for (α, β, el, y) in eval_data_loader
            pred = zeros(Float32, length(model.classes), size(α, 2))
            for i in 1:model.quant.neval_samples

                x = Array{Float32}(undef, n, size(α, 2))
                Polee.rand!(sampler, α, β, el, x)
                x_gpu = expr_trans(device(x))
                pred .+= Flux.softmax(cpu(model.layers(x_gpu)))
            end
            pred ./= model.quant.neval_samples

            acc += sum(Flux.onecold(pred) .== Flux.onecold(y))
        end
        Flux.trainmode!(model.layers)

        acc /= num_eval_samples
        return acc
    end

    loss = make_loss(model)

    function total_loss()
        l = 0f0
        for (α, β, el, y) in train_data_loader
            x = Array{Float32}(undef, n, size(α, 2))
            Polee.rand!(sampler, α, β, el, x)
            x_gpu = expr_trans(device(x))
            l += loss(x_gpu, y)
        end
        return l / length(train_data_loader)
    end

    opt = make_optimizer()
    prog = ProgressMeter.Progress(nepochs, desc="training: ")
    last_total_loss = Inf
    for epoch in 1:nepochs
        # This is expensive, so let's only do it so often
        if (epoch - 1) % report_gap == 0
            last_total_loss = total_loss()
        end

        if (epoch - 1) % report_gap == 0
            println(
                output,
                "approx_likelihood", ',',
                epoch - 1, ',',
                num_train_samples, ',',
                num_eval_samples, ',',
                eval_accuracy())
            flush(output)
        end

        ps = params(model.layers)
        for (α, β, el, y) in train_data_loader
            x = Array{Float32}(undef, n, size(α, 2))
            Polee.rand!(sampler, α, β, el, x)
            x_gpu = expr_trans(device(x))

            gs = gradient(ps) do
                return loss(x_gpu, y)
            end
            Flux.update!(opt, ps, gs)
        end
        ProgressMeter.next!(prog, showvalues = [(:loss, last_total_loss)])
    end
end


function fit_and_monitor_splits!(
        model::Classifier{PTTLatentExpr},
        train_spec::Dict, eval_spec::Dict,
        μs_train, σs_train, αs_train, els_train, train_classes,
        μs_eval, σs_eval, αs_eval, els_eval, eval_classes,
        modelfn::Function, nepochs::Int,
        output::IO, report_gap::Int=25)

    model.classes = get_classes(train_spec, model.factor)
    num_train_samples = size(μs_train, 2)

    train_data_loader = Flux.Data.DataLoader(
        μs_train, σs_train, αs_train, els_train,
        device(train_classes),
        batchsize=batchsize, shuffle=true)

    n = size(μs_train, 1) + 1
    n_out = length(model.classes)
    model.layers = device(modelfn(n-1, n_out))

    num_eval_samples = size(μs_eval, 2)

    eval_data_loader = Flux.Data.DataLoader(
        μs_eval, σs_eval, αs_eval, els_eval, eval_classes, batchsize=batchsize)

    function sample_ys!(ys, μ, σ, α)
        nm1, m = size(ys)
        eps = 1e-14
        Threads.@threads for i in 1:nm1*m
            ys[i] = clamp(logistic(sinh(asinh(randn(Float32)) + α[i]) * σ[i] + μ[i]), eps, 1-eps)
        end
    end

    function eval_accuracy()
        acc = 0.0
        for (μ, σ, α, el, y) in eval_data_loader
            pred = zeros(Float32, length(model.classes), size(μ, 2))
            for i in 1:model.quant.neval_samples
                u = Array{Float32}(undef, n-1, size(μ, 2))
                sample_ys!(u, μ, σ, α)
                u_gpu = device(u)
                pred .+= Flux.softmax(cpu(model.layers(u_gpu)))
            end
            pred ./= model.quant.neval_samples

            acc += sum(Flux.onecold(pred) .== Flux.onecold(y))
        end

        acc /= num_eval_samples
        return acc
    end

    loss = make_loss(model)

    function total_loss()
        l = 0f0
        for (μ, σ, α, el, y) in train_data_loader
            u = Array{Float32}(undef, n-1, size(μ, 2))
            sample_ys!(u, μ, σ, α)
            u_gpu = device(u)
            l += loss(u_gpu, y)
        end
        return l / length(train_data_loader)
    end

    opt = make_optimizer()
    prog = ProgressMeter.Progress(nepochs, desc="training: ")
    last_total_loss = Inf
    for epoch in 1:nepochs
        # This is expensive, so let's only do it so often
        if (epoch - 1) % report_gap == 0
            last_total_loss = total_loss()
        end

        if (epoch - 1) % report_gap == 0
            println(
                output,
                "approx_likelihood", ',',
                epoch - 1, ',',
                num_train_samples, ',',
                num_eval_samples, ',',
                eval_accuracy())
            flush(output)
        end

        ps = params(model.layers)
        for (μ, σ, α, el, y) in train_data_loader
            u = Array{Float32}(undef, n-1, size(μ, 2))
            sample_ys!(u, μ, σ, α)
            @show extrema(u)
            u_gpu = device(u)

            gs = gradient(ps) do
                return loss(u_gpu, y)
            end
            Flux.update!(opt, ps, gs)
        end
        ProgressMeter.next!(prog, showvalues = [(:loss, last_total_loss)])
    end
end

end # module PoleeClassifier