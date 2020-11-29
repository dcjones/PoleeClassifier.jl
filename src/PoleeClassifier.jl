
module PoleeClassifier

using Polee
using Polee.PoleeModel
using Flux
using HDF5
import CUDA
import ProgressMeter

const device = Flux.gpu
# const device = Flux.cpu

const batchsize = 50
# const nepochs = 1000
const nepochs = 200

"""
Generic transformation applid to expression vectors. This seems to make
training easier.
"""
expr_trans(x) = log.(x) .- log(1f0/size(x, 1))

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
    M = 50
    return Chain(
        Dense(n_in, M, leakyrelu, initW=initW),
        # Dropout(0.25, M),
        Dense(M, M, leakyrelu, initW=initW),
        # Dropout(0.25, M),
        # Dense(M, M, leakyrelu, initW=initW),
        Dense(M, n_out, initW=initW))
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

    opt = ADAM()
    Flux.@epochs nepochs Flux.Optimise.train!(
            (x, y) -> Flux.Losses.logitcrossentropy(model.layers(x), y),
            params(model.layers),
            train_data_loader,
            opt,
            cb=() -> @show total_loss())
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

    acc = 0.0
    for i in 1:num_samples
        pred = cpu(model.layers(device(eval_expr[:,i])))
        acc += eval_spec["samples"][i]["factors"][model.factor] == model.classes[Flux.onecold(pred)]
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
    Threads.@threads for i in 1:size(μ, 2)
        Polee.set_transform!(als, t, μ[:,i], σ[:,i], α[:,i])
        Polee.rand!(als, @view x[:,i])
    end
    return x
end


function fit!(model::Classifier{PTTLatentExpr}, train_spec::Dict)
    # Just going to assume logit-skew-normal approx here. May need
    # to generalize in the future if we start using beta approx.

    # TODO: This is insanely slow. I think computing samples on the cpu then
    # transferring to the gpu may be a big bottleneck. We should be able to
    # use CUDA.jl to build a gpu ptt implementation.

    model.classes = get_classes(train_spec, model.factor)
    μs, σs, αs, train_classes = load_pttlatent_data(model, train_spec)

    train_data_loader = device.(Flux.Data.DataLoader(
        μs, σs, αs,
        train_classes,
        batchsize=batchsize, shuffle=true))

    n = size(μs, 1) + 1
    n_out = length(model.classes)
    model.layers = device(build_model(n, n_out))

    als = [Polee.ApproxLikelihoodSampler() for _ in 1:batchsize]

    function total_loss()
        l = 0f0
        for (μ, σ, α, y) in train_data_loader
            # x = Array{Float32}(undef, (n, size(μ, 2)))
            # for i in 1:size(μ, 2)
            #     Polee.set_transform!(als, model.quant.t, μ[:,i], σ[:,i], α[:,i])
            #     Polee.rand!(als, @view x[:,i])
            # end
            # l += Flux.Losses.logitcrossentropy(
            #     model.layers(device(expr_trans(x))),
            #     device(y))

            # x = expr_trans(approx_sample(model.quant.t, als, μ, σ, α))
            # l += Flux.Losses.logitcrossentropy(model.layers(device(x)), y)

            x = expr_trans(cuda_approx_sample(model.quant.t, μ, σ, α))
            l += Flux.Losses.logitcrossentropy(model.layers(x), y)
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
            # x = Array{Float32}(undef, (n, size(μ, 2)))
            # for i in 1:size(μ, 2)
            #     Polee.set_transform!(als, model.quant.t, μ[:,i], σ[:,i], α[:,i])
            #     Polee.rand!(als, @view x[:,i])
            # end
            # x_gpu = device(expr_trans(x))
            # y_gpu = device(y)

            x = expr_trans(cuda_approx_sample(model.quant.t, μ, σ, α))

            # x = expr_trans(approx_sample(model.quant.t, als, μ, σ, α))

            gs = gradient(ps) do
                return loss(x, y)
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
    x = Vector{Float32}(undef, n)
    # als = Polee.ApproxLikelihoodSampler()
    als = [Polee.ApproxLikelihoodSampler() for _ in 1:batchsize]

    acc = 0.0
    total_count = 0
    pred = zeros(Float32, length(model.classes))
    for (μ, σ, α, y) in eval_data_loader
        # x = expr_trans(approx_sample(model.quant.t, als, μ, σ, α))
        x = expr_trans(cuda_approx_sample(model.quant.t, μ, σ, α))
        pred = model.layers(x)
        acc += sum(Flux.onecold(pred) .== Flux.onecold(y))

        # @assert size(μ, 2) == 1
        # Polee.set_transform!(als, model.quant.t, μ[:,1], σ[:,1], α[:,1])
        # for i in 1:model.quant.neval_samples
        #     Polee.rand!(als, x)
        #     pred .+= Flux.softmax(cpu(model.layers(device(expr_trans(x)))))
        # end
        # pred /= model.quant.neval_samples
        # acc += Flux.onecold(pred) == Flux.onecold(y[:,1])
        # total_count += 1
    end

    acc /= m
    return acc
end


end # module PoleeClassifier