
module PoleeClassifier

using Polee
using Polee.PoleeModel
using Flux
using HDF5
using ProgressMeter

const device = Flux.gpu
# const device = Flux.cpu

const batchsize = 50
const nepochs = 1000

# Different ways of approaching quantification each with a different take
# on training the clasifier
abstract type QuantMethod end

mutable struct Classifier{T <: QuantMethod}
    quant::T
    layers::Union{Chain, Nothing}
    factor::String
    classes::Vector{String}
end


struct PointEstimate <: QuantMethod
    ts::Polee.Transcripts
    ts_metadata::Polee.TranscriptsMetadata
    point_estimate::String
end


function get_classes(spec::Dict, factor::String)
    classes = Set{String}()
    for sample in spec["samples"]
        push!(classes, sample["factors"][factor])
    end
    return collect(classes)
end


"""
Construct what will be our standard classifier given input and output size.
"""
function build_model(n_in::Int, n_out::Int)
    initW = (dims...) -> 1e-3 * randn(Float32, dims...)
    M = 50
    return Chain(
        Dense(n_in, M, leakyrelu, initW=initW),
        Dropout(0.25, M),
        Dense(M, M, leakyrelu, initW=initW),
        Dropout(0.25, M),
        Dense(M, M, leakyrelu, initW=initW),
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
    train_expr = Array(transpose(log.(train_data.x0_values))) .- log(1/n)
    train_classes = hcat(
        [Flux.onehot(sample["factors"][model.factor], model.classes)
         for sample in train_spec["samples"]]...)
jj
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
    eval_expr = Array(transpose(log.(eval_data.x0_values))) .- log(1/n)

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

    expr_data = Array(transpose(log.(vcat(expr_data_vecs...)))) .- log(1/n)
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

            pred .+= cpu(model.layers(device(log.(bs[1,:]) .- log(1/n))))
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


# TODO: Figure out how to set up a sampler as the input to the model. Probably
# a custom training loop is the easiest way to pull that off.
struct PTTLatentExpr <: QuantMethod end


end # module PoleeClassifier