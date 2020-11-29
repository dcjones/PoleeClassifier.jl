
module PoleeClassifier

using Polee
using Polee.PoleeModel
using Flux

const device = Flux.gpu
# const device = Flux.cpu


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

    train_data_loader = device.(Flux.Data.DataLoader(
        train_expr, train_classes,
        batchsize=50, shuffle=true))

    function total_loss()
        l = 0f0
        for (x, y) in train_data_loader
            l += Flux.Losses.logitcrossentropy(model.layers(x), y)
        end
        return l / length(train_data_loader)
    end

    @show typeof(train_data_loader)

    opt = ADAM(1e-3)
    Flux.@epochs 300 Flux.Optimise.train!(
            (x, y) -> Flux.Losses.logitcrossentropy(model.layers(x), y),
            params(model.layers),
            train_data_loader,
            opt,
            cb=() -> @show total_loss())
end


function eval(model::Classifier{PointEstimate}, eval_spec::Dict)
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


struct KallistoBootstrap <: QuantMethod end
    ts::Polee.Transcripts
    ts_metadata::Polee.TranscriptsMetadata
    pseudocount::Float32
end


function fit!(model::Classifier{KallistoBootstrap}, train_spec::Dict)
    # TODO: I think we should use bootstrap samples to simply inflate
    # the training data, rather than do any kind of normal approximation.

end


function eval(model::Classifier{KallistoBootstrap}, eval_spec::Dict)
    # TODO: Here I think we should take essentially an expectation using
    # bootstrap samples.
end


# TODO:
struct PTTLatentExpr         <: QuantMethod end
struct PTTBetaParams         <: QuantMethod end


end # module PoleeClassifier