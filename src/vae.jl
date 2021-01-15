
using YAML
using HDF5
using Flux
using Distributions
using Zygote
using ZygoteRules
using Random
using Polee: PolyaTreeTransform
using StaticArrays
using NaNMath

include("gamma_adjoint.jl")


function load_pttlatent_data(spec::Dict)
    α_vecs = Vector{Float32}[]
    β_vecs = Vector{Float32}[]
    el_vecs = Vector{Float32}[]

    for sample in spec["samples"]
        input = h5open(sample["file"])
        push!(α_vecs, read(input["alpha"]))
        push!(β_vecs, read(input["beta"]))
        close(input)
    end

    αs = hcat(α_vecs...)
    βs = hcat(β_vecs...)

    return αs, βs
end

# Weight initialization
initW = (dims...) -> 1f-4 * randn(Float32, dims...)
# σ_initb = (dims...) -> ones(Float32, dims...)

struct Encoder
    f
    fμ
    fσ

    function Encoder(n::Int, hiddendim::Int, latentdim::Int, device)
        return new(
            Chain(
                Dense(2*(n-1), hiddendim, tanh),
                # Dense(M, M, leakyrelu),
                ) |> device,
            Dense(hiddendim, latentdim, identity) |> device,
            Dense(hiddendim, latentdim, identity) |> device)
    end
end


function (encoder::Encoder)(α, β)
    h = encoder.f(vcat(α, β))
    return encoder.fμ(h), encoder.fσ(h)
end



function Decoder(latentdim::Int, hiddendim::Int, n::Int, device)
    initb = (dims...) -> ones(Float32, dims...)
    return Chain(
        Dense(latentdim, hiddendim, leakyrelu),
        # Dense(hiddendim, hiddendim, leakyrelu),
        Dense(hiddendim, n, softplus, initb=initb)) |> device
end


"""
Send expression vectors xs to splits ys using transformation t, and intermediate
space us.
"""
function inv_transform(t::PolyaTreeTransform, xs)
    n, m = size(xs)
    eps = 1e-10
    epsf = 1f-10

    num_nodes = size(t.index, 2)
    @assert div(num_nodes + 1, 2) == n

    ys = Array{Float64}(undef, n-1, m)
    us = Array{Float64}(undef, 2*n-1, m)
    ladj = zeros(Float64, m)

    # polya tree
    us[1,:] .= 1.0

    chunks = collect(Iterators.partition(1:m, 8))
    Threads.@threads for chunk in chunks
        k = n - 1 # internal node number
        for i in num_nodes:-1:1
            output_idx = t.index[1, i]

            # leaf node
            if output_idx != 0
                for j in chunk
                    us[i, j] = xs[output_idx, j]
                end
                continue
            end

            left_idx = t.index[2, i]
            right_idx = t.index[3, i]

            for j in chunk
                us[i,j] = us[left_idx,j] + us[right_idx,j]
                # ys[k,j] = clamp(us[left_idx,j] / us[i,j], 1e-10, 1-1e-10)
                ys[k,j] = us[left_idx,j] / us[i,j]
                ladj[j] -= log(us[i,j])
            end

            k -= 1
        end
    end

    return ys, ladj
end


Zygote.@adjoint function inv_transform(t::PolyaTreeTransform, xs)
    ys, ladj = inv_transform(t, xs)

    function inv_transform_pullback(ȳ_ladj_bar)
        ȳ, ladj_bar = ȳ_ladj_bar
        @show extrema(ȳ)
        n = size(ȳ, 1) + 1
        m = size(ȳ, 2)
        num_nodes = 2*n-1

        us = Array{Float64}(undef, num_nodes, m)
        us[1,:] .= 1.0

        vs = Array{Float64}(undef, num_nodes, m)
        vs[1,:] .= 0.0

        ∂x = Array{Float32}(undef, n, m)
        @show size(∂x)

        chunks = collect(Iterators.partition(1:m, 8))
        Threads.@threads for chunk in chunks
            k = 1 # internal node number
            for i in 1:num_nodes
                output_idx = t.index[1, i]
                if output_idx != 0
                    for j in chunk
                        ∂x[output_idx, j] = vs[i, j]
                    end
                    continue
                end

                left_idx = t.index[2, i]
                right_idx = t.index[3, i]

                for j in chunk
                    u_left = us[i, j] * ys[k, j]
                    u_right = us[i, j] * (1.0 - ys[k, j])
                    dladj_du = -1.0/us[i,j];
                    u2 = us[i, j]^2

                    vs[left_idx, j] = dladj_du * ladj_bar[j] + vs[i, j] + (u_right/u2) * ȳ[k, j]
                    vs[right_idx, j] = dladj_du * ladj_bar[j] + vs[i, j] - (u_left/u2) * ȳ[k, j]

                    us[left_idx, j] = u_left
                    us[right_idx, j] = u_right
                end

                k += 1
            end
        end

        @show extrema(∂x)

        return nothing, ∂x
    end

    return (ys, ladj), inv_transform_pullback
end


# Generate size(λ, 2) dirichlet vectors, with concentrations
# λ[1,:], λ[2,:], ...
function rand_dirichlet(λ)
    x = rand_gamma1s(λ)
    return x ./ sum(x, dims=1)
end


function elbo(t::PolyaTreeTransform, encoder, decoder, α, β, device)
    # sample from latent space posterior

    @show size(α)

    μ, logσ2 = encoder(α, β)
    σ2 = exp.(logσ2)
    σ = sqrt.(σ2)

    @show extrema(cpu(μ))
    @show extrema(cpu(σ))

    z = μ + device(randn(Float32, size(μ))) .* σ

    @show extrema(cpu(z))

    # # decode into dirichlet concentrations
    λ = decoder(z) .+ 1f0

    @show extrema(cpu(λ))

    # # sample expression
    x = rand_dirichlet(cpu(λ))

    # @show extrema(x)

    # TODO: Probably there is something wrong with inv_transfor gradients.
    # ll goes down a little, but gradients for x keep growing to insane levels
    # and then we a get a nan.


    # TODO: I think I really actually do need to compute the ladj here to
    # correctly compute q(x; α, β)

    # # reverse ptt
    y, ptt_ladj = inv_transform(t, x)

    @show ptt_ladj

    # # sending y to the gpu converts to Float32, we we need to compute
    # # particularily log.(1 .- y) in Float64 precision.
    # # TODO: use threads for this
    logy = device(log.(y))
    log1my = device(log.(1 .- y))

    # @show extrema(logy)
    # @show extrema(log1my)

    # # evaluate beta log likelihood
    ll = sum((α .- 1f0) .* logy + (β .- 1f0) .* log1my) + sum(device(ptt_ladj))
    # @show ll

    # kl between two normals
    # TODO: We get -infs when σ is allowed to get too small
    # σ2 = σ.^2
    # @show extrema(cpu(log.(σ2)))
    # @show extrema(cpu(log.(μ.^2)))
    kl = -0.5f0 .* sum(logσ2 .- σ2 .- μ.^2 .+ 1f0)

    @show cpu(ll)
    @show cpu(kl)

    return ll - kl
    # return -kl
end


# For debugging, see if we can successfully optimize dirichlet concentrations
# directly.
function dirichlet_ll(t, logλ, α, β, device)
    # @show (typeof(α), size(α))
    # @show (typeof(β), size(β))
    # @show (typeof(logλ), size(logλ))

    λ = 1f0 .+ exp.(logλ)
    # x = rand_dirichlet(cpu(λ))

    # @show typeof(λ)
    # @show size(λ)
    # Zygote.dropgrad(@show extrema(cpu(λ)))

    λsum = sum(λ, dims=1)

    # @show typeof(λsum)
    # @show size(λsum)
    # Zygote.dropgrad(@show extrema(cpu(λsum)))

    # Let's try using the mean to see if rand_dirichlet is the problem
    x = λ ./ λsum

    # Zygote.dropgrad(@show extrema(cpu(x)))

    # # reverse ptt
    y, ptt_ladj = inv_transform(t, cpu(x))

    logy = device(log.(y))
    log1my = device(log.(1 .- y))

    # # evaluate beta log likelihood
    ll = sum((α .- 1f0) .* logy + (β .- 1f0) .* log1my) + sum(device(ptt_ladj))

    @show cpu(ll)

    return ll
end



function optimize_dirichlet_λ()
    ptt_filename = ARGS[1]
    input = h5open(ptt_filename)
    t = PolyaTreeTransform(
        read(input["node_parent_idxs"]),
        read(input["node_js"]))
    close(input)

    experiment_filename = ARGS[2]
    spec = YAML.load_file(experiment_filename)
    αs, βs = load_pttlatent_data(spec)

    n = size(αs, 1) + 1
    numobs = size(αs, 2)

    # device = cpu
    device = gpu
    nepochs = 50

    logλ = device(zeros(Float32, (n, numobs)))
    # @show dirichlet_ll(t, logλ, αs, βs, device)

    ps = [logλ]

    opt = ADAM(1e-3)
    Flux.@epochs nepochs Flux.Optimise.train!(
        (α, β) -> -dirichlet_ll(t, logλ, α, β, device),
        ps, [(αs, βs)], opt)
end



function main()
    ptt_filename = ARGS[1]
    input = h5open(ptt_filename)
    t = PolyaTreeTransform(
        read(input["node_parent_idxs"]),
        read(input["node_js"]))
    close(input)

    experiment_filename = ARGS[2]
    spec = YAML.load_file(experiment_filename)
    αs, βs = load_pttlatent_data(spec)

    n = size(αs, 1) + 1
    numobs = size(αs, 2)

    # device = gpu
    device = cpu
    batchsize = 20
    hiddendim = 50
    latentdim = 10
    nepochs = 50

    function total_elbo()
        l = 0f0
        for (α, β) in data_loader
            l += elbo(t, encoder, decoder, α, β, device)
        end
        return l
    end

    data_loader = device.(Flux.Data.DataLoader(
        αs, βs, batchsize=batchsize, shuffle=true))

    encoder = Encoder(n, hiddendim, latentdim, device)
    decoder = Decoder(latentdim, hiddendim, n, device)

    ps = Flux.params(encoder.f, encoder.fμ, encoder.fσ, decoder)

    Flux.@epochs nepochs Flux.Optimise.train!(
        (α, β) -> -elbo(t, encoder, decoder, α, β, device),
        ps, data_loader, opt)
        # cb=() -> @show total_elbo())
end


# main()
optimize_dirichlet_λ()
