
# A little program for computing pairwise distance matrices

const F32x8 = Vec{8, Float32}

function point_estimate_pairwise_distance(
        spec::Dict, transcriptome_filename::String,
        point_estimate::String, output_filename::String)

    ts, ts_metadata = Polee.read_transcripts_from_fasta(transcriptome_filename, Set{String}())

    expr_data = load_point_estimates_from_specification(
        spec,
        ts,
        ts_metadata,
        point_estimate)

    xs0 = Array(transpose(expr_data.x0_values))

    # TODO: this is for vlr
    xs0 .+= 1f-5

    @show extrema(xs0)

    map!(log, xs0, xs0)

    @show extrema(xs0)

    m = size(xs0, 2)

    collen = div(size(xs0, 1), 8, RoundUp) * 8 # round up for avx

    # pad with zeros for avx
    xs = zeros(Float32, collen, size(xs0, 2))
    for i in 1:size(xs0, 1), j in 1:size(xs0, 2)
        xs[i, j] = xs0[i, j]
    end

    D = zeros(Float32, m, m)

    println("computing pairwise distances...")
    all_columns_distance!(D, xs, collen)

    for j in 1:m, i in j:m
        D[i, j] = D[j, i]
    end

    writedlm(output_filename, D, ',')
end


function load_ptt_params(spec::Dict)
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

    μs = hcat(μ_vecs...)
    σs = hcat(σ_vecs...)
    αs = hcat(α_vecs...)
    els = hcat(el_vecs...)

    return μs, σs, αs, els
end


function ptt_pairwise_distance(spec::Dict, nsamples::Int, output_filename::String)
    input = h5open(spec["transformation"])
    t = Polee.PolyaTreeTransform(
        read(input["node_parent_idxs"]),
        read(input["node_js"]))
    close(input)

    μs, σs, αs, els = load_ptt_params(spec)

    n = size(μs, 1) + 1
    m = size(μs, 2)

    @show (n, m)

    sampler = VecApproxLikelihoodSampler{LogitSkewNormalApprox}(t, n, m)

    collen = div(n, 8, RoundUp) * 8 # round up for avx

    xs = zeros(Float32, collen, m)
    D = zeros(Float32, m, m)

    println("computing pairwise distances...")
    for iter in 1:nsamples
        @show iter
        Polee.rand!(sampler, μs, σs, αs, els, xs)

        # @show extrema(xs)

        # TODO: for vlr
        for i in 1:n, j in 1:m
            xs[i, j] = log(xs[i, j] + 1f-5)
        end

        @show extrema(xs)

        all_columns_distance!(D, xs, collen)
    end

    D ./= nsamples

    for j in 1:m, i in j:m
        D[i, j] = D[j, i]
    end

    writedlm(output_filename, D, ',')
end


function avx_l2(xs::Ptr{Float32}, ys::Ptr{Float32}, len::Int)
    @assert len % 8 == 0

    accum = zero(F32x8)
    @inbounds for i in 1:8:len
        x = vload(F32x8, xs+4*(i-1))
        y = vload(F32x8, ys+4*(i-1))
        accum += (x - y)^2
    end
    return sum(accum)
end


"""
variance of log ratios, assumes xs and ys are already log transformed.
"""
function avx_vlr(xs::Ptr{Float32}, ys::Ptr{Float32}, n::Int, len::Int)
    # compute mean difference
    accum = zero(F32x8)
    @inbounds for i in 1:8:len
        x = vload(F32x8, xs+4*(i-1))
        y = vload(F32x8, ys+4*(i-1))
        accum += x - y
    end
    mean_diff = sum(accum) / n

    # compute variance
    mean_diff_v = F32x8(mean_diff)
    accum = zero(F32x8)
    @inbounds for i in 1:8:len
        x = vload(F32x8, xs+4*(i-1))
        y = vload(F32x8, ys+4*(i-1))
        z = (x - y) - mean_diff
        accum += z^2
    end
    return sum(accum) / n
end


function all_columns_distance!(D::Matrix{Float32}, xs::Matrix{Float32}, nrow::Int)
    collen = size(xs, 1)
    Threads.@threads for idx in eachindex(UpperTriangular(D))
        i, j = Tuple(idx)
        if j > i
            D[i,j] += avx_vlr(
                pointer(xs, 1+collen*(i-1)),
                pointer(xs, 1+collen*(j-1)), nrow, collen)
            # D[i,j] += avx_l2(
            #     pointer(xs, 1+collen*(i-1)),
            #     pointer(xs, 1+collen*(j-1)), collen)
        end
    end
end
