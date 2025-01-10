module CVNN

export lbr
export complex_glorot_uniform, complex_aug_glorot_uniform
export zrelu, abslu, rrelu
export GenDescent, apply!, update!, update_params!
export GenerateSquareDiagonalData, GenerateSquareQuadrantData
export SwitchTrainer, SwitchTrainerSymplectic
export CovarianceMatrix, GradAnalysis, NSVecAnalysis, LossLandscapeAnalysis01

using Flux, Zygote, Optimisers
using Flux: logitcrossentropy, crossentropy, onecold, onehotbatch, mse
using Flux: params, train!, relu
import Flux.Optimise: apply!, Descent, AbstractOptimiser
using LinearAlgebra, Random, SparseArrays
using StatsBase: sample

lbr(x::Char) = println(x^80)
lbr() = lbr('=')

function complex_glorot_uniform(
    rng::AbstractRNG, dims::Integer...; gain::Real=1
)
    return Flux.glorot_uniform(rng, dims...; gain) +
           1im .* Flux.glorot_uniform(rng, dims...; gain)
end

complex_glorot_uniform(dims::Integer...; kw...) =
    complex_glorot_uniform(Random.default_rng(), dims...; kw...)

# augmented complex glorot uniform weight initialiser
function complex_aug_glorot_uniform(
    rng::AbstractRNG, dims::Integer...; gain::Real=1
)
    return 0im .+ Flux.glorot_uniform(rng, dims...; gain)
end

complex_aug_glorot_uniform(dims::Integer...; kw...) =
    complex_aug_glorot_uniform(Random.default_rng(), dims...; kw...)

zrelu(x::Number) = ifelse(0 <= angle(x) <= pi, x, zero(x))
abslu(x::Number) = abs(x)
rrelu(x::Number) = real(x)

mutable struct GenDescent <: AbstractOptimiser
    eta::Number
end

GenDescent() = GenDescent(0.1)

function apply!(o::GenDescent, x, Δ)
    Δ .*= o.eta
end

function update!(opt::AbstractOptimiser, ps::Tuple, gs::Tuple)
    for (p, g) ∈ zip(ps, gs)
        Flux.Optimise.update!(opt, p, g)
    end
end

function update_params!(model, params_new)
    for (old_param, new_param) ∈ zip(params(model), params_new)
        old_param .= new_param
    end
end

function GenerateSquareDiagonalData(rng::AbstractRNG, size::Int;)
    x = rand(rng, Float32, (2, size))
    y = [(x[:, i][2] >= x[:, i][1]) ? 1 : 0 for i ∈ 1:size]
    return (x, y)
end

function GenerateSquareQuadrantData(rng::AbstractRNG, size::Int;)
    x = rand(rng, Float32, (2, size)) * 2 .- 1
    y = []
    for i ∈ 1:size
        y_ = 0
        if x[:, i][1] >= 0 && x[:, i][2] >= 0
            y_ = 1
        elseif x[:, i][1] < 0 && x[:, i][2] >= 0
            y_ = 2
        elseif x[:, i][1] < 0 && x[:, i][2] < 0
            y_ = 3
        elseif x[:, i][1] >= 0 && x[:, i][2] < 0
            y_ = 4
        end
        push!(y, y_)
    end
    return (x, y)
end

function SwitchTrainer(
    model, loss, datalist, epoch_cfg;
    subf=nothing
)
    ps = params(model)
    ndata = length(datalist)
    epochs_list = [p.first for p in epoch_cfg]
    eta_list = [p.second[1] for p in epoch_cfg]
    datai_list = [p.second[2] for p in epoch_cfg]
    loss_lists = [[] for _ ∈ 1:ndata]
    lbr('=')
    subf_out = Any[]
    for tl_i ∈ eachindex(epoch_cfg)
        ps0 = deepcopy(params(model))
        normgrad = length(epoch_cfg[tl_i].second) > 2 ? true : false
        DI = datai_list[tl_i]
        ETA = eta_list[tl_i]
        DATA = datalist[DI]
        EPOCHS = epochs_list[tl_i]
        MODE = isa(ETA, Complex) ? "H" : "G"
        if !isnothing(subf) && tl_i == 1
            push!(subf_out, subf(model, loss, DATA, ps0))
        end
        println("Training loop $tl_i")
        lbr('-')
        for epoch ∈ 1:EPOCHS
            l, gs = withgradient(ps) do
                loss(model(DATA[1]), DATA[2])
            end
            if !normgrad
                update!(GenDescent(ETA), ps, gs)
            elseif normgrad
                alpha = epoch_cfg[tl_i].second[3]
                gradnorm = norm(reduce(vcat, [vec(gs[p]) for p ∈ ps]))
                update!(
                    GenDescent(ETA / (1 + alpha * gradnorm)), ps, gs
                )
            end
            losses = [
                loss(model(datalist[di][1]), datalist[di][2]) for di ∈ 1:ndata
            ]
            for di ∈ 1:ndata
                push!(loss_lists[di], losses[di])
            end
            println("[$tl_i-$MODE] E: $epoch; L ($DI) = $(loss_lists[DI][end])")
            if !isnothing(subf)
                push!(subf_out, subf(model, loss, DATA, ps0))
            end
        end
        lbr('=')
    end
    if !isnothing(subf)
        return loss_lists, subf_out
    else
        return loss_lists
    end
end

function SwitchTrainerSymplectic(
    model, loss, datalist, epoch_cfg;
    subf=nothing
)
    ps = params(model)
    ndata = length(datalist)
    epochs_list = [p.first for p in epoch_cfg]
    eta_list = [p.second[1] for p in epoch_cfg]
    datai_list = [p.second[2] for p in epoch_cfg]
    loss_lists = [[] for _ ∈ 1:ndata]
    lbr('=')
    subf_out = Any[]
    for tl_i ∈ eachindex(epoch_cfg)
        ps0 = deepcopy(params(model))
        int_method = nothing
        if length(epoch_cfg[tl_i].second) > 2
            int_method = epoch_cfg[tl_i].second[3]
        end
        DI = datai_list[tl_i]
        ETA = eta_list[tl_i]
        DATA = datalist[DI]
        EPOCHS = epochs_list[tl_i]
        MODE = isa(ETA, Complex) ? "H" : "G"
        if !isnothing(subf) && tl_i == 1
            push!(subf_out, subf(model, loss, DATA, ps0))
        end
        println("Training loop $tl_i")
        lbr('-')
        for epoch ∈ 1:EPOCHS
            if isnothing(int_method)
                l, gs = withgradient(ps) do
                    loss(model(DATA[1]), DATA[2])
                end
                Flux.Optimise.update!(GenDescent(ETA), ps, gs)
            elseif int_method == "norm"
                l, gs = withgradient(ps) do
                    loss(model(DATA[1]), DATA[2])
                end
                alpha = epoch_cfg[tl_i].second[4]
                gradnorm = norm(reduce(vcat, [vec(gs[p]) for p ∈ ps]))
                update!(
                    GenDescent(ETA / (1 + alpha * gradnorm)), ps, gs
                )
            elseif int_method == "semi-implicit-euler"
                l1, gs1 = withgradient(ps) do
                    loss(model(DATA[1]), DATA[2])
                end
                pst1 = tuple(ps...)
                gs1t_ = tuple(gs1...) .* ETA
                gs1t_re = (gs1t_ .+ conj.(gs1t_)) ./ 2
                # gs1t_im = (gs1t_ .- conj.(gs1t_)) ./ 2
                update!(GenDescent(1), pst1, gs1t_re)
                update_params!(model, pst1)
                l2, gs2 = withgradient(ps) do
                    loss(model(DATA[1]), DATA[2])
                end
                pst2 = tuple(ps...)
                gs2t_ = tuple(gs2...) .* ETA
                # gs2t_re = (gs2t_ .+ conj.(gs2t_)) ./ 2
                gs2t_im = (gs2t_ .- conj.(gs2t_)) ./ 2
                update!(GenDescent(1), pst2, gs2t_im)
                update_params!(model, pst2)
            elseif int_method == "normed-semi-implicit-euler"
                alpha = epoch_cfg[tl_i].second[4]
                l1, gs1 = withgradient(ps) do
                    loss(model(DATA[1]), DATA[2])
                end
                gnorm1 = norm(reduce(vcat, [vec(gs1[p]) for p ∈ ps]))
                pst1 = tuple(ps...)
                gs1t_ = tuple(gs1...) .* ETA
                gs1t_re = (gs1t_ .+ conj.(gs1t_)) ./ 2
                # gs1t_im = (gs1t_ .- conj.(gs1t_)) ./ 2
                update!(
                    GenDescent(1 / (1 + alpha * gnorm1)), pst1, gs1t_re
                )
                update_params!(model, pst1)
                l2, gs2 = withgradient(ps) do
                    loss(model(DATA[1]), DATA[2])
                end
                gnorm2 = norm(reduce(vcat, [vec(gs2[p]) for p ∈ ps]))
                pst2 = tuple(ps...)
                gs2t_ = tuple(gs2...) .* ETA
                # gs2t_re = (gs2t_ .+ conj.(gs2t_)) ./ 2
                gs2t_im = (gs2t_ .- conj.(gs2t_)) ./ 2
                update!(
                    GenDescent(1 / (1 + alpha * gnorm2)), pst2, gs2t_im
                )
                update_params!(model, pst2)
            end
            losses = [
                loss(model(datalist[di][1]), datalist[di][2]) for di ∈ 1:ndata
            ]
            for di ∈ 1:ndata
                push!(loss_lists[di], losses[di])
            end
            println("[$tl_i-$MODE] E: $epoch; L ($DI) = $(loss_lists[DI][end])")
            if !isnothing(subf)
                push!(subf_out, subf(model, loss, DATA, ps0))
            end
        end
        lbr('=')
    end
    if !isnothing(subf)
        return loss_lists, subf_out
    else
        return loss_lists
    end
end

function CovarianceMatrix(model, loss, data; k::Int, retsamp::Bool=false)
    Random.seed!(42)
    xdata, ydata = data
    ind = sample(1:size(xdata)[2], k, replace=false)
    xsample, ysample = xdata[:, ind], ydata[:, ind]
    sampledata = (xsample, ysample)
    cm = zeros(Complex, k, k)
    grads = [
        Flux.gradient(
            () -> loss(model(xsample[:, k_]), ysample[:, k_]),
            params(model)
        ) for k_ ∈ 1:k
    ]
    grad_vecs = [reduce(vcat, [
        vec(grads[k_][p]) for p in params(model) if grads[k_][p] !== nothing
    ]) for k_ ∈ 1:k]
    for i ∈ 1:k
        for j ∈ 1:k
            cm[i, j] += dot(
                grad_vecs[i], grad_vecs[j]
            ) / (
                norm(grad_vecs[i]) * norm(grad_vecs[j])
            )
        end
    end
    if retsamp
        return cm, sampledata
    else
        return cm
    end
end

function GradAnalysis(model, loss, data, ps0)
    ps = params(model)
    _, gs = withgradient(ps) do
        loss(model(data[1]), data[2])
    end
    wvec = vec(gs[ps[1]])
    bvec = vec(gs[ps[2]])
    gvec = reduce(vcat, [wvec, bvec])
    return [norm(wvec), norm(bvec), norm(gvec)]
end

function NSVecAnalysis(model, loss, data, ps0)
    ps = params(model)
    NSvec0 = reduce(vcat, [vec(p) for p in ps0 if p !== nothing])
    NSvec = reduce(vcat, [vec(p) for p in ps if p !== nothing])
    return norm(NSvec - NSvec0)
end

function LossLandscapeAnalysis01(model, loss, data, ps0)
    ps = params(model)
    _, gs = withgradient(ps) do
        loss(model(data[1]), data[2])
    end
    wvec = vec(gs[ps[1]])
    bvec = vec(gs[ps[2]])
    gvec = reduce(vcat, [wvec, bvec])
    NSvec0 = reduce(vcat, [vec(p) for p in ps0 if p !== nothing])
    NSvec = reduce(vcat, [vec(p) for p in ps if p !== nothing])
    return [norm(gvec), norm(NSvec - NSvec0)]
end



end