include("CVNN.jl")
using .CVNN

begin
    using Flux, Zygote, Optimisers
    using Flux: logitcrossentropy, crossentropy, onecold, onehotbatch, mse
    using Flux: params, train!, relu
    import Flux.Optimise: apply!, Descent, AbstractOptimiser
    using LinearAlgebra, Random, SparseArrays
    using StatsBase: sample
    using MLDatasets: MNIST
    using CairoMakie, ProgressMeter
    using BSON: @save, @load
    using Printf: @sprintf
end

trainset = MNIST(:train)

begin
    rng = MersenneTwister()
    Random.seed!(rng, 1337)
end

# training data set
begin
    trainset = MNIST(:train)
    DSIZE = length(trainset)

    x_train = reduce(hcat, [vec(item.features) for item in trainset])
    y_train_raw = [item.targets for item in trainset]
    y_train = onehotbatch(y_train_raw, 0:9)

    DATA = (x_train, y_train)

    # odd numbers
    x_train_o = stack([x_train[:, i] for i ∈ 1:DSIZE if isodd(y_train_raw[i])], dims=2)
    y_train_o = stack([y_train[:, i] for i ∈ 1:DSIZE if isodd(y_train_raw[i])], dims=2)

    # even numbers
    x_train_e = stack([x_train[:, i] for i ∈ 1:DSIZE if iseven(y_train_raw[i])], dims=2)
    y_train_e = stack([y_train[:, i] for i ∈ 1:DSIZE if iseven(y_train_raw[i])], dims=2)

    # data containers
    DATA_ODD = (x_train_o, y_train_o)
    DATA_EVEN = (x_train_e, y_train_e)
end

# testing data set
begin
    testset = MNIST(:test)
    TSIZE = length(testset)

    x_test = reduce(hcat, [vec(item.features) for item in testset])
    y_test_raw = [item.targets for item in testset]
    y_test = onehotbatch(y_test_raw, 0:9)

    DATA_T = (x_test, y_test)

    # odd numbers
    x_test_o = stack([x_test[:, i] for i ∈ 1:TSIZE if isodd(y_test_raw[i])], dims=2)
    x_test_e = stack([x_test[:, i] for i ∈ 1:TSIZE if iseven(y_test_raw[i])], dims=2)

    # even numbers
    y_test_o = stack([y_test[:, i] for i ∈ 1:TSIZE if isodd(y_test_raw[i])], dims=2)
    y_test_e = stack([y_test[:, i] for i ∈ 1:TSIZE if iseven(y_test_raw[i])], dims=2)

    # data containers
    TDATA_ODD = (x_test_o, y_test_o)
    TDATA_EVEN = (x_test_e, y_test_e)
end

# training

begin
    mtrain_cfg_ = [
        # 75 => (0.04, 1),
        200 => (0.008im, 1, "semi-implicit-euler"),
        # 100 => (0.1im, 1),
        # 100 => (-0.01, 1),
        # 50 => (0.05im, 1),
    ]
    L = 1
    mtrain_cfg = repeat(mtrain_cfg_, L)
end

begin
    mmodel = Chain(
        Dense(28^2 => 16, CVNN.zrelu; init=CVNN.complex_glorot_uniform),
        Dense(16 => 16, CVNN.zrelu; init=CVNN.complex_glorot_uniform),
        Dense(16 => 10, CVNN.abslu; init=CVNN.complex_glorot_uniform),
        softmax
    )
    mloss(x, y) = crossentropy(mmodel(x), y)
    mps = params(mmodel)
    @save "./data/mmodel.bson" mmodel
end

mpst = tuple(mps...)
ns_vec = reduce(vcat, vec.(tuple(params(mmodel)...)))
shapes = [size(p) for p in tuple(params(mmodel)...)]
start_idx = 1
reconstructed_params = []
for s in shapes
    len = prod(s)  # Number of elements in the current parameter
    end_idx = start_idx + len - 1
    param = reshape(ns_vec[start_idx:end_idx], s)  # Slice and reshape
    push!(reconstructed_params, param)
    start_idx = end_idx + 1
end
tuple(reconstructed_params...) == mpst


mldata, msubfout = CVNN.SwitchTrainerSymplectic(
    mmodel, crossentropy, [DATA_ODD, DATA_EVEN], mtrain_cfg;
    subf=CVNN.LossLandscapeAnalysis01
)

begin
    mf = Figure(size=(1000, 400))
    axm = Axis(
        mf[1, 1], xlabel="Epoch", ylabel="Loss",
        title="Training of MNIST data"
    )
    switches = cumsum([p.first for p in mtrain_cfg])
    switches_01 = [
        switches[i] for i in 1:length(switches) if mtrain_cfg[i].second[2] == 1
    ]
    if !isempty(switches_01)
        vlines!(switches_01, color="black", linestyle=:dash)
    end
    switches_02 = [
        switches[i] for i in 1:length(switches) if mtrain_cfg[i].second[2] == 2
    ]
    if !isempty(switches_02)
        vlines!(switches_02, color="black", linestyle=:dot)
    end
    lines!(mf[1, 1], mldata[1], label="loss(Data1)", color="red")
    lines!(mf[1, 1], mldata[2], label="loss(Data2)", color="blue")
    mf[1, 2] = Legend(mf, axm, "", framevisible=false)
    display("image/png", mf)
end

mgradnorm, mnsvecnorm = eachcol(vcat(msubfout'...))

begin
    f = Figure(size=(1000, 400))
    ax = Axis(
        f[1, 1], xlabel="Epoch", ylabel="Norm[Grad]",
        title="Evolution of gradient vector (MNIST data)"
    )
    vlines!(
        cumsum([p.first for p in mtrain_cfg]), color="black", linestyle=:dot
    )
    lines!(f[1, 1], mgradnorm, label="NSvec distance", color="Black")
    # f[1, 2] = Legend(f, ax, "", framevisible=false)
    display("image/png", f)
end

begin
    f = Figure(size=(1000, 400))
    ax = Axis(
        f[1, 1], xlabel="Epoch", ylabel="Norm[NSvec]",
        title="Evolution of Network Space (NS) vector (MNIST data)"
    )
    vlines!(
        cumsum([p.first for p in mtrain_cfg]), color="black", linestyle=:dot
    )
    lines!(f[1, 1], mnsvecnorm, label="NSvec distance", color="Black")
    # f[1, 2] = Legend(f, ax, "", framevisible=false)
    display("image/png", f)
end