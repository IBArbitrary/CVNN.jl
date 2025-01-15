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
    using CairoMakie, GLMakie, ProgressMeter
    using BSON: @save, @load
    using Printf: @sprintf
    using Dates
end

begin
    rng = MersenneTwister()
    Random.seed!(rng, 1337)
    DATE = Dates.format(now(), "ddmmyy")
end

# DATA IMPORT ##################################################################

# training data set
begin
    trainset = MNIST(:train)
    DSIZE = length(trainset)

    x_train = reduce(hcat, [vec(item.features) for item in trainset])
    # x_train .= x_train ./ 255.0
    x_train .= Float64.(x_train)
    y_train_raw = [item.targets for item in trainset]
    y_train = onehotbatch(y_train_raw, 0:9)

    DATA = (x_train, y_train)

    function filter_data(condition::Function)
        indices = filter(i -> condition(y_train_raw[i]), 1:DSIZE)
        x_filtered = x_train[:, indices]
        y_filtered = y_train[:, indices]
        return (x_filtered, y_filtered)
    end

    DATA_ODD = filter_data(isodd)
    DATA_EVEN = filter_data(iseven)
end

# testing data set
begin
    testset = MNIST(:test)
    TSIZE = length(testset)

    x_test = reduce(hcat, [vec(item.features) for item in testset])
    # x_test .= x_test ./ 255.0
    x_test .= Float64.(x_test)
    y_test_raw = [item.targets for item in testset]
    y_test = onehotbatch(y_test_raw, 0:9)

    DATA_T = (x_test, y_test)

    # Odd and even dataset creation
    function filter_test_data(condition::Function)
        indices = filter(i -> condition(y_test_raw[i]), 1:TSIZE)
        x_filtered = x_test[:, indices]
        y_filtered = y_test[:, indices]
        return (x_filtered, y_filtered)
    end

    TDATA_ODD = filter_test_data(isodd)
    TDATA_EVEN = filter_test_data(iseven)
end

# TRAINING TO GET MINIMIZER ####################################################

begin
    mtrain_cfg_ = [
        100 => (0.08, 2,),
        # 400 => (0.05im, 1, "normed-semi-implicit-euler", 0.2),
    ]
    L = 1
    mtrain_cfg = repeat(mtrain_cfg_, L)
end

begin
    DATE = Dates.format(now(), "ddmmyy")
    # DATE = "120125"
    mmodel = Chain(
        Dense(28^2 => 16, CVNN.zrelu; init=CVNN.complex_glorot_uniform),
        Dense(16 => 16, CVNN.zrelu; init=CVNN.complex_glorot_uniform),
        Dense(16 => 10, CVNN.abslu; init=CVNN.complex_glorot_uniform),
        softmax
    ) |> f64
    mloss(x, y) = crossentropy(mmodel(x), y)
    mps = params(mmodel)
    mpst_04 = deepcopy(tuple(mps...))
    @save "./data/$DATE/mmodel-04.bson" mmodel
end

mldata, msubfout = CVNN.SwitchTrainerSymplectic(
    mmodel, crossentropy, [DATA_ODD, DATA_EVEN], mtrain_cfg;
    subf=CVNN.LossLandscapeAnalysis01
)
@save "./data/$DATE/mmodel-04-trained.bson" mmodel
mpst_trained4 = deepcopy(tuple(params(mmodel)...))
@save "./data/$DATE/mpst_trained4.bson" mpst_trained4

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