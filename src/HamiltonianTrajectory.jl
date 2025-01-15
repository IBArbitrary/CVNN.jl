include("CVNN.jl")
include("LossSurface.jl")
using .CVNN
using .LossSurface
LS = LossSurface

begin
    using Flux, Zygote, Optimisers
    using Flux: logitcrossentropy, crossentropy, onecold, onehotbatch, mse
    using Flux: params, train!, relu
    import Flux.Optimise: apply!, Descent, AbstractOptimiser
    using LinearAlgebra, Random, SparseArrays, MultivariateStats
    using StatsBase: sample, mean, svd
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

# IMPORTING MODEL DATA #########################################################

begin
    DATE2 = "140125"
    @load "./data/$DATE2/mpst_trained1.bson" mpst_trained1
    @load "./data/$DATE2/mpst_trained2.bson" mpst_trained2
    @load "./data/$DATE2/mpst_trained3.bson" mpst_trained3
    @load "./data/$DATE2/mpst_trained4.bson" mpst_trained4
    ns_vec_T1 = ParamTupleToNSVec(mpst_trained1)
    ns_vec_T2 = ParamTupleToNSVec(mpst_trained2)
    ns_vec_T3 = ParamTupleToNSVec(mpst_trained3)
    ns_vec_T4 = ParamTupleToNSVec(mpst_trained4)
    println(isindependent(ns_vec_T1, ns_vec_T2))
    println(isindependent(ns_vec_T3, ns_vec_T4))
    e1, e2 = GramSchmidtOrtho2(ns_vec_T1, ns_vec_T2)
    e3, e4 = GramSchmidtOrtho2(ns_vec_T3, ns_vec_T4)
    # n1 = NormalOfPlane(e1, e2)
    # n2 = NormalOfPlane(e3, e4)
end

# GENERATING LOSS SURFACES #####################################################

begin
    DATE = Dates.format(now(), "ddmmyy")
    mmodel = Chain(
        Dense(28^2 => 16, CVNN.zrelu; init=CVNN.complex_glorot_uniform),
        Dense(16 => 16, CVNN.zrelu; init=CVNN.complex_glorot_uniform),
        Dense(16 => 10, CVNN.abslu; init=CVNN.complex_glorot_uniform),
        softmax
    ) |> f64
    @save "./data/$DATE/mmodel-temp.bson" mmodel
end

begin
    s_range = -45.0:3:45.0
    t_range = -45.0:3:45.0
    loss_grid_O = zeros(length(s_range), length(t_range))
    loss_grid_E = zeros(length(s_range), length(t_range))
    loss_grid_F = zeros(length(s_range), length(t_range))
end

for (si, s_) ∈ enumerate(s_range)
    @showprogress for (ti, t_) ∈ enumerate(t_range)
        LS.NSVec2Params!(PlaneOriginNSVec(s_, t_, e1, e2), mmodel)
        loss_grid_O[si, ti] += crossentropy(mmodel(DATA_ODD[1]), DATA_ODD[2])
        loss_grid_E[si, ti] += crossentropy(mmodel(DATA_EVEN[1]), DATA_EVEN[2])
        loss_grid_F[si, ti] += crossentropy(mmodel(DATA[1]), DATA[2])
    end
end

loss_surface_1k = [
    "O" => loss_grid_O,
    "E" => loss_grid_E,
    "F" => loss_grid_F
]
@save "./data/$DATE/loss_surface_1k.bson" loss_surface_1k

################################################################################
LOAD = true

if LOAD
    s_range = -45.0:3:45.0
    t_range = -45.0:3:45.0
    @load "./data/$DATE/loss_surface_1k.bson" loss_surface_1k
    loss_grid_O = loss_surface_1k[1].second
    loss_grid_E = loss_surface_1k[2].second
    loss_grid_F = loss_surface_1k[3].second
end

begin
    st1 = PlaneProjection(ns_vec_T1, e1, e2)
    st2 = PlaneProjection(ns_vec_T2, e1, e2)
    st3 = PlaneProjection(ns_vec_T3, e3, e4)
    st4 = PlaneProjection(ns_vec_T4, e3, e4)
end

Plot1Heatmap1Scatter(
    s_range, t_range, loss_grid_O, [st1, st2],
    :inferno, true, "s (e1 coordinate)", "t (e2 coordinate)",
    "Loss Surface for odd data (MinSet 1)", (500, 400)
)

Plot1Surface(
    s_range, t_range, loss_grid_O, :inferno,
    NoShading, "s (e1 coordinate)", "t (e2 coordinate)",
    "Loss", "Loss surface of Data 1", (500, 500)
)

# TRAINING TRAJECTORY ANALYSIS #################################################

function PlaneProjectModel(model, e1, e2)
    mpst = deepcopy(params(model))
    ns_vec = LS.ParamTupleToNSVec(mpst)
    return PlaneProjection(ns_vec, e1, e2)
end

PlaneProjectModelSubf(model, loss, data, ps0) = PlaneProjectModel(model, e1, e2)

function NSVecSubf(model, loss, data, ps0)
    return LS.ParamTupleToNSVec(tuple(params(model)...))
end

begin
    mtrain_cfg_ = [
        50 => (0.06, 1),
        250 => (0.02im, 1, "semi-implicit-euler"),
        # 400 => (0.05im, 1, "normed-semi-implicit-euler", 0.2),
    ]
    L = 1
    mtrain_cfg = repeat(mtrain_cfg_, L)
end

begin
    DATE = Dates.format(now(), "ddmmyy")
    mmodel = Chain(
        Dense(28^2 => 16, CVNN.zrelu; init=CVNN.complex_glorot_uniform),
        Dense(16 => 16, CVNN.zrelu; init=CVNN.complex_glorot_uniform),
        Dense(16 => 10, CVNN.abslu; init=CVNN.complex_glorot_uniform),
        softmax
    ) |> f64
    mps = params(mmodel)
    # LS.NSVec2Params!(PlaneOriginNSVec(25, 25, e1, e2), mmodel)
    mpst_01 = deepcopy(tuple(mps...))
    @save "./data/$DATE/mmodel-01.bson" mmodel
end

mldata, msubfout = CVNN.SwitchTrainerSymplectic(
    mmodel, crossentropy, [DATA_ODD, DATA_EVEN], mtrain_cfg;
    subf=NSVecSubf
)

# @save "./data/$DATE/mmodel-04-trained.bson" mmodel
# mpst_trained4 = deepcopy(tuple(params(mmodel)...))
# @save "./data/$DATE/mpst_trained4.bson" mpst_trained4

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

stcoords = [LossSurface.PlaneProjection(vec, e1, e2) for vec ∈ msubfout]

LS.Plot1Heatmap1ScatterTraj(
    s_range, t_range, loss_grid_O, [st1, st2],
    stcoords, :inferno, :white, true, "s (e1 coordinate)",
    "t (e2 coordinate)", "Loss surface for Data 1 and a training trajectory",
    (500, 500);
    limits=(0.16, 0.26, -0.28, -0.20)
);

# PCA FOR PROJECTION ###########################################################

# msubfout is vector of network space VECTORS
@save "./data/$DATE/Grad+HamTrainingNSVecs03.bson" msubfout

M_h = transpose((reduce(hcat, tuple(msubfout[51:end]...))))
M_g = transpose((reduce(hcat, tuple(msubfout[1:50]...))))

function PCA_N(X, N)
    avg = mean(X, dims=1)
    center = X .- avg
    U, S, V = svd(center, full=false)
    pcs = V[:, 1:N]
    stds = S[1:N]
    return stds .^ 2, pcs
end

vars_h, pcs_h = PCA_N(M_h, 2)
vars_g, pcs_g = PCA_N(M_g, 2)
pc1_h, pc2_h = pcs_h[:, 1], pcs_h[:, 2]
pc1_g, pc2_g = pcs_g[:, 1], pcs_g[:, 2]

stcoords_g = [PlaneProjection(vec, pc1_g, pc2_g) for vec ∈ msubfout[1:50]]
stcoords_h = [PlaneProjection(vec, pc1_h, pc2_h) for vec ∈ msubfout[51:end]]

begin
    f = Figure(size=(500, 500))
    ax = Axis(f[1, 1], aspect=1, xlabel="e1", ylabel="e2", title="grad+ham")
    scatter!(
        ax, stcoords[1:50], color="red",
        markersize=range(2, 16, length=length(stcoords[1:50]))
    )
    scatter!(
        ax, stcoords[51:end], color="blue",
        markersize=range(2, 16, length=length(stcoords[51:end]))
    )
    display("image/png", f)
end

begin
    f = Figure(size=(500, 500))
    ax = Axis(f[1, 1], aspect=1, xlabel="pc1_g", ylabel="pc2_g", title="grad")
    scatter!(
        ax, stcoords_g, color="red",
        markersize=range(2, 16, length=length(stcoords_g))
    )
    display("image/png", f)
end

begin
    f = Figure(size=(500, 500))
    ax = Axis(f[1, 1], aspect=1, xlabel="pc1_h", ylabel="pc2_h", title="ham")
    scatter!(
        ax, stcoords_h, color="blue",
        markersize=range(2, 16, length=length(stcoords_h))
    )
    display("image/png", f)
end

# 1. find many minimizers and plot PROJECTION to see the distribution
# 2. generate surfaces for simpler models
# 3. modify trajector plot to make it better, remove colors, add size
# 4. genreate more refined surface image