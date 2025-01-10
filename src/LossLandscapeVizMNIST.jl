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
end

begin
    rng = MersenneTwister()
    Random.seed!(rng, 1337)
end

# DATA IMPORT ##################################################################

trainset = MNIST(:train)

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

# TRAINING TO GET MINIMIZER ####################################################

begin
    mtrain_cfg_ = [
        200 => (0.02, 1),
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
    mpst_02 = deepcopy(tuple(mps...))
    @save "./data/mmodel2.bson" mmodel
end

mldata, msubfout = CVNN.SwitchTrainerSymplectic(
    mmodel, crossentropy, [DATA_ODD, DATA_EVEN], mtrain_cfg;
    subf=CVNN.LossLandscapeAnalysis01
)
@save "./data/mmodel-trained2.bson" mmodel
mpst_trained2 = deepcopy(tuple(params(mmodel)...))

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

# 1D LINEAR INTERPOLATION METHOD ###############################################

LerpNSVec(α, NSVec1, NSVec2) = (1 - α) .* NSVec1 + α .* NSVec2

function NSVec2Params(nsvec, model)
    shapes = [size(p) for p ∈ tuple(params(model)...)]
    start_idx = 1
    reconstructed_params = []
    for s in shapes
        len = prod(s)
        end_idx = start_idx + len - 1
        param = reshape(nsvec[start_idx:end_idx], s)
        push!(reconstructed_params, param)
        start_idx = end_idx + 1
    end
    return tuple(reconstructed_params...)
end

function NSVec2Params!(nsvec, model)
    shapes = [size(p) for p ∈ tuple(params(model)...)]
    start_idx = 1
    reconstructed_params = []
    for s in shapes
        len = prod(s)
        end_idx = start_idx + len - 1
        param = reshape(nsvec[start_idx:end_idx], s)
        push!(reconstructed_params, param)
        start_idx = end_idx + 1
    end
    CVNN.update_params!(model, tuple(reconstructed_params...))
end

# LERP LOSS LANDSCAPE ANALYSIS #################################################

ns_vec_0 = reduce(vcat, vec.(mpst_0))
ns_vec_T1 = reduce(vcat, vec.(mpst_trained1))
ns_vec_T2 = reduce(vcat, vec.(mpst_trained2))

dα = 0.005
alphas = 0:dα:1
losses_alpha = []

norm(ns_vec_T2 - ns_vec_0)

@showprogress for alpha ∈ alphas
    NSVec2Params!(LerpNSVec(alpha, ns_vec_T2, ns_vec_T1), mmodel)
    push!(losses_alpha, crossentropy(mmodel(DATA_ODD[1]), DATA_ODD[2]))
end

begin
    f = Figure(size=(1000, 400))
    ax = Axis(
        f[1, 1], xlabel="Interpolation factor α", ylabel="Loss",
        title="1D Loss Surface"
    )
    lines!(f[1, 1], losses_alpha, label="NSvec distance", color="Black")
    # f[1, 2] = Legend(f, ax, "", framevisible=false)
    display("image/png", f)
end

# 2DLERP LOSS LANDSCAPE ANALYSIS ###############################################

# use chatgpt code to write the random directions code
# then write loss calculation in a grid defined by the directions f(alpha, beta)

# ensuring the two minimizer NS vectors are linearly independent
round(norm(ns_vec_T1 - ns_vec_T2), digits=3) != 0.0
dot(ns_vec_T1, ns_vec_T2) != 0.0
rank(hcat(ns_vec_T1, ns_vec_T2)) == 2


# Generating the unit vectors
e1 = ns_vec_T1
proj2 = (dot(e1, ns_vec_T2) / dot(e1, e1)) * e1
e2 = ns_vec_T2 - proj2

e12 = ns_vec_T1 / norm(ns_vec_T1)
e22 = ns_vec_T2 / norm(ns_vec_T2)



# e1 and e2 are the two unit vectors

PlaneOriginNSVec(s, t, e1, e2) = s * e1 + t * e2
PlaneNSVec(s, t, e1, e2, p0) = p0 + PlaneOriginNSVec(s, t, e1, e2)

function PlaneProjection(v, e1, e2)
    basis_matrix = hcat(e1, e2)
    st = real.(basis_matrix \ v)
    return st[1], st[2]
end

function PlaneProjection2(v, e1, e2)
    e1 /= norm(e1)
    e2 /= norm(e2)
    s = real(conj(e1)' * v)
    t = real(conj(e2)' * v)
    return (s, t)
end

s_range = -5.0:0.2:5.0
t_range = -5.0:0.2:5.0

loss_grid_O = zeros(length(s_range), length(t_range))
loss_grid_E = zeros(length(s_range), length(t_range))
loss_grid_F = zeros(length(s_range), length(t_range))

@showprogress for (si, s_) ∈ enumerate(s_range)
    for (ti, t_) ∈ enumerate(t_range)
        NSVec2Params!(PlaneOriginNSVec(s_, t_, e1, e2), mmodel)
        loss_grid_O[si, ti] += crossentropy(mmodel(DATA_ODD[1]), DATA_ODD[2])
        loss_grid_E[si, ti] += crossentropy(mmodel(DATA_EVEN[1]), DATA_EVEN[2])
        loss_grid_F[si, ti] += crossentropy(mmodel(DATA[1]), DATA[2])
    end
end

st1 = real.(PlaneProjection(ns_vec_T1, e1, e2))
st2 = real.(PlaneProjection(ns_vec_T2, e1, e2))

begin
    # CairoMakie.activate!()
    f = Figure(size=(500, 400))
    ax = Axis(
        f[1, 1], xlabel="s (e1 coordinate)", ylabel="t (e2 coordinate)",
        title="Loss (Sub)Surface", aspect=1 # Ensures the heatmap has square cells
    )
    heatmap!(
        ax, s_range, t_range, loss_grid,
        colormap=:inferno, interpolate=true
    )
    cb = Colorbar(f[:, 2]; colormap=:inferno)
    display("image/png", f)
end

begin
    # CairoMakie.activate!()
    f = Figure(size=(500, 400))
    ax = Axis(
        f[1, 1], xlabel="s (e1 coordinate)", ylabel="t (e2 coordinate)",
        title="Loss Surface for Data1", aspect=1 # Ensures the heatmap has square cells
    )
    heatmap!(
        ax, s_range, t_range, loss_grid_O,
        colormap=:inferno, interpolate=true
    )
    contour!(
        ax, s_range, t_range, loss_grid_O,
        levels=5, color=:black
    )
    scatter!(
        ax, [st1, st2],
        color=:white, strokecolor=:black, strokewidth=1
    )
    cb = Colorbar(f[:, 2]; colormap=:inferno)
    display("image/png", f)
end

begin
    # CairoMakie.activate!()
    f = Figure(size=(500, 400))
    ax = Axis(
        f[1, 1], xlabel="s (e1 coordinate)", ylabel="t (e2 coordinate)",
        title="Loss Surface for Data2", aspect=1 # Ensures the heatmap has square cells
    )
    heatmap!(
        ax, s_range, t_range, loss_grid_E,
        colormap=:inferno, interpolate=true
    )
    contour!(
        ax, s_range, t_range, loss_grid_E,
        levels=5, color=:black
    )
    scatter!(
        ax, [st1, st2],
        color=:white, strokecolor=:black, strokewidth=1
    )
    cb = Colorbar(f[:, 2]; colormap=:inferno)
    display("image/png", f)
end

begin
    # CairoMakie.activate!()
    f = Figure(size=(500, 400))
    ax = Axis(
        f[1, 1], xlabel="s (e1 coordinate)", ylabel="t (e2 coordinate)",
        title="Loss Surface for full data", aspect=1 # Ensures the heatmap has square cells
    )
    heatmap!(
        ax, s_range, t_range, loss_grid_F,
        colormap=:inferno, interpolate=true
    )
    contour!(
        ax, s_range, t_range, loss_grid_F,
        levels=5, color=:black
    )
    scatter!(
        ax, [st1, st2],
        color=:white, strokecolor=:black, strokewidth=1
    )
    cb = Colorbar(f[:, 2]; colormap=:inferno)
    display("image/png", f)
end

begin
    GLMakie.activate!()
    f = Figure(size=(500, 400))
    ax = Axis3(
        f[1, 1], xlabel="s (e1 coordinate)", ylabel="t (e2 coordinate)", zlabel="Loss",
        title="Loss Surface for Data1", aspect=(1, 1, 1)
    )
    surface!(
        ax, s_range, t_range, loss_grid,
        colormap=:inferno, shading=NoShading
    )
    # display("image/png", f)
    f
end

begin
    GLMakie.activate!()
    f = Figure(size=(500, 400))
    ax = Axis3(
        f[1, 1], xlabel="s (e1 coordinate)", ylabel="t (e2 coordinate)", zlabel="Loss",
        title="Loss Surface for Data2", aspect=(1, 1, 1)
    )
    surface!(
        ax, s_range, t_range, loss_grid_E,
        colormap=:inferno, shading=NoShading
    )
    # display("image/png", f)
    f
end

# MODEL EVOLUTION ANALYSIS #####################################################

begin
    mtrain_cfg_ = [
        200 => (0.02, 1),
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
    mpst_02 = deepcopy(tuple(mps...))
    @save "./data/mmodel2.bson" mmodel
end

mldata, msubfout = CVNN.SwitchTrainerSymplectic(
    mmodel, crossentropy, [DATA_ODD, DATA_EVEN], mtrain_cfg;
    subf=CVNN.LossLandscapeAnalysis01
)
@save "./data/mmodel-trained2.bson" mmodel
mpst_trained2 = deepcopy(tuple(params(mmodel)...))