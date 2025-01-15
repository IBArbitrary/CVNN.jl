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

# mpst_ = deepcopy(mpst_trained1)
# @save "./data/$DATE/mpst_.bson" mpst_

# NSVEC TO MODEL CONVERSION METHODS ############################################

ParamTupleToNSVec(mpst) = reduce(vcat, vec.(mpst))

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

function isindependent(vec1, vec2)
    val1 = round(norm(vec1 - vec2), digits=4)
    val2 = round(dot(vec1, vec2), digits=4)
    val3 = rank(hcat(vec1, vec2))
    return [
        "distance" => val1,
        "dot" => val2,
        "rank" => val3
    ]
end

# NS VECTORS CALCULATION #######################################################
begin
    ns_vec_01 = ParamTupleToNSVec(mpst_01)
    ns_vec_02 = ParamTupleToNSVec(mpst_02)
    ns_vec_03 = ParamTupleToNSVec(mpst_03)
    ns_vec_04 = ParamTupleToNSVec(mpst_04)
    ns_vec_T1 = ParamTupleToNSVec(mpst_trained1)
    ns_vec_T2 = ParamTupleToNSVec(mpst_trained2)
    ns_vec_T3 = ParamTupleToNSVec(mpst_trained3)
    ns_vec_T4 = ParamTupleToNSVec(mpst_trained4)

    isindependent(ns_vec_01, ns_vec_02)
    isindependent(ns_vec_T1, ns_vec_T2)
    isindependent(ns_vec_03, ns_vec_04)
    isindependent(ns_vec_T3, ns_vec_T4)
end

# LERP LOSS LANDSCAPE ANALYSIS #################################################
dα = 0.005
alphas = 0:dα:1
losses_alpha = []

@showprogress for alpha ∈ alphas
    NSVec2Params!(LerpNSVec(alpha, ns_vec_T2, ns_vec_T1), mmodel)
    push!(losses_alpha, crossentropy(mmodel(DATA_ODD[1]), DATA_ODD[2]))
end

begin
    f = Figure(size=(1000, 400))
    ax = Axis(
        f[1, 1], xlabel="Interpolation factor α", ylabel="Loss",
        title="1D Loss Surface (even)"
    )
    lines!(f[1, 1], losses_alpha, label="NSvec distance", color="Black")
    display("image/png", f)
end

# 2DLERP LOSS LANDSCAPE ANALYSIS ###############################################

function GramSchmidtOrtho2(v1, v2)
    e1 = v1
    proj = (dot(e1, v2) / dot(e1, e1)) * e1
    e2 = v2 - proj
    e1 /= norm(e1)
    e2 /= norm(e2)
    return e1, e2
end

PlaneOriginNSVec(s, t, e1, e2) = s * e1 + t * e2

PlaneNSVec(s, t, e1, e2, p0) = p0 + PlaneOriginNSVec(s, t, e1, e2)

function PlaneProjection(v, e1, e2)
    e1 /= norm(e1)
    e2 /= norm(e2)
    s = real(dot(v, e1))
    t = real(dot(v, e2))
    return (s, t)
end

begin
    e1, e2 = GramSchmidtOrtho2(ns_vec_T1, ns_vec_T2)
    e3, e4 = GramSchmidtOrtho2(ns_vec_T3, ns_vec_T4)
end

begin
    s_range = -45.0:1.75:45.0
    t_range = -45.0:1.75:45.0
    loss_grid_O = zeros(length(s_range), length(t_range))
    loss_grid_E = zeros(length(s_range), length(t_range))
    loss_grid_F = zeros(length(s_range), length(t_range))
    loss_grid_O2 = zeros(length(s_range), length(t_range))
    loss_grid_E2 = zeros(length(s_range), length(t_range))
    loss_grid_F2 = zeros(length(s_range), length(t_range))
end

@showprogress for (si, s_) ∈ enumerate(s_range)
    for (ti, t_) ∈ enumerate(t_range)
        NSVec2Params!(PlaneOriginNSVec(s_, t_, e1, e2), mmodel)
        loss_grid_O[si, ti] += crossentropy(mmodel(DATA_ODD[1]), DATA_ODD[2])
        loss_grid_E[si, ti] += crossentropy(mmodel(DATA_EVEN[1]), DATA_EVEN[2])
        loss_grid_F[si, ti] += crossentropy(mmodel(DATA[1]), DATA[2])
        NSVec2Params!(PlaneOriginNSVec(s_, t_, e3, e4), mmodel)
        loss_grid_O2[si, ti] += crossentropy(mmodel(DATA_ODD[1]), DATA_ODD[2])
        loss_grid_E2[si, ti] += crossentropy(mmodel(DATA_EVEN[1]), DATA_EVEN[2])
        loss_grid_F2[si, ti] += crossentropy(mmodel(DATA[1]), DATA[2])
    end
end

begin
    both_training_results = [
        "LS_O" => loss_grid_O,
        "LS_E" => loss_grid_E,
        "LS_F" => loss_grid_F,
        "LS_O2" => loss_grid_O2,
        "LS_E2" => loss_grid_E2,
        "LS_F2" => loss_grid_F2,
        "e1" => e1,
        "e2" => e2,
        "e3" => e3,
        "e4" => e4,
    ]
    @save "./data/$DATE/training_results-01" odd_training_results
end

begin
    st1 = PlaneProjection(ns_vec_T1, e1, e2)
    st2 = PlaneProjection(ns_vec_T2, e1, e2)
    st3 = PlaneProjection(ns_vec_T3, e3, e4)
    st4 = PlaneProjection(ns_vec_T4, e3, e4)
    st31 = PlaneProjection(ns_vec_T3, e1, e2)
    st41 = PlaneProjection(ns_vec_T4, e1, e2)
    st12 = PlaneProjection(ns_vec_T1, e3, e4)
    st22 = PlaneProjection(ns_vec_T2, e3, e4)
end

# PLOT METHODS #################################################################

function Plot1Heatmap(
    xrange, yrange, grid, cmap, interp,
    xlabel, ylabel, title, figsize;
)
    f = Figure(size=figsize)
    ax = Axis(
        f[1, 1], xlabel=xlabel, ylabel=ylabel,
        title=title, aspect=1
    )
    heatmap!(
        ax, xrange, yrange, grid,
        colormap=cmap, interpolate=interp
    )
    contour!(
        ax, xrange, yrange, grid,
        levels=5, color=:black
    )
    cb = Colorbar(f[:, 2]; colormap=cmap)
    display("image/png", f)
    # return f
end

function Plot1HeatmapCRFixed(
    xrange, yrange, grid, cmap, interp, colorrange,
    xlabel, ylabel, title, figsize;
)
    f = Figure(size=figsize)
    ax = Axis(
        f[1, 1], xlabel=xlabel, ylabel=ylabel,
        title=title, aspect=1
    )
    heatmap!(
        ax, xrange, yrange, grid,
        colormap=cmap, interpolate=interp, colorrange=colorrange
    )
    contour!(
        ax, xrange, yrange, grid,
        levels=5, color=:black
    )
    cb = Colorbar(f[:, 2]; colormap=cmap, colorrange=colorrange)
    display("image/png", f)
    # return f
end

function Plot1Heatmap1Scatter(
    xrange, yrange, grid, scpts,
    cmap, interp,
    xlabel, ylabel, title, figsize
)
    f = Figure(size=figsize)
    ax = Axis(
        f[1, 1], xlabel=xlabel, ylabel=ylabel,
        title=title, aspect=1
    )
    heatmap!(
        ax, xrange, yrange, grid,
        colormap=cmap, interpolate=interp
    )
    contour!(
        ax, xrange, yrange, grid,
        levels=5, color=:black
    )
    scatter!(
        ax, scpts,
        color=:white, strokecolor=:black, strokewidth=1
    )
    cb = Colorbar(f[:, 2]; colormap=cmap)
    display("image/png", f)
    # return f
end

function Plot1Heatmap2Scatter(
    xrange, yrange, grid, scpts,
    cmap, interp,
    xlabel, ylabel, title, figsize
)
    # CairoMakie.activate!()
    f = Figure(size=figsize)
    ax = Axis(
        f[1, 1], xlabel=xlabel, ylabel=ylabel,
        title=title, aspect=1
    )
    heatmap!(
        ax, xrange, yrange, grid,
        colormap=cmap, interpolate=interp
    )
    contour!(
        ax, xrange, yrange, grid,
        levels=5, color=:black
    )
    scatter!(
        ax, scpts[1],
        color=:white, strokecolor=:black, strokewidth=1
    )
    scatter!(
        ax, scpts[2],
        color=:black, strokecolor=:white, strokewidth=1
    )
    cb = Colorbar(f[:, 2]; colormap=cmap)
    display("image/png", f)
    # return f
end

function Plot1Surface(
    xrange, yrange, grid,
    cmap, shading,
    xlabel, ylabel, zlabel, title, figsize
)
    GLMakie.activate!()
    f = Figure(size=(500, 400))
    ax = Axis3(
        f[1, 1], xlabel=xlabel, ylabel=ylabel, zlabel=zlabel,
        title=title, aspect=(1, 1, 1)
    )
    surface!(
        ax, xrange, yrange, grid,
        colormap=cmap, shading=shading
    )
    display(GLMakie.Screen(), f)
    return f
end

# FIGURES ######################################################################

Plot1Surface(
    s_range, t_range, loss_grid_F,
    :inferno, NoShading, "s (e1 coordinate)", "t (e2 coordinate)", "Loss",
    "Loss Surface for full data (MinSet 1)", (500, 400)
)

Plot1Heatmap2Scatter(
    s_range, t_range, loss_grid_F, ([st1, st2], [st31, st41]),
    :inferno, true, "s (e1 coordinate)", "t (e2 coordinate)",
    "Loss Surface for full data (MinSet 1)", (500, 400)
)

Plot1Heatmap2Scatter(
    s_range, t_range, loss_grid_O, ([st1, st2], [st31, st41]),
    :inferno, true, "s (e1 coordinate)", "t (e2 coordinate)",
    "Loss Surface for odd data (MinSet 1)", (500, 400)
)

# 3D SLICES ####################################################################

function NormalOfPlane(e1, e2)
    B = hcat(e1, e2)
    n = nullspace(B')[:, end] # taking only first
    n = normalize(n)
    return n
end

Space3DOriginNSVec(s, t, u, e1, e2, n) = s * e1 + t * e2 + u * n

Space3DNSVec(s, t, u, e1, e2, n, p0) = p0 + Space3DOriginNSVec(
    s, t, u, e1, e2, n
)

begin
    mmodel = Chain(
        Dense(28^2 => 16, CVNN.zrelu; init=CVNN.complex_glorot_uniform),
        Dense(16 => 16, CVNN.zrelu; init=CVNN.complex_glorot_uniform),
        Dense(16 => 10, CVNN.abslu; init=CVNN.complex_glorot_uniform),
        softmax
    ) |> f64
    @save "./data/$DATE/mmodel-temp.bson" mmodel
end

begin
    @load "./data/$DATE/mpst_trained1.bson" mpst_trained1
    @load "./data/$DATE/mpst_trained2.bson" mpst_trained2
    @load "./data/$DATE/mpst_trained3.bson" mpst_trained3
    @load "./data/$DATE/mpst_trained4.bson" mpst_trained4
    ns_vec_T1 = ParamTupleToNSVec(mpst_trained1)
    ns_vec_T2 = ParamTupleToNSVec(mpst_trained2)
    ns_vec_T3 = ParamTupleToNSVec(mpst_trained3)
    ns_vec_T4 = ParamTupleToNSVec(mpst_trained4)
    println(isindependent(ns_vec_T1, ns_vec_T2))
    println(isindependent(ns_vec_T3, ns_vec_T4))
    e1, e2 = GramSchmidtOrtho2(ns_vec_T1, ns_vec_T2)
    e3, e4 = GramSchmidtOrtho2(ns_vec_T3, ns_vec_T4)
    n1 = NormalOfPlane(e1, e2)
    n2 = NormalOfPlane(e3, e4)
end

begin
    s_range = -30.0:2:30.0
    t_range = -30.0:2:30.0
    u_range = -80.0:8:80.0
    loss_volgrid_O = zeros(length(s_range), length(t_range), length(u_range))
end

@showprogress for (s, s_) ∈ enumerate(s_range)
    for (t, t_) ∈ enumerate(t_range)
        for (u, u_) ∈ enumerate(u_range)
            NSVec2Params!(Space3DOriginNSVec(s_, t_, u_, e1, e2, n1), mmodel)
            loss_volgrid_O[s, t, u] += crossentropy(
                mmodel(DATA_ODD[1]), DATA_ODD[2]
            )
        end
    end
end
@save "./data/$DATE/loss_volgrid_O_02.bson" loss_volgrid_O

crange = (minimum(loss_volgrid_O), maximum(loss_volgrid_O))

Plot1HeatmapCRFixed(
    s_range, t_range, loss_volgrid_O[:, :, 1], :inferno, true,
    crange, "s (e1 coordinate)", "t (e2 coordinate)",
    "Loss Surface for full data (MinSet 1)", (500, 400)
)

n_ = NormalOfPlane(e1, e2)

norm(Space3DOriginNSVec(10, 10, 10, e1, e2, n1) - PlaneOriginNSVec(10, 10, e1, e2))

dot(n_, n1)

NSVec2Params!(Space3DOriginNSVec(10, 10, 80, e1, e2, n_), mmodel)
NSVec2Params!(PlaneOriginNSVec(10, 10, e1, e2), mmodel)
crossentropy(mmodel(DATA_ODD[1]), DATA_ODD[2])

norm(loss_volgrid_O[:, :, 1] - loss_volgrid_O[:, :, 21])

norm(PlaneOriginNSVec(10, 10, e1, e2) - PlaneOriginNSVec(10, 100, e1, e2))