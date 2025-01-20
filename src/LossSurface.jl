module LossSurface

export ParamTupleToNSVec, LerpNSVec
export NSVec2Params, NSVec2Params!
export isindependent, isindependent2, GramSchmidtOrtho2
export PlaneOriginNSVec, PlaneNSVec, PlaneProjection
export Plot1Heatmap, Plot1Heatmap1Scatter, Plot1Heatmap2Scatter
export Plot1HeatmapCRFixed, Plot1Surface
export NormalOfPlane, Space3DOriginNSVec, Space3DNSVec

include("CVNN.jl")
using .CVNN: update_params!
using Flux: params
using LinearAlgebra
using CairoMakie, GLMakie

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
    update_params!(model, tuple(reconstructed_params...))
end

function isindependent2(vec1, vec2)
    val1 = round(norm(vec1 - vec2), digits=4)
    val2 = round(dot(vec1, vec2), digits=4)
    val3 = rank(hcat(vec1, vec2))
    return [
        "distance" => val1,
        "dot" => val2,
        "rank" => val3
    ]
end

function isindependent(veclist)
    M = hcat(veclist...)
    N = size(M, 2)
    return rank(M) == N
end

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

function Plot1Heatmap(
    xrange, yrange, grid, cmap, interp,
    xlabel, ylabel, title, figsize;
)
    f = Figure(size=figsize)
    ax = Axis(
        f[1, 1], xlabel=xlabel, ylabel=ylabel,
        title=title, aspect=1
    )
    colorrange = (minimum(grid), maximum(grid))
    heatmap!(
        ax, xrange, yrange, grid, colorrange=colorrange,
        colormap=cmap, interpolate=interp
    )
    contour!(
        ax, xrange, yrange, grid,
        levels=5, color=:black
    )
    cb = Colorbar(f[:, 2]; colormap=cmap, colorrange=colorrange)
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
    cmap, interp, clevels,
    xlabel, ylabel, title, figsize
)
    f = Figure(size=figsize)
    ax = Axis(
        f[1, 1], xlabel=xlabel, ylabel=ylabel,
        title=title, aspect=1
    )
    colorrange = (minimum(grid), maximum(grid))
    heatmap!(
        ax, xrange, yrange, grid, colorrange=colorrange,
        colormap=cmap, interpolate=interp
    )
    contour!(
        ax, xrange, yrange, grid,
        levels=clevels, color=:white
    )
    scatter!(
        ax, scpts,
        color=:white, strokecolor=:black, strokewidth=1
    )
    cb = Colorbar(f[:, 2]; colormap=cmap, colorrange=colorrange)
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
    colorrange = (minimum(grid), maximum(grid))
    heatmap!(
        ax, xrange, yrange, grid, colorrange=colorrange,
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
    cb = Colorbar(f[:, 2]; colormap=cmap, colorrange=colorrange)
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

function Plot1Heatmap1ScatterTraj(
    xrange, yrange, grid, scpts,
    traj, cmap, trcolor, interp, clevel,
    xlabel, ylabel, title, figsize; limits=nothing
)
    # CairoMakie.activate!()
    f = Figure(size=figsize)
    if isnothing(limits)
        limits = (nothing, nothing)
    end
    ax = Axis(
        f[1, 1], xlabel=xlabel, ylabel=ylabel,
        title=title, aspect=1, limits=limits
    )
    colorrange = (minimum(grid), maximum(grid))
    heatmap!(
        ax, xrange, yrange, grid, colorrange=colorrange,
        colormap=cmap, interpolate=interp
    )
    contour!(
        ax, xrange, yrange, grid,
        levels=clevel, color=:white
    )
    scatter!(
        ax, scpts,
        color=:white, strokecolor=:black, strokewidth=1
    )
    scatter!(
        ax, traj, color=trcolor,
        markersize=range(2, 16, length=length(traj))
    )
    cb = Colorbar(f[:, 2]; colormap=cmap, colorrange=colorrange)
    display("image/png", f)
    return f
end

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

end