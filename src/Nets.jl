__precompile__()

module Nets

export Net, Params, predict, predict_sensitivity

using Parameters: @with_kw
import ReverseDiff
import StochasticOptimization
using CoordinateTransformations: AffineMap, UniformScaling, transform_deriv
using MLDataPattern: batchview, shuffleobs

head(t::Tuple) = tuple(t[1])

function viewblocks{T <: NTuple}(data::AbstractArray, shapes::AbstractVector{T})
    starts = cumsum(vcat([1], prod.(shapes)))
    [reshape(view(data, starts[i]:(starts[i+1] - 1)), shapes[i]) for i in 1:length(shapes)]
end

struct Params{T, D <: AbstractVector{T}, M<:AbstractMatrix{T}, V<:AbstractVector{T}}
    data::D
    weights::Vector{M}
    biases::Vector{V}
    bypasses::Vector{M}
    shapes::Vector{NTuple{2, Int}}

    Params{T, D, M, V}(data::D, weights::Vector{M}, biases::Vector{V}, bypasses::Vector{M}) where {T, D, M, V} =
        new{T, D, M, V}(data, weights, biases, bypasses, size.(weights))
end

Params{T, D<:AbstractVector{T}, M<:AbstractMatrix{T}, V<:AbstractVector{T}}(
    data::D, weights::Vector{M}, biases::Vector{V}, bypasses::Vector{M}) =
    Params{T, D, M, V}(data, weights, biases, bypasses)

function Params(shapes::Vector{<:NTuple}, data::AbstractVector)
    weights = viewblocks(data, shapes)
    i = nweights(shapes)
    biases = viewblocks(@view(data[i + (1:nbiases(shapes))]), head.(shapes))
    i += nbiases(shapes)
    bypasses = viewblocks(@view(data[i + (1:nbypasses(shapes))]), bypass_shapes(shapes))

    Params(data, weights, biases, bypasses)
end

Params(widths::AbstractVector{<:Integer}, data::AbstractVector) =
    Params(collect(zip(widths[2:end], widths[1:end-1])), data)

ninputs(p::Params) = size(p.weights[1], 2)
noutputs(p::Params) = size(p.weights[end], 1)
shapes(p::Params) = p.shapes
nparams(widths::AbstractVector{<:Integer}) = nparams(collect(zip(widths[2:end], widths[1:end-1])))
nweights(shapes::AbstractVector{<:NTuple{2, Integer}}) = sum(prod, shapes)
nbiases(shapes::AbstractVector{<:NTuple{2, Integer}}) = sum(first, shapes)
nparams(shapes::AbstractVector{<:NTuple{2, Integer}}) = nweights(shapes) + nbiases(shapes) + nbypasses(shapes)
bypass_shapes(shapes::AbstractVector{<:NTuple{2, Integer}}) = [(s[1], shapes[1][2]) for s in shapes]
nbypasses(shapes::AbstractVector{<:NTuple{2, Integer}}) = sum(prod, bypass_shapes(shapes))

Base.zeros(::Type{Params{T}}, widths::AbstractVector{<:Integer}) where {T} =
    Params(widths, zeros(T, nparams(widths))) 
Base.rand(::Type{Params{T}}, widths::AbstractVector{<:Integer}) where {T} =
    Params(widths, rand(T, nparams(widths)))
Base.randn(::Type{Params{T}}, widths::AbstractVector{<:Integer}) where {T} =
    Params(widths, randn(T, nparams(widths)))

struct Net{P <: Params, T <: AffineMap} <: Function
    params::P
    input_tform::T
    output_tform::T
end

Net(params::Params) = 
    Net(params,
        AffineMap(UniformScaling(1.0), zeros(ninputs(params))),
        AffineMap(UniformScaling(1.0), zeros(noutputs(params))))

nweights(net::Net) = nweights(net.params)
nbiases(net::Net) = nbiases(net.params)
nparams(net::Net) = nparams(net.params)
ninputs(net::Net) = ninputs(net.params)
noutputs(net::Net) = noutputs(net.params)
params(net::Net) = net.params
(net::Net)(x) = predict(net, x)

Base.rand(net::Net) = rand(nparams(net))
Base.randn(net::Net) = randn(nparams(net))

Base.similar(net::Net, data::AbstractVector) =
    Net(Params(shapes(params(net)), data),
        net.input_tform,
        net.output_tform)

@ReverseDiff.forward leaky_relu(y) = y >= 0 ? y : 0.1 * y
@ReverseDiff.forward leaky_relu_sensitivity(y, j) = y >= 0 ? j : 0.1 * j

function predict(net::Net, x_::AbstractVector)
    x = net.input_tform(x_)
    params = net.params
    y = params.weights[1] * x + params.biases[1] + params.bypasses[1] * x
    for i in 2:length(params.weights)
        y = leaky_relu.(y)
        y = params.weights[i] * y + params.biases[i] + params.bypasses[i] * x
    end
    net.output_tform(y)
end

function predict_sensitivity(net::Net, x_::AbstractVector)
    x = net.input_tform(x_)
    params = net.params
    y = params.weights[1] * x + params.biases[1] + params.bypasses[1] * x
    J = params.weights[1] * transform_deriv(net.input_tform, x) + params.bypasses[1]
    for i in 2:length(params.weights)
        y = leaky_relu.(y)
        J = leaky_relu_sensitivity.(y, J)
        y = params.weights[i] * y + params.biases[i] + params.bypasses[i] * x
        J = params.weights[i] * J + params.bypasses[i]
    end
    hcat(net.output_tform(y), transform_deriv(net.output_tform, y) * J)
end


include("Explicit.jl")
include("optimization.jl")
include("scaling.jl")

end