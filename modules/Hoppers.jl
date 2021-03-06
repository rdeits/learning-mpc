__precompile__()

module Hoppers

using RigidBodyDynamics
using LCPSim

const urdf = joinpath(@__DIR__, "hopper.urdf")

struct Hopper{T}
    mechanism::Mechanism{T}
    environment::Environment{T}
end

function Hopper()
    mechanism = parse_urdf(Float64, urdf)
    world = root_body(mechanism)
    foot = findbody(mechanism, "foot")
    floor = planar_obstacle(default_frame(world), [0, 0, 1.], [0, 0, 0.], 2.0, :xz)
    env = Environment([
        (foot, Point3D(default_frame(foot), 0., 0, 0), floor)
    ])
    Hopper(mechanism, env)
end

function nominal_state(c::Hopper)
    x = MechanismState{Float64}(c.mechanism)
    set_configuration!(x, findjoint(c.mechanism, "base_z"), [2.0])
    set_configuration!(x, findjoint(c.mechanism, "foot_extension"), [0.75])
    x
end

function default_costs(c::Hopper)
    Q = diagm([10., 1e-1, 1e-3, 1])
    R = diagm([1e-3, 1e-3])
    Q, R
end

end
