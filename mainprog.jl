include("eigenspectrum.jl")

function harmonicpotential{T<:Real}(x::T)
    kx = 1.0
    kx * (x^2) / 2
end

function harmonicpotential{T<:Real}(x::T, y::T)
    kx = 1.0
    ky = 1.0
    ( kx * (x^2) / 2 ) + (ky * (y^2) / 2 )
end

function harmonicpotential{T<:Real}(x::T, y::T, z::T)
    kx = 1.0
    ky = 1.0
    kz = 1.0
    ( kx * (x^2) / 2 ) + (ky * (y^2) / 2 ) + (kz * (z^2) / 2 )
end

function kineticenergy{T<:Real}(px::T)
    m = 1.0
    px^2 / (2*m)
end

function kineticenergy{T<:Real}(px::T, py::T)
    m = 1.0
    (px^2 + py^2) / (2*m)
end

function kineticenergy{T<:Real}(px::T, py::T, pz::T)
    m = 1.0
    (px^2 + py^2 + pz^2) / (2*m)
end

function solveschrodingereqn()
    pdesc = ProblemDescription(0.05, [4.0], [Int16(64)], 0.4, Int8(4))
    sysdesc = generatesystemdescriptor(pdesc)

    s = StateSet(Complex128, 4, 64)

    T = discretizeOperators(kineticenergy, sysdesc, pdesc, false)
    V = discretizeOperators(harmonicpotential, sysdesc, pdesc, true)

    # second order factorization
    eV = exp(-0.5*pdesc.timestep*V)
    eT = exp(-1.0*pdesc.timestep*T)

    for i=1:100
        multiply!(eV, s, sysdesc)
        multiply!(eT, s, sysdesc)
        multiply!(eV, s, sysdesc)
        orthonormalize!(s)
    end

    energy = expectationvalue(s, V)
    momentumspace!(s, sysdesc)
    energy += expectationvalue(s, T)
    positionspace!(s, sysdesc)
    abs(energy)
end

@show solveschrodingereqn()
