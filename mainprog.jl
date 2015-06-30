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
    pdesc = ProblemDescription(0.5, [4.0], [Int16(64)], 0.05, Int8(4))
    sysdesc = generatesystemdescriptor(pdesc)

    ψ = StateSet(Complex128, 4, 64)

    T = discretizeOperators(kineticenergy, sysdesc, pdesc, false)
    V = discretizeOperators(harmonicpotential, sysdesc, pdesc, true)

    # convergence constants
    ϵ = pdesc.timestep
    χ = 0.5
    γ = pdesc.tolerence

    # second order factorization
    eV = exp(-0.5*ϵ*V)
    eVˈ = exp(-0.5*ϵ*χ*V)
    eT = exp(-1.0*ϵ*T)
    eTˈ = exp(-1.0*ϵ*χ*T)
    τ = eV*eT*eV
    τˈ = eVˈ*eTˈ*eVˈ

    multiply!(τ, ψ, sysdesc)
    sysdesc.normenergies = log(orthonormalize!(ψ))/(-2.0*ϵ)

    H = expectationvalue(ψ, V)
    momentumspace!(ψ, sysdesc)
    H += expectationvalue(ψ, T)
    positionspace!(ψ, sysdesc)
    δH = 0.0
    δE = 0.0

    for i=1:100
        ψˈ = deepcopy(ψ)

        multiply!(τ, ψ, sysdesc)
        E = log(orthonormalize!(ψ))/(-2.0*ϵ)
        rmsE = norm(E)/sqrt(pdesc.nstates)
        δE = norm(E - sysdesc.normenergies)/sqrt(pdesc.nstates)

        multiply!(τˈ,ψˈ, sysdesc)
        Eˈ = log(orthonormalize!(ψˈ))/(-2.0*ϵ*χ)
        rmsEˈ = norm(Eˈ)/sqrt(pdesc.nstates)
        δEˈ = norm(Eˈ - sysdesc.normenergies)/sqrt(pdesc.nstates)

        H = expectationvalue(ψ, V)
        momentumspace!(ψ, sysdesc)
        H += expectationvalue(ψ, T)
        positionspace!(ψ, sysdesc)
        δH = norm(E - H)/sqrt(pdesc.nstates)

        Hˈ = expectationvalue(ψˈ, V)
        momentumspace!(ψˈ, sysdesc)
        Hˈ += expectationvalue(ψˈ, T)
        positionspace!(ψˈ, sysdesc)
        δHˈ = norm(Eˈ - Hˈ)/sqrt(pdesc.nstates)

        if rmsE<rmsEˈ || δE<δEˈ
            χ = sqrt(χ)
            sysdesc.normenergies = E
            sysdesc.energies = H
        else
            ϵ *= χ
            χ = χ^2
            ψ = ψˈ
            sysdesc.normenergies = Eˈ
            sysdesc.energies = Hˈ
            δE = δEˈ
            δH = δHˈ
        end

        if δE <= γ*δH
            @show i
            break
        end

        eV = exp(-0.5*ϵ*V)
        eVˈ = exp(-0.5*ϵ*χ*V)
        eT = exp(-1.0*ϵ*T)
        eTˈ = exp(-1.0*ϵ*χ*T)
        τ = eV*eT*eV
        τˈ = eVˈ*eTˈ*eVˈ
    end

    @show abs(H)
    @show δE
    @show δH
end

@time solveschrodingereqn()
