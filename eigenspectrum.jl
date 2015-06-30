using Iterators

# type definitions
immutable ProblemDescription{T<:FloatingPoint}
    timestep::T

    # organized in the order of x-y-z
    boundary::Array{T,1}
    discretesteps::Array{Int16,1}

    tolerence::T

    nstates::Int8
end

type SystemDescription{T<:FloatingPoint}
    cres::Array{T, 1}
    coordindex::Iterators.Product
    pcoordinategrid::Array{Array{T,1}, 1}
    mcoordinategrid::Array{Array{T,1}, 1}

    fftplan!::Function
    ifftplan!::Function

    normenergies::Array{Complex{T},1}
    energies::Array{Complex{T}, 1}

    fftnorm::Complex{T}
end

type StateZero
end

type State{T<:Complex, N}
    components::Array{T, N}
    inpositionbasis::Bool
    frozen::Bool
end

typealias State1D{T<:Complex} State{T, 1}
typealias State2D{T<:Complex} State{T, 2}
typealias State3D{T<:Complex} State{T, 3}

type StateSet{T<:Complex, N}
    states::Array{State{T,N},1}
end

typealias StateSet1D{T<:Complex} StateSet{T, 1}
typealias StateSet2D{T<:Complex} StateSet{T, 2}
typealias StateSet3D{T<:Complex} StateSet{T, 3}

immutable QuantumOperator{T<:Complex, N}
    doperator::Array{T, N}
    positionbasis::Bool
end

immutable CompositeQuantumOperator{T<:Complex, N}
    queue::Array{QuantumOperator{T,N},1}
end

QOperator = Union(QuantumOperator, CompositeQuantumOperator)
### Outer constructors
State{T<:Complex}(::Type{T},vars...) = State(rand(T, vars...), true, false)

StateSet{T<:Complex}(::Type{T}, number, vars...) =
                                    StateSet([State(T,vars...) for i=1:number])
StateSet{T<:Complex,N}(x::Array{Array{T,N},1}) = StateSet([State(i, true, false) for i in x])

### Method definitions
Base.call{T<:Complex}(::Type{State1D}, ::Type{T}, a) = State{T,1}(a)
Base.call{T<:Complex}(::Type{State2D}, ::Type{T}, a, b) = State{T,2}(a, b)
Base.call{T<:Complex}(::Type{State3D}, ::Type{T}, a, b, c) = State{T,3}(a, b, c)
Base.call{T<:Complex}(::Type{StateSet1D}, ::Type{T}, n, a) =
                                                StateSet{T,1}(n, a)
Base.call{T<:Complex}(::Type{StateSet2D}, ::Type{T}, n, a, b) =
                                                StateSet{T,2}(n, a, b)
Base.call{T<:Complex}(::Type{StateSet3D}, ::Type{T}, n, a, b, c) =
                                                StateSet{T,3}(n, a, b, c)


Base.call{T,N}(x::State{T,N}, y::Array{T,N}) = begin
    if(size(x)==size(y))
        x.components=y
    else
        error("Cannot change the dimensions of a state.")
    end
end
Base.call{T,N}(x::StateSet{T,N}, a::Array{Array{T,N},1}) = begin
    for i=1:length(x)
        x.states[i](a[i])
    end
end
Base.call{T,N}(x::StateSet{T,N}, a::Array{State{T,N},1}) = (x.states = a)

Base.getindex(x::State, vars...) = x.components[vars...]
Base.getindex(x::StateSet, n) = x.states[n]
Base.getindex(x::StateSet, n, vars...) = x.states[n].components[vars...]
Base.getindex(x::QuantumOperator, vars...) = x.doperator[vars...]

Base.size{T<:State}(x::T) = size(x.components)
Base.size{T<:StateSet}(x::T) = [length(x.states), size(x.states[1])...]

Base.length{T<:State}(x::T) = length(x.components)
Base.length{T<:StateSet}(x::T) = length(x.states)

Base.convert{T,N}(::Type{Array{T,N}}, x::State{T,N}) = x.components
Base.convert{T,N}(::Type{Array{State{T,N},1}}, x::StateSet{T,N}) = x.states
Base.convert{T,N}(::Type{Array{Array{T,N},1}}, x::StateSet{T,N}) =
                                            [i.components for i in x.states]
Base.convert{T,N}(::Type{StateSet{T,N}}, x::Array{Any, 1}) = begin
    a::Array{State{T,N}, 1} = [i for i in x]
    StateSet(a)
end

Base.start{T<:StateSet}(x::T) = 1
Base.done{T<:StateSet}(x::T, s) = ((length(x)+1)==s)
Base.next{T<:StateSet}(x::T, s) = x[s], s+1


Base.zero(::Type(Any)) = StateZero()
### Function definitions

function generatesystemdescriptor{T<:FloatingPoint}(pdesc::ProblemDescription{T})
    cres = zeros(T, length(pdesc.boundary))
    cgrid = product([[i for i=1:pdesc.discretesteps[j]] for j=1:length(pdesc.discretesteps)])
    for i=1:length(pdesc.boundary)
        cres[i] = 2*pdesc.boundary[i]/pdesc.discretesteps[i]
    end
    csys = [[-pdesc.boundary[j]+(i-1)*cres[j] for i=1:(pdesc.discretesteps[j])] for j=1:length(pdesc.discretesteps)]
    kcoord = [[(i<=(pdesc.discretesteps[j]/2)?(i-1):(i-pdesc.discretesteps[j]-1))*π/pdesc.boundary[j] for i=1:pdesc.discretesteps[j]] for j=1:length(pdesc.discretesteps)]

    fplan = plan_fft!(rand(Complex{T},pdesc.discretesteps...))
    ifplan = plan_bfft!(rand(Complex{T},pdesc.discretesteps...))
    nenergies = zeros(Complex{T},pdesc.nstates)
    energies = zeros(Complex{T},pdesc.nstates)

    csys1 = convert(Array{Array{T,1},1},[convert(Array{T,1},[p...]) for p in product(csys...)])
    kcoord1 = convert(Array{Array{T,1},1},[convert(Array{T,1},[p...]) for p in product(kcoord...)])

    SystemDescription(cres, cgrid, csys1, kcoord1,fplan, ifplan, nenergies, energies, convert(Complex{T},prod(pdesc.discretesteps)^0.5))
end

### too expensive to assert the same dimensions every single time?
function assertsamedimensions{T<:State}(x::T, y::T)
    size(x) == size(y) ? size(x): error("Incompatible States")
end

function discretizeOperators{T<:FloatingPoint}(operator::Function,
                    syd::SystemDescription{T}, pdesc::ProblemDescription{T},
                                positionbasis::Bool=true)
    if positionbasis==true
        grid = syd.pcoordinategrid
    else
        grid = syd.mcoordinategrid
    end

    t = [convert(Complex{T}, operator(p...)) for p in grid]
    QuantumOperator(convert(Array{Complex{T}},reshape(t, convert(Array{Int64,1},pdesc.discretesteps)...)), positionbasis)
end

(+){T<:State}(a::StateZero, b::T) = b
(+){T<:State}(a::T, b::T) = begin
    if a.inpositionbasis==b.inpositionbasis
        c = deepcopy(a)
        c.components = a.components + b.components
        c
    else
        error("States represented in two different bases cannot be added.")
    end
end
(+){T<:StateSet}(a::T, b::T) = begin
    convert(T, map(+, a, b))
end

(+){T<:QuantumOperator}(a::T, b::T) = begin
    if a.positionbasis == b.positionbasis
        QuantumOperator(a.doperator+b.doperator, a.positionbasis)
    else
        error("Two operators are expressed in different basis.")
    end
end

(-){T<:State}(a::T, b::T) = a + (-1*b)
(-){T<:StateSet}(a::T, b::T) = a + (-1*b)
(-){T<:QuantumOperator}(a::T, b::T) = a + (-1*b)

(*){T<:Complex,Z<:Number,N}(a::Z, b::State{T,N}) = begin
    c = deepcopy(b)
    c.components = convert(T,a) * b.components
    c
end
(*){T<:Complex,Z<:Number,N}(a::Z, b::StateSet{T,N}) = begin
    c::Array{State{T,N},1} = [a*i for i in b]
    StateSet(c)
end
(*){T<:Complex,N,Z<:Number}(a::Array{Z,2}, b::StateSet{T,N}) = begin
    convert(StateSet{T,N}, a*b.states)
end

(*){T<:Complex,N}(O::QuantumOperator{T, N}, s::State{T,N}) = begin
    s1 = deepcopy(s)
    multiply!(O, s1)
    s1
end

(*){T<:Complex,N}(O::QuantumOperator{T,N}, s::StateSet{T,N}) = begin
    s1 = deepcopy(s)
    multiply!(O, s1)
    s1
end

(*){T<:State}(s1::T, s2::T) = innerproduct(s1, s2)

(*){T<:QuantumOperator,Z<:Number}(a::Z, O::T) = begin
    c = a*O.doperator
    QuantumOperator(c, O.positionbasis)
end

(*){T<:QuantumOperator}(a::T, b::T) = begin
    CompositeQuantumOperator([b,a])
end

(*){T<:Complex,N}(a::QuantumOperator{T,N},b::CompositeQuantumOperator{T,N}) = begin
    c = deepcopy(b)
    push!(c.queue, a)
    c
end

(*){T<:Complex,N}(a::CompositeQuantumOperator{T,N}, b::QuantumOperator{T,N}) = begin
    c = deepcopy(a)
    insert!(c.queue, 1, b)
    c
end

(*){T<:Complex,N}(a::CompositeQuantumOperator{T,N}, b::State{T,N}) = begin
    c = deepcopy(b)
    o = deepcopy(a)
    while !isempty(o.queue)
        c  = pop!(o.queue)*c
    end
    c
end

(/){T<:Complex,N,Z<:Number}(x::State{T,N}, a::Z) = begin
    c = deepcopy(x)
    c.components = x.components / convert(T,a)
    c
end

(/){T<:QuantumOperator,Z<:Number}(a::Z, O::T) = begin
    c = O.doperator/a
    QuantumOperator(c, O.positionbasis)
end

Base.exp{T<:QuantumOperator}(O::T) = begin
    c = exp(O.doperator)
    QuantumOperator(c, O.positionbasis)
end

function multiply!{T<:Complex,N}(O::QuantumOperator{T,N}, b::State{T,N})
    if O.positionbasis==b.inpositionbasis
        for ind in eachindex(b.components)
            b.components[ind...] = O[ind...]*b[ind...]
        end
    else
        error("Operator represented in a  different basis.")
    end
end

function multiply!{T<:Complex,N}(O::CompositeQuantumOperator{T,N}, b::State{T,N})
    Op = deepcopy(O)
    while !isempty(Op.queue)
        multiply!(pop!(Op.queue), b)
    end
    b
end

function multiply!{T<:Complex,N}(O::QuantumOperator{T,N}, b::StateSet{T,N})
    for c in b
        multiply!(O, c)
    end
end

function multiply!{T<:Complex,N}(O::CompositeQuantumOperator{T,N}, b::StateSet{T,N})
    for c in b
        multiply!(O, c)
    end
end

function multiply!{T<:FloatingPoint,N}(O::QuantumOperator{Complex{T},N}, s::State{Complex{T},N}, syd::SystemDescription{T})
    if O.positionbasis==s.inpositionbasis
        multiply!(O, s)
    else
        if O.positionbasis
            positionspace!(s, syd)
            for ind in eachindex(s.components)
                s.components[ind...] = O[ind...]*s[ind...]
            end
            momentumspace!(s, syd)
        else
            momentumspace!(s, syd)
            for ind in eachindex(s.components)
                s.components[ind...] = O[ind...]*s[ind...]
            end
            positionspace!(s, syd)
        end
    end
end

function multiply!{T<:FloatingPoint,N}(O::CompositeQuantumOperator{Complex{T},N}, s::State{Complex{T},N}, syd::SystemDescription{T})
    Op = deepcopy(O)
    while !isempty(Op.queue)
        multiply!(pop!(Op.queue), s, syd)
    end    
end

function multiply!{T<:FloatingPoint,N}(O::QuantumOperator{Complex{T},N}, s::StateSet{Complex{T},N}, syd::SystemDescription{T})
    for c in s
        multiply!(O,c,syd)
    end
end

function multiply!{T<:FloatingPoint,N}(O::CompositeQuantumOperator{Complex{T},N}, s::StateSet{Complex{T},N}, syd::SystemDescription{T})
    for c in s
        multiply!(O,c,syd)
    end
end

function multiply!{T<:Complex,Z<:Number,N}(a::Z, b::State{T,N})
    b.components = convert(T,a)*b.components
end

function multiply!{T<:Complex,N,Z<:Number}(a::Array{Z,2}, b::StateSet{T,N})
    b.states = a*b.states
end

function add!{T<:State}(a::T, b::T)
    a.components = a.components + b.components
end

function divide!{T<:Complex,Z<:Number,N}(a::State{T,N}, b::Z)
    a.components = a.components / convert(T, b)
end

# Does not check if they are of the same basis
function innerproduct{T<:Complex,N}(x::State{T,N}, y::State{T,N})
    s = assertsamedimensions(x,y)
    n = length(x.components)
    k = zero(T)
    for i=1:n
        indices = ind2sub(s, i)
        k += conj(x[indices...])*y[indices...]
    end
    k
end

function innerproduct{T<:Complex,N}(x::StateSet{T,N}, y::StateSet{T,N})
    convert(Array{T,1}, map(innerproduct, x, y))
end

LinAlg.norm{T<:State}(x::T) = sqrt(innerproduct(x,x))
LinAlg.norm{T<:StateSet}(x::T) = [norm(i) for i in x]

function overlapmatrix{T<:Complex,N}(x::StateSet{T,N})
    temp = zeros(T, length(x), length(x))
    for i=1:length(x), j=1:length(x)
        temp[i,j] = innerproduct(x[i], x[j])
    end
    Hermitian(temp)
end

function normalize!{T,N}(x::State{T,N})
    divide!(x,norm(x))
end

function normalize!{T,N}(x::StateSet{T,N})
    for i in x
        normalize!(i)
    end
end

function orthonormalize!{T,N}(s::StateSet{T,N})
    overlap = overlapmatrix(s)
    eigfactors = eigfact(overlap)
    multiply!(eigfactors[:vectors].', s)
    for i=1:length(s)
        divide!(s[i], sqrt(complex(eigfactors[:values][i])))
    end
    eigfactors[:values]
end

function orthonormalize{T,N}(s::StateSet{T,N})
    overlap = overlapmatrix(s)
    eigfactors = eigfact(overlap)
    l = eigfactors[:vectors].' * s
    for i=1:length(l)
        divide!(l[i], sqrt(complex(eigfactors[:values][i])))
    end
    l, eigfactors[:values]
end

function positionspace!{T<:FloatingPoint,N}(s::State{Complex{T},N}, syd::SystemDescription{T})
    if s.inpositionbasis==true
        return
    end
    syd.ifftplan!(s.components)
    divide!(s,syd.fftnorm)
    s.inpositionbasis = true
end

function positionspace!{T<:FloatingPoint,N}(s::StateSet{Complex{T}, N}, syd::SystemDescription{T})
    for st in s
        positionspace!(st, syd)
    end
end

function positionspace!{T<:FloatingPoint,N}(s::State{Complex{T},N})
    if s.inpositionbasis==true
        return
    end
    bfft!(s.components)
    divide!(s,sqrt(prod(size(s))))
    s.inpositionbasis = true
end

function momentumspace!{T<:FloatingPoint,N}(s::State{Complex{T},N} , syd::SystemDescription{T})
    if s.inpositionbasis==false
        return
    end

    syd.fftplan!(s.components)
    divide!(s,syd.fftnorm)
    s.inpositionbasis = false
end

function momentumspace!{T<:FloatingPoint,N}(s::State{Complex{T},N})
    if s.inpositionbasis==false
        return
    end

    fft!(s.components)
    divide!(s, sqrt(prod(size(s))))
    s.inpositionbasis = false
end

function momentumspace!{T<:FloatingPoint,N}(s::StateSet{Complex{T},N}, syd::SystemDescription{T})
    for st in s
        momentumspace!(st, syd)
    end
end

function expectationvalue{T<:Complex,N}(s::State{T,N}, o::QuantumOperator{T})
    innerproduct(s, o*s)
end

function expectationvalue{T<:Complex,N}(s::StateSet{T,N}, o::QuantumOperator{T})
    innerproduct(s, o*s)
end
