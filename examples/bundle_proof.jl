using SolitonsInBECs
using RadiiPolynomial
using UnPack
using JLD2

### Loading the data
file = load("data/Soliton_1_data.jld2")

λ = file["λ"];
v = file["v"];
l_scaling = file["scaling"];
a = file["a"];
b = file["b"];
c = file["c"];

parameters = (a = a, b = b, c = c)
N_F = order(space(component(v, 1)))


### The bundle spaces and the sequence variables
S_F = Fourier(N_F, 1.0)
X_F = ParameterSpace() × S_F^2
x = Sequence(X_F, rand(ComplexF64, dimension(X_F)))

component(x, 1) .= real(λ)
project!(component(component(x, 2), 1), component(v, 1))
project!(component(component(x, 2), 2), component(v, 2))

### The validation maps and its derivatives 
function f!(fv, v, parameters)

    @unpack a, b, c = parameters
    v₁, v₂ = eachcomponent(v)
    fv .= interval(0)

    cos2t = Sequence(Fourier(2, interval(1.0)), interval.([0.5, 0, 0, 0, 0.5]))

    project!(component(fv, 1), v₂)
    project!(component(fv, 2), interval(-a) * v₁ + interval(b) * cos2t * v₁)

    return fv

end

function F!(Fx, x, parameters, l_scaling)

    Fx .= interval(0)

    λ = component(x, 1)[1]
    v = component(x, 2)

    Fx[1] = component(v, 1)(0) - interval(l_scaling)
    project!(component(Fx, 2), differentiate(v) - f!(component(Fx, 2), v, parameters) + λ * v)

    return Fx

end

function Df!(Dfx, parameters)

    @unpack a, b, c = parameters
    Dfx .= interval(0)
    cos2t = Sequence(Fourier(2, interval(1.0)), interval.([0.5, 0, 0, 0, 0.5]))

    project!(component(Dfx, 1, 2), UniformScaling(interval(1)))
    project!(component(Dfx, 2, 1), Multiplication(-interval(a) + interval(b) * cos2t))

    return Dfx

end

function DF!(DFx, x, parameters)

    DFx .= interval(0)

    λ = component(x, 1)[1]
    v = component(x, 2)

    project!(component(component(DFx, 1, 2), 1), Evaluation(0))

    project!(component(DFx, 2, 1), v)
    project!(component(DFx, 2, 2), Derivative(1) - Df!(component(DFx, 2, 2), parameters) + λ * UniformScaling(interval(1)))

    return DFx
end


### Evaluations
x_interval = Sequence(ParameterSpace() × Fourier(N_F, interval(1))^2, interval.(coefficients(x)))

Fx_interval = Sequence(ParameterSpace() × Fourier(2N_F, interval(1.0))^2, similar(coefficients(x_interval), 1 + 2 * (4N_F + 1)))
F!(Fx_interval, x_interval, parameters, interval(l_scaling))

DF_interval = LinearOperator(space(Fx_interval), space(x_interval), similar(coefficients(x_interval), length(x_interval), length(Fx_interval)))
DF!(DF_interval, x_interval, parameters)

### The inverse of the mid-point of the operator
A = inv(mid.(project(DF_interval, space(x_interval), space(x_interval))))

### The norms of the spaces
v_weight = interval(1.05)
S_F_norm = ℓ¹(GeometricWeight(v_weight))
S_F²_norm = NormedCartesianSpace(S_F_norm, ℓ∞())
X_F_norm = NormedCartesianSpace((ℓ∞(), S_F²_norm), ℓ∞())

### The bound on the inverse of operator L times the projection and on the vector field
bound_inv_L = inv(sqrt(interval(N_F + 1)^2 + interval(λ)^2))
cos2t = Sequence(Fourier(2, interval(1.0)), interval.([0.5, 0, 0, 0, 0.5]))
bound_f = max(interval(1), abs(interval.(a)) + abs(interval.(b)) * norm(cos2t, S_F_norm))

### The Y bound
Y_tail = deepcopy(component(Fx_interval, 2))
for i ∈ 1:2
    component(Y_tail, i)[-N_F:N_F] .= interval(0)
    for j ∈ N_F+1:2N_F
        component(Y_tail, i)[j] .= inv(sqrt(interval(j)^2 + interval(λ)^2)) * component(Y_tail, i)[j]
        component(Y_tail, i)[-j] .= inv(sqrt(interval(j)^2 + interval(λ)^2)) * component(Y_tail, i)[-j]
    end
end
Y = norm(interval.(A) * Fx_interval, X_F_norm) + norm(Y_tail, S_F²_norm)

### The Z1 bound
Z₁_finite = opnorm(interval(A) * DF_interval - UniformScaling(interval(1)), X_F_norm)
Z₁_infinite_tail = bound_inv_L * bound_f
Z1 = Z₁_finite + Z₁_infinite_tail

### The Z2 bound
Z2 = interval(2) * max(opnorm(interval(A), X_F_norm), bound_inv_L)

### The Proof
R = Inf
setdisplay(:full)
interval_of_existence(Y, Z1, Z2, R)

