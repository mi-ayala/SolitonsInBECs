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
w = file["w"];

parameters = (a = a, b = b, c = c)

r_F = interval(4.122891017172993e-13)

N_F = order(space(component(v, 1)))
N_T = order(space(component(w, 1)))[2]


### Building sequences variables
x_manifold = Sequence((Taylor(N_T) ⊗ Fourier(N_F, 1.0))^2, zeros(Complex{Float64}, dimension((Taylor(N_T) ⊗ Fourier(N_F, 1.0))^2)))
x_bundle = Sequence(ParameterSpace() × Fourier(N_F, interval(1.0))^2, zeros(Complex{Float64}, dimension(ParameterSpace() × Fourier(N_F, interval(1.0))^2)))

component(x_bundle, 1) .= λ
project!(component(component(x_bundle, 2), 1), component(v, 1))
project!(component(component(x_bundle, 2), 2), component(v, 2))

project!(component(x_manifold, 1), component(w, 1))
project!(component(x_manifold, 2), component(w, 2))

v = component(x_bundle, 2)

v_bundle = Sequence(Fourier(N_F, interval(1.0))^2, interval.(coefficients(v)))

### The validation maps and its derivatives    
function f!(fw, w, parameters, w₃)

    @unpack a, b, c = parameters
    fw .= interval(0.0)

    w₁, w₂ = eachcomponent(w)

    project!(component(fw, 1), w₂)
    project!(component(fw, 2), interval(-a) * w₁ + interval(b) * w₃ * w₁ + interval(c) * (w₁^3))

end

function F!(Fx, x, parameters, v, λ, γ, w₃)

    Fx .= interval(0.0)
    f!(Fx, x, parameters, w₃)

    m_bundle = order(space(component(v, 1)))

    for i = 1:2
        component(Fx, i) .= -component(Fx, i) + Derivative(0, 1) * component(x, i) + λ * component(Derivative(1, 0) * x, i) * Sequence(Taylor(1) ⊗ Fourier(0, interval(1.0)), interval.([0, 1]))
        component(Fx, i)[(0, -m_bundle:m_bundle)] .= component(x, i)[(0, -m_bundle:m_bundle)] .- component(γ, i)
        component(Fx, i)[(1, -m_bundle:m_bundle)] .= component(x, i)[(1, -m_bundle:m_bundle)] .- component(v, i)
    end

end

function Df!(Dfw, w, parameters, w₃)

    Dfw .= interval(0)

    @unpack a, b, c = parameters
    w₁, w₂ = eachcomponent(w)


    project!(component(Dfw, 1, 2), Multiplication(interval(-1) * one(w₂)))


    project!(component(Dfw, 2, 1), Multiplication(interval(a) - interval(b) * w₃ - interval(3 * c) * (w₁^2)))




end

function DF!(DFx, x, parameters, λ, w₃, N_F)

    DFx .= interval(0)

    Df!(DFx, x, parameters, w₃)

    type = eltype(x[1])
    dom = domain(component(DFx, 1, 1))
    codom = codomain(component(DFx, 1, 1))

    project!(component(DFx, 1, 1), project(Derivative(0, 1) + λ * project(Multiplication(Sequence(Taylor(1) ⊗ Fourier(0, interval(1.0)), interval.([0, 1]))), dom, codom) * project(Derivative(1, 0), dom, codom, type), dom, codom))

    project!(component(DFx, 2, 2), component(DFx, 1, 1))


    for i = 1:2
        for j in -N_F:N_F
            component(DFx, i, i)[(0, j), (0, j)] = interval(1)
            component(DFx, i, i)[(1, j), (1, j)] = interval(1)
        end
    end

    component(DFx, 1, 2)[(0, :), (0, :)] .= interval(0)
    component(DFx, 1, 2)[(1, :), (1, :)] .= interval(0)

    component(DFx, 2, 1)[(0, :), (:, :)] .= interval(0)
    component(DFx, 2, 1)[(1, :), (:, :)] .= interval(0)


end

### Evaluations
w₃ = Sequence((Taylor(N_T) ⊗ Fourier(N_F, interval(1.0))), interval.(coefficients(component(w, 3))))

γ = project(
    Sequence(Fourier(2, interval(1.0))^4, interval.([zeros(5); zeros(5); [0.5, 0, 0, 0, 0.5]; -2 * [0.5im, 0, 0, 0, -0.5im]])),
    Fourier(N_F, interval(1.0))^4)

padded_space = (Taylor(3N_T) ⊗ Fourier(3N_F, interval(1.0)))^2

x_interval = Sequence((Taylor(N_T) ⊗ Fourier(N_F, interval(1.0)))^2, interval.(coefficients(x_manifold)))
Fx_interval = Sequence(padded_space, zeros(eltype(x_interval), dimension(padded_space)))
Fx_interval_N = Sequence(space(x_interval), zeros(eltype(x_interval), dimension(space(x_interval))))

F!(Fx_interval_N, x_interval, parameters, v_bundle, interval(λ), γ, w₃)
F!(Fx_interval, x_interval, parameters, v_bundle, interval(λ), γ, w₃)


# DFx_interval = LinearOperator(space(Fx_interval), space(x_interval), similar(coefficients(x_interval), length(x_interval), length(Fx_interval)))
# DF!(DFx_interval, x_interval, parameters, interval(λ), w₃, N_F)
# save("DFx_interval_manifold.jld2", "DFx_interval", DFx_interval)

file2 = jldopen("data/DFx_interval_manifold.jld2")
DFx_interval = file2["DFx_interval"]

### The inverse of the mid-point of the operator
A = inv(mid.(project(DFx_interval, space(x_interval), space(x_interval))))

### The norms of the spaces
v_weight = interval(1.05)
S_TF_norm = ℓ¹(IdentityWeight(), GeometricWeight(v_weight))
S_TF²_norm = NormedCartesianSpace(S_TF_norm, ℓ∞())


### The bound on the inverse of operator L times the projection and the vector field
bound_inv_L = inv(sqrt(interval(2λ)^2 + interval(N_F + 1)^2)) + inv(abs(interval(λ) * interval(N_T + 1)))

bound_Df = max(interval(1), abs(interval.(a)) + abs(interval.(b)) * v_weight^2 + interval(3) * abs(interval.(c)) * norm(component(x_interval, 1) * component(x_interval, 1), S_TF_norm))

opnorm_A = opnorm(interval.(A), S_TF²_norm)


### The Y bound

Y_tail = deepcopy(Fx_interval)
for i ∈ 1:2
    component(Y_tail, i)[(0:N_T, -N_F:N_F)] .= interval(0)
    for j ∈ N_T+1:3N_T
        for k ∈ 0:3N_F
            component(Y_tail, i)[(j, k)] .= inv(abs(interval(k * im) + interval(λ) * interval(j))) * component(Y_tail, i)[(j, k)]
            component(Y_tail, i)[(j, -k)] .= inv(abs(interval(-k * im) + interval(λ) * interval(j))) * component(Y_tail, i)[(j, -k)]
        end
    end
end

for i ∈ 1:2
    for j ∈ 2:N_T
        for k ∈ N_F+1:3N_F
            component(Y_tail, i)[(j, k)] .= inv(abs(interval(k * im) + interval(λ) * interval(j))) * component(Y_tail, i)[(j, k)]
            component(Y_tail, i)[(j, -k)] .= inv(abs(interval(-k * im) + interval(λ) * interval(j))) * component(Y_tail, i)[(j, -k)]
        end
    end
end

Y = interval(r_F) * opnorm_A + norm(interval.(A) * Fx_interval_N, S_TF²_norm) + norm(Y_tail, S_TF²_norm) + interval(r_F)


### The Z1 bound

# Z₁_finite = opnorm(interval.(A) * DFx_interval - UniformScaling(interval(1)), S_TF²_norm)
# ### Save  Z₁_finite
# save("data/Z1_finite_manifold.jld2", "Z1_finite", Z₁_finite)

file3 = jldopen("data/Z1_finite_manifold.jld2")
Z₁_finite = file3["Z1_finite"]
Z₁_tail = bound_inv_L * bound_Df
Z1 = Z₁_finite + Z₁_tail

### The Z2 bound
r_star_TF = 1e-3
Z2 = interval(3) * abs(interval(c)) * (opnorm_A + bound_inv_L) * (interval(r_star_TF) + interval(2) * norm(component(x_interval, 1), S_TF_norm))

### The proof
setdisplay(:full)
interval_of_existence(Y, Z1, Z2, r_star_TF)


