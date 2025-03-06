using SolitonsInBECs
using RadiiPolynomial
using UnPack
using JLD2

### Loading the data
file = load("data/Soliton_1_data.jld2")

λ = file["λ"];
v = file["v"];
a = file["a"];
b = file["b"];
c = file["c"];
σ = file["σ"];
θ = file["θ"];
P = file["w"];
s = file["s"];
l_scaling = file["scaling"];


### Then radius from the other two proofs
parameters = (a=a, b=b, c=c)
r_TF = interval(1.5204458252945915e-7)
r_F = interval(4.122891017172993e-13)
σ = real(σ[1])
κ = (1 + 2 * pi) / 2

N_F = order(space(component(v, 1)))
N_T = order(space(component(P, 1)))[2]
N_C = order(space(component(s, 1)))

function f!(f_, P, parameters)

    f_ .= 0

    @unpack a, b, c = parameters

    P₁, P₂, P₃, P₄ = eachcomponent(P)

    project!(component(f_, 1), P₂)
    project!(component(f_, 2), -a * P₁ + b * P₃ * P₁ + c * (P₁^3))
    project!(component(f_, 3), P₄)
    project!(component(f_, 4), -4.0 * P₃)

    return f_

end

function EvaluateManifold(θ, σ, P)

    [real(component(P, i)(real(σ[1]), real(θ))) for i ∈ 1:4]

end

function shift(Df, n)

    Df_shifted = deepcopy(Df)
    for i in 1:4
        for j in 1:4
            component(Df_shifted, i, j) .= circshift(component(Df, i, j)[:, :], (n, 0))
        end
    end

    return Df_shifted

end

function Df!(Df_, P, parameters)

    @unpack a, b, c = parameters
    P₁, P₂, P₃, P₄ = eachcomponent(P)

    project!(component(Df_, 1, 1), 0 * I)
    project!(component(Df_, 1, 2), Multiplication(-1 * one(P₂)))
    project!(component(Df_, 1, 3), 0 * I)
    project!(component(Df_, 1, 4), 0 * I)

    project!(component(Df_, 2, 1), Multiplication(a - b * P₃ - 3 * c * (P₁^2)))
    project!(component(Df_, 2, 2), 0 * I)
    project!(component(Df_, 2, 3), Multiplication(-b * P₁))
    project!(component(Df_, 2, 4), 0 * I)

    project!(component(Df_, 3, 1), 0 * I)
    project!(component(Df_, 3, 2), 0 * I)
    project!(component(Df_, 3, 3), 0 * I)
    project!(component(Df_, 3, 4), Multiplication(-1 * one(P₄)))

    project!(component(Df_, 4, 1), 0 * I)
    project!(component(Df_, 4, 2), 0 * I)
    project!(component(Df_, 4, 3), Multiplication(4 * one(P₃)))
    project!(component(Df_, 4, 4), 0 * I)


    return Df_
end

function F_soliton!(F, s, parameters, manifold)

    F .= 0

    ### Solving for sigma - everything else is fixed
    σ = real(component(s, 1)[1])
    s₁, s₂, s₃, s₄ = eachcomponent(component(s, 2))
    M = order(s₁)[1]

    ### Fixed variables
    θ = 1
    L = θ + 2 * pi

    rightB = EvaluateManifold(θ, σ, manifold)
    leftB = [0, 0, 1, 0]


    ### Getting  Cheb c_k sequence
    C = (L / 2) * f!(component(F, 2), component(s, 2), parameters)
    c₁, c₂, c₃, c₄ = eachcomponent(C)

    ### Initial conditions 
    component(component(F, 2), 1)[0] .= s₂(1) - rightB[2]
    component(component(F, 2), 2)[0] .= s₂(-1) - leftB[2]
    component(component(F, 2), 3)[0] .= s₃(-1) - leftB[3]
    component(component(F, 2), 4)[0] .= s₄(-1) - leftB[4]


    ### Main loop
    for i ∈ 1:M-1

        component(component(F, 2), 1)[i] .= 2 * i * s₁[i] + c₁[i+1] - c₁[i-1]
        component(component(F, 2), 2)[i] .= 2 * i * s₂[i] + c₂[i+1] - c₂[i-1]
        component(component(F, 2), 3)[i] .= 2 * i * s₃[i] + c₃[i+1] - c₃[i-1]
        component(component(F, 2), 4)[i] .= 2 * i * s₄[i] + c₄[i+1] - c₄[i-1]


    end

    ### Last coefficient
    component(component(F, 2), 1)[M] .= 2 * M * s₁[M] - c₁[M]
    component(component(F, 2), 2)[M] .= 2 * M * s₂[M] - c₂[M]
    component(component(F, 2), 3)[M] .= 2 * M * s₃[M] - c₃[M]
    component(component(F, 2), 4)[M] .= 2 * M * s₄[M] - c₄[M]

    ### Extra Equation
    component(F, 1)[1] .= s₁(1) - rightB[1]

    return F
end


function DF_soliton!(DF_, s, parameters, manifold)

    ### Solving for sigma - everything else is fixed  
    DF_ .= 0
    σ = real(component(s, 1)[1])
    M = order(codomain(DF_)[2])[1]
    M_dom = order(domain(DF_)[2])[1]


    ### Fixed variables
    θ = 1
    L = θ + 2 * pi

    #### Check the space
    maniDerivative = EvaluateManifold(θ, σ, Derivative(1, 0) * manifold)

    Df = Df!(component(DF_, 2, 2), component(s, 2), parameters)

    # Space_cheb = domain(Df)[2]
    Df_up = shift(Df, -1)
    Df_down = shift(Df, 1)

    Df_21 = component(Df_up, 2, 1)
    Df_21 .= Df_21 - component(Df_down, 2, 1)
    Df_21[end, :] .= -component(Df, 2, 1)[end, :]
    Df_21[0, :] .= 0

    Df_23 = component(Df_up, 2, 3)
    Df_23 .= Df_23 - component(Df_down, 2, 3)
    Df_23[end, :] .= -component(Df, 2, 3)[end, :]
    Df_23[0, :] .= 0


    ### Main loop
    for i ∈ 1:M-1
        component(component(DF_, 2, 2), 1, 2)[i, i+1] .= 1.0
        component(component(DF_, 2, 2), 3, 4)[i, i+1] .= 1.0
        component(component(DF_, 2, 2), 4, 3)[i, i+1] .= -4.0
    end

    for i ∈ 1:M-1
        component(component(DF_, 2, 2), 1, 2)[i, i-1] .= -1.0
        component(component(DF_, 2, 2), 3, 4)[i, i-1] .= -1.0
        component(component(DF_, 2, 2), 4, 3)[i, i-1] .= 4.0
    end

    component(component(DF_, 2, 2), 2, 1) .= Df_21
    component(component(DF_, 2, 2), 2, 3) .= Df_23
    component(component(DF_, 2, 2), 1, 2)[M, M] .= -1.0
    component(component(DF_, 2, 2), 3, 4)[M, M] .= -1.0
    component(component(DF_, 2, 2), 4, 3)[M, M] .= 4.0

    DF_ = (L / 2) * (DF_)


    for i ∈ 1:M
        component(component(DF_, 2, 2), 1, 1)[i, i] .= 2 * i
        component(component(DF_, 2, 2), 2, 2)[i, i] .= 2 * i
        component(component(DF_, 2, 2), 3, 3)[i, i] .= 2 * i
        component(component(DF_, 2, 2), 4, 4)[i, i] .= 2 * i
    end


    ### The boundary conditions 
    component(component(DF_, 2, 2), 1, 2)[0, :] .= [1; 2 * (1) .^ (1:M_dom)]
    component(component(DF_, 2, 2), 2, 2)[0, :] .= [1; 2 * (-1) .^ (1:M_dom)]
    component(component(DF_, 2, 2), 3, 3)[0, :] .= [1; 2 * (-1) .^ (1:M_dom)]
    component(component(DF_, 2, 2), 4, 4)[0, :] .= [1; 2 * (-1) .^ (1:M_dom)]

    component(component(DF_, 2, 1), 1)[0, :] .= -maniDerivative[2]
    component(DF_, 1, 1) .= -maniDerivative[1]
    component(component(DF_, 1, 2), 1) .= transpose([1; 2 * (1) .^ (1:M_dom)])

    return DF_


end


### Spaces
solitonSpace = ParameterSpace() × Chebyshev(N_C)^4
padded_space = ParameterSpace() × Chebyshev(3N_C + 1)^4

### Evaluations
x_interval = Sequence(solitonSpace, rand(Float64, dimension(solitonSpace)))
component(x_interval, 1) .= σ
project!(component(x_interval, 2), s)
x_interval = interval.(x_interval)

Fx_interval = Sequence(padded_space, zeros(Float64, dimension(padded_space)))
Fx_interval = interval.(Fx_interval)

Fx_interval_N = copy(x_interval)
Fx_interval_N .= 0.0

DF_interval = LinearOperator(space(Fx_interval), space(x_interval), similar(coefficients(x_interval), length(x_interval), length(Fx_interval)))

F_soliton!(Fx_interval_N, x_interval, parameters, P)
F_soliton!(Fx_interval, x_interval, parameters, P)
DF_interval = DF_soliton!(DF_interval, x_interval, parameters, P)

### A operator finite part
A = inv(mid.(project(DF_interval, space(x_interval), space(x_interval))))

### The norms
w_weight = 1.05
S_C_norm = ℓ¹(GeometricWeight(w_weight))
S_C⁴_norm = NormedCartesianSpace(S_C_norm, ℓ∞())
X_C_norm = NormedCartesianSpace((ℓ∞(), S_C⁴_norm), ℓ∞())

### Components of the soliton and parameterization
s₁, s₂, s₃, s₄ = eachcomponent(component(x_interval, 2))
W = EvaluateManifold(θ, σ, P)
dW = EvaluateManifold(θ, σ, Derivative(1, 0)P)

### The bound on the inverse of operator L times the projection and on the vector field
bound_inv_L = inv(interval(2N_C))
bound_Df = max(4,
    abs(interval(a)) +
    abs(interval(b)) * norm(s₃, S_C_norm) +
    abs(interval(b)) * norm(s₁, S_C_norm) +
    interval(3) * abs(interval(c)) * norm(s₁ * s₁, S_C_norm)
)

bound_T = 2 * v

opnorm_A = opnorm(interval.(A), X_C_norm)

bound_B = max(
    abs(s₁(1) - W[1]) + r_TF,
    abs(s₂(1) - W[2]) + r_TF,
    abs(s₃(-1) - 1),
    abs(s₄(-1))
)

bound_A_par_N_0 = opnorm(project(A, ParameterSpace() × Chebyshev(0)^4, ParameterSpace() × Chebyshev(N_C)^4), X_C_norm)

bound_z1_B = r_TF / (1 - abs(σ))^2 + 1 / w_weight^(3N_C + 2)

### The Y bound
Y_tail = deepcopy(component(Fx_interval, 2))

for i ∈ 1:4
    component(Y_tail, i)[0:N_C] .= 0.0
    for j ∈ N_C+1:3N_C+1
        component(Y_tail, i)[j] .= inv(2 * j) * component(Y_tail, i)[j]
    end
end

Y = norm(interval.(A) * Fx_interval_N, X_C_norm) + norm(Y_tail, S_C⁴_norm) + opnorm_A * bound_B

### The Z1 bound
Z₁_finite = opnorm(A * DF_interval - I, X_C_norm)
Z₁_B = (w_weight * abs(κ) / N_C) * bound_Df
Z₁_C = bound_A_par_N_0 * bound_z1_B
Z1 = Z₁_finite + Z₁_B + Z₁_C

### The Z2 bound
r_star_C = 1e-2

int_sigma = interval(σ, r_star_C; format=:midpoint)
ddW = EvaluateManifold(1, int_sigma, Derivative(2, 0)P)

Z2_1 = opnorm_A + (1 / (2N_C))
Z2_2 = 2 * abs(interval(b)) + interval(6) * abs(interval(c)) * norm(s₁, S_C_norm) + interval(3) * abs(interval(c)) * r_star_C
Z2_3 = max(
    (abs(int_sigma)^2 + abs(int_sigma)) * r_TF / (1 - abs(int_sigma))^3 +
    (2 + abs(int_sigma)) * r_TF / (1 - abs(int_sigma))^2 +
    abs(ddW[1]),
    (abs(int_sigma)^2 + abs(int_sigma)) * r_TF / (1 - abs(int_sigma))^3 +
    (2 + abs(int_sigma)) * r_TF / (1 - abs(int_sigma))^2 +
    abs(ddW[2]),
)

Z2 = 2 * w_weight * abs(κ) * Z2_1 * Z2_2 + Z2_3 * bound_A_par_N_0


### The proof
setdisplay(:full)
interval_of_existence(Y, Z1, Z2, r_star_C)



