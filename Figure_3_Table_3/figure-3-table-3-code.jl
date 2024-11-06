using CairoMakie
using NLsolve

# This code generates Figures 3 and Table 3 in https://arxiv.org/abs/2409.14776.

# Preliminaries
## We assess the worth of a distribution of outcomes using the notion of a quasi-arithmetic mean with generator f
## (https://en.wikipedia.org/wiki/Quasi-arithmetic_mean)
## The class of homogeneous quasi-arithmetic means can be parametrized by gamma as follows:

function f(x, gamma)     
    if gamma == 1 
        return log(x)
    else
       return x^(1 - gamma) / (1 - gamma)
    end
end

## With inverse given by:

function f_inv(u, gamma)
    if gamma == 1 
        return exp(u)
    else
        return ((1-gamma)*u)^(1/(1-gamma))
    end
end

## See https://osf.io/tnu2q/ for more on this class (and other classes) of quasi-arithmetic means.

# In what follows, I refer to tables and results in https://arxiv.org/abs/2409.14776.
## Define the left hand side and the right hand side of the last equation in Theorem 3.1.

RHS(eeb_u, eea_l, gamma, z) = eeb_u - f_inv(f(eea_l, gamma) + (f(eeb_u,gamma) - f(eea_l,gamma)) * z, gamma)

LHS(eeb_l, eea_u, gamma, z) = eea_u - f_inv(f(eea_u, gamma) + (f(eeb_l,gamma) - f(eea_u,gamma)) * z, gamma)

# This function solves the egalitarian equivalent minimax regret decision problem 
function solve_for_z(eeb_u, eea_l, eeb_l, eea_u, gamma)
    function system!(F, z)
        F[1] = RHS(eeb_u, eea_l, gamma, z[1]) - LHS(eeb_l, eea_u, gamma, z[1])
    end

    # Initial guess for z
    z_guess = [0.5]
    solution = nlsolve(system!, z_guess)

    if converged(solution)
        return  max.(0, min.(solution.zero, 1))
    else
        return "No solution found"
    end
end


# Preparing Table 3

## Horowitz - Manski bounds, gamma = 0 (From Michelle's calculations)
gamma_0 = 0

eea_u_hm_level = 10.7493
eea_l_hm_level = 5.4431

eeb_u_hm_level = 11.9906
eeb_l_hm_level = 6.1253

sol_hm_level = solve_for_z(eeb_u_hm_level, eea_l_hm_level, eeb_l_hm_level, eea_u_hm_level, gamma_0)


## Horowitz - Manski bounds, gamma = 2 (from Lee 2009, p. 1080, and Michelle's computations)
gamma_2 = 2

E_l_f_y_a_2 = -.2457982
E_u_f_y_a_2 = -.1106156

E_l_f_y_b_2 = -.2661934
E_u_f_y_b_2 = -.1167666

eea_u_hm_2 = f_inv(E_u_f_y_a_2, gamma_2)
eea_l_hm_2 =f_inv(E_l_f_y_a_2, gamma_2)

eeb_u_hm_2 = f_inv(E_u_f_y_b_2, gamma_2)
eeb_l_hm_2 =f_inv(E_l_f_y_b_2, gamma_2)

sol_hm_2 = solve_for_z(eeb_u_hm_2, eea_l_hm_2, eeb_l_hm_2, eea_u_hm_2, gamma_2)

## Lee bounds, gamma = 0 (From Michelle's calculations)
eea_u_Lee_level = 7.9156
eea_l_Lee_level = 7.9156

eeb_u_Lee_level = 8.6729
eeb_l_Lee_level = 7.5262

sol_l_level = solve_for_z(eeb_u_Lee_level, eea_l_Lee_level, eeb_l_Lee_level, eea_u_Lee_level, gamma_0)


## Lee bounds, gamma = 2 (From Lee 2009, p. 1092 and Michelle's computations)
E_l_f_y_a_Lee_2 = -.1520497
E_u_f_y_a_Lee_2 = -.1520497

E_l_f_y_b_Lee_2 = -.1538839
E_u_f_y_b_Lee_2 = -.1290454

eea_u_Lee_2 = f_inv(E_u_f_y_a_Lee_2, gamma_2)
eea_l_Lee_2 =f_inv(E_l_f_y_a_Lee_2, gamma_2)

eeb_u_Lee_2 = f_inv(E_u_f_y_b_Lee_2, gamma_2)
eeb_l_Lee_2 =f_inv(E_l_f_y_b_Lee_2, gamma_2)

sol_l_2 = solve_for_z(eeb_u_Lee_2, eea_l_Lee_2, eeb_l_Lee_2, eea_u_Lee_2, gamma_2)


## Chen and Flores bounds, gamma = 0 (From Michelle's calculations)
eea_u_CF_level = eea_u_Lee_level
eea_l_CF_level = eea_l_Lee_level
eeb_u_CF_level = eeb_u_Lee_level
eeb_l_CF_level = 8.3317

sol_CF_level = solve_for_z(eeb_u_CF_level, eea_l_CF_level, eeb_l_CF_level, eea_u_CF_level, gamma_0)

## Chen and Flores bounds, gamma = 2 (From Michelle's calculations)
E_l_f_y_a_CF_2 = E_l_f_y_a_Lee_2
E_u_f_y_a_CF_2 = E_u_f_y_a_Lee_2

E_l_f_y_b_CF_2 = -.1475316
E_u_f_y_b_CF_2 = E_u_f_y_b_Lee_2

eea_u_CF_2 = f_inv(E_u_f_y_a_CF_2, gamma_2)
eea_l_CF_2 =f_inv(E_l_f_y_a_CF_2, gamma_2)

eeb_u_CF_2 = f_inv(E_u_f_y_b_CF_2, gamma_2)
eeb_l_CF_2 =f_inv(E_l_f_y_b_CF_2, gamma_2)

sol_CF = solve_for_z(eeb_u_CF_2, eea_l_CF_2, eeb_l_CF_2, eea_u_CF_2, gamma_2)

# Preparing Figure 3

## Generate z values
z_values = 0:0.01:1

## Calculate LHS and RHS values for Horowitz-Manski (HM), Lee and CF bounds, gamma = 0
rhs_values_hm_level = [RHS(eeb_u_hm_level, eea_l_hm_level, gamma_0, z) for z in z_values]
lhs_values_hm_level = [LHS(eeb_l_hm_level, eea_u_hm_level, gamma_0, z) for z in z_values]

rhs_values_Lee_level = [RHS(eeb_u_Lee_level, eea_l_Lee_level, gamma_0, z) for z in z_values]
lhs_values_Lee_level = [LHS(eeb_l_Lee_level, eea_u_Lee_level, gamma_0, z) for z in z_values]

rhs_values_CF_level = [RHS(eeb_u_CF_level, eea_l_CF_level, gamma_0, z) for z in z_values]
lhs_values_CF_level = [LHS(eeb_l_CF_level, eea_u_CF_level, gamma_0, z) for z in z_values]

## Calculate LHS and RHS values for Horowitz-Manski (HM), Lee and CF bounds, gamma = 2
rhs_values_hm_2 = [RHS(eeb_u_hm_2, eea_l_hm_2, gamma_2, z) for z in z_values]
lhs_values_hm_2 = [LHS(eeb_l_hm_2, eea_u_hm_2, gamma_2, z) for z in z_values]

rhs_values_Lee_2 = [RHS(eeb_u_Lee_2, eea_l_Lee_2, gamma_2, z) for z in z_values]
lhs_values_Lee_2 = [LHS(eeb_l_Lee_2, eea_u_Lee_2, gamma_2, z) for z in z_values]

rhs_values_CF_2 = [RHS(eeb_u_CF_2, eea_l_CF_2, gamma_2, z) for z in z_values]
lhs_values_CF_2 = [LHS(eeb_l_CF_2, eea_u_CF_2, gamma_2, z) for z in z_values]

## Create a figure with six subplots (axes)
fig = Figure();

ax1 = Axis(fig[1, 1], ylabel="EE Regret, γ=0", title="w / Horowitz-Manski Bounds")
ax2 = Axis(fig[1, 2], title="w / Lee Bounds")
ax3 = Axis(fig[1, 3], title="w / Chen-Flores Bounds")
ax4 = Axis(fig[2, 1], ylabel="EE Regret, γ=2", xlabel="δ")
ax5 = Axis(fig[2, 2], xlabel="δ")
ax6 = Axis(fig[2, 3], xlabel="δ")

## Plot LHS and RHS on the first axis (Horowitz-Manski bounds)
lines!(ax4, z_values, lhs_values_hm_2, label="LHS")
lines!(ax4, z_values, rhs_values_hm_2, label="RHS")

## Plot LHS and RHS on the second axis (Lee bounds)
lines!(ax5, z_values, lhs_values_Lee_2, label="LHS")
lines!(ax5, z_values, rhs_values_Lee_2, label="RHS")

## Plot LHS and RHS on the third axis (Chen-Flores bounds)
lines!(ax6, z_values, lhs_values_CF_2, label="LHS")
lines!(ax6, z_values, rhs_values_CF_2, label="RHS")

## Plot LHS and RHS on the fourth axis (level Horowitz-Manski bounds)
lines!(ax1, z_values, lhs_values_hm_level, label="LHS")
lines!(ax1, z_values, rhs_values_hm_level, label="RHS")

## Plot LHS and RHS on the fifth axis (level Lee bounds)
lines!(ax2, z_values, lhs_values_Lee_level, label="LHS")
lines!(ax2, z_values, rhs_values_Lee_level, label="RHS")

## Plot LHS and RHS on the sixth axis (level Chen-Flores bounds)
lines!(ax3, z_values, lhs_values_CF_level, label="LHS")
lines!(ax3, z_values, rhs_values_CF_level, label="RHS")

## Link the y-axes of both plots
linkyaxes!(ax1, ax2, ax3, ax4, ax5, ax6)

fig

save("Figure_3_Table_3/jobcorps_ota_2_and_1.png", fig)
