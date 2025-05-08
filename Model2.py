import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.interpolate import interp1d
import scipy.constants as const

# ---------------- Constants ----------------
T_surface = 163     # K
T_core_vals = [245, 278, 295]  # min, mean, max core temps
R = 470e3           # m
G = const.G         # Gravitational constant
del_r = 10           # m

# ---------------- Layers -------------------
def define_layers(T_core):
    return [
        {
            "name": "core",
            "r_min": 0,
            "r_max": 350e3,
            "rho_0": 2465.35,
            "k": 2.0,
            "alpha": 3e-5,
            "eta": 1e21,
            "Cp": 900,
            "K": 3e10,
        },
        {
            "name": "mantle",
            "r_min": 350e3,
            "r_max": 426e3,
            "rho_0": 2389,
            "k": 1.5,
            "alpha": 25e-6,
            "eta": 1e19,
            "Cp": 1500,
            "K": 1.5e10,
        },
        {
            "name": "crust",
            "r_min": 426e3,
            "r_max": R,
            "rho_0": 1360,
            "k": 1.5,
            "alpha": 40e-6,
            "eta": 1.5e20,
            "Cp": 1300,
            "K": 5e9,
        }
    ]

# ---------------- Helper Functions -------------------
def get_layer(radius, layers):
    for layer in layers:
        if layer["r_min"] <= radius <= layer["r_max"]:
            return layer
    raise ValueError("Radius out of bounds.")

def get_density_profile(r, layers):
    return np.array([get_layer(ri, layers)["rho_0"] for ri in r])

# ---------------- Physics Functions -------------------
def upward_integration(r, del_r, rho_profile):
    M = np.zeros_like(r)
    g = np.zeros_like(r)
    for i in range(1, len(r)):
        dMdr = 4 * np.pi * rho_profile[i - 1] * r[i - 1] ** 2
        M[i] = M[i - 1] + dMdr * del_r
    g[1:] = G * M[1:] / r[1:] ** 2
    return M, g

def downward_integration(r, g, del_r, rho_profile, layers):
    P = np.zeros_like(r)
    P[-1] = 0
    for i in range(len(r)-1, 0, -1):
        P[i-1] = P[i] + rho_profile[i] * g[i] * del_r
    P_boundaries = [P[np.searchsorted(r, layer["r_min"])] for layer in layers]
    return P, P_boundaries

def conductive_temp(r, layers, T_surface, T_core):
    n_layers = len(layers)
    radii = [layer["r_min"] for layer in layers] + [layers[-1]["r_max"]]
    ks = [layer["k"] for layer in layers]
    A = [ks[i] / (radii[i + 1] - radii[i]) for i in range(n_layers)]
    T_boundaries = np.zeros(n_layers + 1)
    T_boundaries[0] = T_core
    T_boundaries[-1] = T_surface
    A_sys = np.zeros((n_layers - 1, n_layers - 1))
    b_sys = np.zeros(n_layers - 1)
    for i in range(n_layers - 1):
        if i > 0:
            A_sys[i, i - 1] = -A[i]
        A_sys[i, i] = A[i] + A[i + 1]
        if i < n_layers - 2:
            A_sys[i, i + 1] = -A[i + 1]
        if i == 0:
            b_sys[i] = A[i] * T_core
        if i == n_layers - 2:
            b_sys[i] += A[i + 1] * T_surface
    T_internal = np.linalg.solve(A_sys, b_sys)
    T_boundaries[1:-1] = T_internal
    T_profile = np.zeros_like(r)
    for i in range(n_layers):
        r_start = radii[i]
        r_end = radii[i + 1]
        T_start = T_boundaries[i]
        T_end = T_boundaries[i + 1]
        mask = (r >= r_start) & (r <= r_end)
        T_profile[mask] = T_start + (T_end - T_start) * (r[mask] - r_start) / (r_end - r_start)
    return T_profile, T_boundaries

def rayleigh_number_layer(r_top, r_bottom, T_top, T_bottom, g_avg, layer):
    delta_T = abs(T_bottom - T_top)
    D = abs(r_bottom - r_top)
    kappa = layer["k"] / (layer["rho_0"] * layer["Cp"])
    Ra = (layer["rho_0"] * layer["alpha"] * g_avg * delta_T * D**3) / (kappa * layer["eta"])
    return Ra

def calculate_rayleigh_numbers(r, g, T, layers):
    results = []
    for layer in layers:
        idx_top = np.searchsorted(r, layer["r_min"])
        idx_bot = np.searchsorted(r, layer["r_max"])
        g_avg = np.mean(g[idx_top:idx_bot+1])
        T_top = T[idx_top]
        T_bot = T[idx_bot]
        Ra = rayleigh_number_layer(r[idx_top], r[idx_bot], T_top, T_bot, g_avg, layer)
        results.append((layer["name"], Ra))
    return results

def convective_layer_profile(Ra, r, T_profile, g, layer, del_r, Ra_crit=1000):
    D = abs(layer["r_max"] - layer["r_min"])
    delta = D * (Ra_crit / Ra)**(1/3)
    delta = min(delta, D / 2)
    idx_top = np.searchsorted(r, layer["r_min"])
    idx_bot = np.searchsorted(r, layer["r_max"])
    T_new = T_profile.copy()
    T_top = T_profile[idx_top]
    T_bottom = T_profile[idx_bot]
    T_avg = (T_top + T_bottom) / 2
    r_segment = r[idx_top:idx_bot+1]
    T_adiabat = T_avg
    for i, radius in enumerate(r_segment):
        idx = idx_top + i
        if radius <= (layer["r_min"] + delta):
            frac = (radius - layer["r_min"]) / delta
            T_new[idx] = T_top + frac * (T_avg - T_top)
        elif radius >= (layer["r_max"] - delta):
            frac = (radius - (layer["r_max"] - delta)) / delta
            T_new[idx] = T_avg + frac * (T_bottom - T_avg)
        else:
            g_i = g[idx]
            dT = (layer["alpha"] * g_i * T_adiabat) / layer["Cp"] * del_r
            T_adiabat += dT
            T_new[idx] = T_adiabat
    return T_new

def update_density_profile(r, T, P, layers, T_boundaries, P_boundaries):
    rho_new = np.zeros_like(r)
    for i in range(len(r)):
        radius = r[i]
        T_local = T[i]
        P_local = P[i]
        layer = get_layer(radius, layers)
        layer_index = next(j for j, l in enumerate(layers) if l["name"] == layer["name"])
        #T_ref = T_boundaries[layer_index]
        #P_ref = P_boundaries[layer_index]
        T_ref = 163        #K on ceres surface
        rho_0 = 1360       # kg/m^3 on ceres surface
        P_ref = 10e-5     # Pa on earth surface
        deltaT = T_local - T_ref
        deltaP = P_local - P_ref
        K = layer["K"]
        rho_0 = layer["rho_0"]
        alpha = layer["alpha"]
        rho_new[i] = rho_0 * (1 - alpha * deltaT + deltaP / K)
    return rho_new

def iterate_until_convergence(r, layers, T_core, max_iter=50, tol=1):
    rho_profile = get_density_profile(r, layers)
    for iteration in range(max_iter):
        T, T_boundaries = conductive_temp(r, layers, T_surface, T_core)
        M, g = upward_integration(r, del_r, rho_profile)
        P, P_boundaries = downward_integration(r, g, del_r, rho_profile, layers)
        Ra_results = calculate_rayleigh_numbers(r, g, T, layers)
        for name, Ra in Ra_results:
            layer = next(l for l in layers if l["name"] == name)
            if Ra > 1000:
                T = convective_layer_profile(Ra, r, T, g, layer, del_r)
        rho_new = update_density_profile(r, T, P, layers, T_boundaries, P_boundaries)
        delta_rho = np.max(np.abs(rho_new - rho_profile))
        print(f"Iteration {iteration+1}: Max Δρ = {delta_rho:.4f} kg/m³")
        if delta_rho < tol:
            print("Convergence reached.")
            break
        rho_profile = rho_new.copy()
    return r, T, P, rho_profile, g, M

def moi_integral(r_array, R, rho_array, M_total):
    rho_func = interp1d(r_array, rho_array, kind='linear', fill_value="extrapolate")
    integrand = lambda r: (8 / 3) * const.pi * r ** 4 * rho_func(r)
    I, _ = quad(integrand, 0, R)
    I_normal = I / (M_total * R ** 2)
    print(f"Moment of Inertia Factor: {I_normal:.4f}")
    return I_normal

def run_all_models():
    plt.rcParams['text.usetex'] = True
    plt.rcParams["font.family"] = 'serif'
    plt.rcParams['font.size'] = 16  # Set default font size
    fig_T, ax_T = plt.subplots(figsize=(8, 6))
    fig_MGP, axs = plt.subplots(1, 3, figsize=(15, 6), sharey=True)
    fig_rho, ax_rho = plt.subplots(figsize=(8, 6))

    for T_core in T_core_vals:
        print(f"\nRunning model for core temperature: {T_core} K")
        layers = define_layers(T_core)
        r = np.linspace(0, R, int(R / del_r) + 1)
        r, T, P, rho_final, g, M = iterate_until_convergence(r, layers, T_core)
        final_mass = M[-1]
        print(f"Final mass: {final_mass:.3e} kg")
        I_normal = moi_integral(r, R, rho_final, final_mass)

        # Optional: Get boundaries if needed
        T, T_boundaries = conductive_temp(r, layers, T_surface, T_core)
        P, P_boundaries = downward_integration(r, g, del_r, rho_final, layers)

        # Radius in km
        r_km = r / 1000

        # Temperature plot
        ax_T.plot(T, r_km, label=f"T_core = {T_core} K")


        # Mass / Gravity / Pressure
        axs[0].plot(M / 1e21, r_km, label=f"T_core = {T_core} K")
        axs[1].plot(g, r_km, label=f"T_core = {T_core} K")
        axs[2].plot(P / 1e9, r_km, label=f"T_core = {T_core} K")

        

        # Density plot
        ax_rho.plot(rho_final, r_km, label=f"T_core = {T_core} K")

    #Boundary labels
    for layer in define_layers(T_core):  # skip last since it's the surface
        boundary_km = layer['r_min'] / 1000
        name = layer['name'].capitalize() + " base"
        ax_T.axhline(y=boundary_km, color='gray', linestyle='--', linewidth=1)
        ax_T.text(x=ax_T.get_xlim()[1] * 0.98,y=boundary_km + 5,s=name,verticalalignment='bottom',
            horizontalalignment='right', fontsize=14,color='gray')
        
        axs[0].axhline(y=boundary_km, color='gray', linestyle='--', linewidth=1)
        axs[0].text(x=axs[0].get_xlim()[1] * 0.98,y=boundary_km + 5,s=name,verticalalignment='bottom',
            horizontalalignment='right', fontsize=14,color='gray')
        
        axs[1].axhline(y=boundary_km, color='gray', linestyle='--', linewidth=1)
        axs[1].text(x=axs[1].get_xlim()[1] * 0.98,y=boundary_km + 5,s=name,verticalalignment='bottom',
            horizontalalignment='right', fontsize=14,color='gray')
        
        axs[2].axhline(y=boundary_km, color='gray', linestyle='--', linewidth=1)
        axs[2].text(x=axs[2].get_xlim()[1] * 0.98,y=boundary_km + 5,s=name,verticalalignment='bottom',
            horizontalalignment='right', fontsize=14,color='gray')
        
        ax_rho.axhline(y=boundary_km, color='gray', linestyle='--', linewidth=1)
        ax_rho.text(x=ax_rho.get_xlim()[1] * 0.98,y=boundary_km + 5,s=name,verticalalignment='bottom',
            horizontalalignment='right', fontsize=14,color='gray')

    # Temperature
    ax_T.set_xlabel("Temperature (K)")
    ax_T.set_ylabel("Radius (km)")
    ax_T.grid(True)
    ax_T.legend()
    fig_T.tight_layout()

    # MGP
    labels = ["Mass [$10^{21}$ kg]", "Gravity [m/s$^2$]", "Pressure [GPa]"]
    for i, ax in enumerate(axs):
        ax.set_xlabel(labels[i])
        if i == 0:
            ax.set_ylabel("Radius (km)")
        ax.grid(True)
        ax.legend()
    fig_MGP.tight_layout()

    # Density
    ax_rho.set_xlabel("Density (kg/m$^3$)")
    ax_rho.set_ylabel("Radius (km)")
    ax_rho.grid(True)
    ax_rho.legend()
    fig_rho.tight_layout()

    plt.show()



if __name__ == "__main__":
    run_all_models()
