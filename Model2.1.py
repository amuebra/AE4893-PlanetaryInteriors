import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const

# ---------------- Constants ----------------
T_surface = 163     # K
T_core_vals = [245, 278, 295]  # min, mean, max core temps
R = 470e3           # m
G = const.G         # Gravitational constant
del_r = 1          # m

# ---------------- Layers -------------------
def define_layers(T_core):
    return [
        {
            "name": "core",
            "r_min": 0,
            "r_max": 350e3,
            "rho_0": 2465.35,
            "k": 2.0,           # Thermal conductivity W/m*K
            "alpha": 3e-5,      # Thermal expansivity 1/K
            "eta": 1e21,        # viscosity Pa*s
            "Cp": 900,         # Specific Heat J/kg*K
            "K": 3e10,          # Bulk modulus
            #"T0": T_core
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
            #"T0": (T_surface + T_core) / 2
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
            #"T0": T_surface
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

# --------------- Temperature Functions ------------------
def conductive_temp(r, layers, T_surface, T_core):
    n_layers = len(layers)

    # Extract boundaries and conductivities
    radii = [layer["r_min"] for layer in layers] + [layers[-1]["r_max"]]
    ks = [layer["k"] for layer in layers]

    # Number of interfaces (n_layers + 1) including surface and center
    A = [ks[i] / (radii[i + 1] - radii[i]) for i in range(n_layers)]

    # Setup linear system for boundary temperatures
    # T[0] = T_core, T[-1] = T_surface
    T_boundaries = np.zeros(n_layers + 1)
    T_boundaries[0] = T_core
    T_boundaries[-1] = T_surface

    # System of n_layers - 1 equations for unknown intermediate temperatures
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

    # Solve system
    T_internal = np.linalg.solve(A_sys, b_sys)
    T_boundaries[1:-1] = T_internal

    # Interpolate temperature profile
    T_profile = np.zeros_like(r)
    for i in range(n_layers):
        r_start = radii[i]
        r_end = radii[i + 1]
        T_start = T_boundaries[i]
        T_end = T_boundaries[i + 1]

        mask = (r >= r_start) & (r <= r_end)
        T_profile[mask] = T_start + (T_end - T_start) * (r[mask] - r_start) / (r_end - r_start)

    return T_profile


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
        print(f"Rayleigh number for layer {layer['name']}: {Ra}")
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
    T_adiabat = T_avg

    for i, radius in enumerate(r[idx_top:idx_bot+1]):
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

# ---------------- Main Model ----------------
def run_model_for_core_temp(T_core):
    layers = define_layers(T_core)
    r = np.linspace(0, R, int(R / del_r) + 1)
    rho_profile = get_density_profile(r, layers)
    M, g = upward_integration(r, del_r, rho_profile)
    T_profile = conductive_temp(r, layers, T_surface, T_core)
    Ra_results = calculate_rayleigh_numbers(r, g, T_profile, layers)

    for (layer_name, Ra) in Ra_results:
        if Ra > 1000:
            layer = next(l for l in layers if l["name"] == layer_name)
            T_profile = convective_layer_profile(Ra, r, T_profile, g, layer, del_r)

    return r, T_profile

# ---------------- Execution ----------------
def main():
    profiles = []
    labels = []

    for T_core in T_core_vals:
        r, T_profile = run_model_for_core_temp(T_core)
        profiles.append(T_profile)
        labels.append(f"T_core = {T_core} K")
        print(f"current T_Core: {T_core}")

    # -------- Plotting --------
    # Enable LaTeX rendering
    plt.rcParams['text.usetex'] = True
    plt.rcParams["font.family"] = 'serif'
    plt.rcParams['font.size'] = 16  # Set default font size

    plt.figure(figsize=(8, 6))
    for T_profile, label in zip(profiles, labels):
        plt.plot(T_profile, r / 1000, label=label)

    plt.xlabel("Temperature (K)")
    plt.ylabel("Radius (km)")
    #plt.title("Temperature Profiles for Different Core Temperatures")
    plt.legend()
    plt.grid(True)

    # Dynamically add boundaries from layer definitions
    for layer in define_layers(T_core):  # skip last since it's the surface
        boundary_km = layer['r_min'] / 1000
        name = layer['name'].capitalize() + " base"
        plt.axhline(y=boundary_km, color='gray', linestyle='--', linewidth=1)
        plt.text(
            x=plt.xlim()[1] * 0.98,
            y=boundary_km + 5,
            s=name,
            verticalalignment='bottom',
            horizontalalignment='right',
            fontsize=14,
            color='gray'
        )

    # Optional: invert for depth view
    # plt.gca().invert_yaxis()

    plt.tight_layout()
    plt.show()

# ---------------- Run ----------------
if __name__ == "__main__":
    main()
