import numpy as np
import scipy.constants as const
from scipy.integrate import quad
import matplotlib.pyplot as plt

# ----------------------- Constants & Layer Setup -----------------------
G = const.G
R_km = 470
R = R_km * 1000  # Convert km to meters
del_r = 1  # Step size in meters
crust_thickness = 44e3      #m
crust_density = 1360        #kg/m^3

core_thickness = 350e3
core_density = 2465.35

mantle_density = 2389
mantle_thickness = R-crust_thickness-core_thickness

r_surface = R
r_crust_base = r_surface - crust_thickness
r_mantle_base = r_crust_base - mantle_thickness
r_core_center = 0

# Define density layers (core, mantle, crust)
layers = [
    (0, r_mantle_base, lambda r: core_density),         # Core,
    (r_mantle_base, r_crust_base, lambda r: mantle_density),# mantle
    (r_crust_base, R, lambda r: crust_density)          # Crust
]

# ----------------------- Density Model Function -----------------------
def get_density(radius):
    for r_start, r_end, rho_func in layers:
        if r_start <= radius <= r_end:
            return rho_func(radius)
    raise ValueError(f"Radius {radius} is outside all defined layers!")

# ----------------------- Upward Integration -----------------------
def upward_integration(r, del_r):
    M = np.zeros_like(r)
    g = np.zeros_like(r)
    for i in range(1, int(R / del_r) + 1):
        rho_i = get_density(r[i - 1])
        dMdr = 4 * const.pi * rho_i * r[i - 1] ** 2
        M[i] = M[i - 1] + dMdr * del_r
    for i in range(1, int(R / del_r) + 1):
        if r[i] != 0:
            g[i] = G * M[i] / r[i] ** 2
    return M, g

# ----------------------- Downward Integration -----------------------
def downward_integration(r, g, del_r):
    P = np.zeros_like(r)
    P[-1] = 0  # Surface pressure
    for i in range(int(R / del_r) , 0, -1):
        rho_i = get_density(r[i])
        P[i-1] = P[i] + rho_i * g[i] * del_r
    return P

# ----------------------- Moment of Inertia -----------------------
def moi_integral(r1, r2, rho_func):
    integrand = lambda r: (8/3) * const.pi * r**4 * rho_func(r)
    I, _ = quad(integrand, r1, r2)
    return I

# ----------------------- Run the Model -----------------------
def run_model():
    r = np.linspace(0, R, int(R / del_r) + 1)  # Radius array
    M, g = upward_integration(r, del_r)
    P = downward_integration(r, g, del_r)

    # Analytical calculations
    print(get_density(R))
    M_ana = get_density(R)*(4/3) * const.pi * R**3 # only for homogenous body
    print(M_ana)
    P_ana = (2/3) * const.pi * G * get_density(R)**2 * R**2

    # Moment of inertia
    I_total = sum(moi_integral(r1, r2, rho_func) for r1, r2, rho_func in layers)
    I_normal = I_total / (M[-1] * R**2)

    # Errors
    mass_diff = M_ana - M[-1]
    pressure_diff = P_ana - P[0]
    mass_percent_error = (mass_diff / M_ana) * 100
    pressure_percent_error = (pressure_diff / P_ana) * 100

    # Output
    print(f"Total numerical mass: {M[-1]:.6e} kg")
    #print(f"Analytical mass: {M_ana:.4e} kg")
    #print(f"Difference in mass: {mass_diff:.4e} kg ({mass_percent_error:.6f}%)")
    print("----------------------------------------------------")
    #print(f"Analytical central pressure:  {P_ana:.4e} Pa")
    print(f"Numerical central pressure:   {P[0]:.4e} Pa")
    #print(f"Pressure difference:          {pressure_diff:.4e} Pa ({pressure_percent_error:.6f}%)")
    #print(f"Total moment of inertia:      {I_total:.4e} kg·m²")
    print(f"Normalized moment of inertia: {I_normal:.4f}")

    # ----------------------- Plotting -----------------------
    # Enable LaTeX rendering
    plt.rcParams['text.usetex'] = True
    plt.rcParams["font.family"] = 'serif'
    plt.rcParams['font.size'] = 16  # Set default font size

    fig, axs = plt.subplots(1, 3, figsize=(15, 6), sharey=True)
    r_km = r / 1000  # convert radius to km for plotting

    # Layer boundaries in km
    r_crust_base_km = r_crust_base / 1000
    r_mantle_base_km = r_mantle_base / 1000

    # Plot Mass
    axs[0].plot(M / 1e21, r_km, color='blue')
    axs[0].set_xlabel(r'Mass [$10^{21}$ kg]')
    axs[0].set_ylabel(r'Radius [km]')
    axs[0].grid()
    #axs[0].set_title("Mass")

    # Plot Gravity
    axs[1].plot(g, r_km, color='green')
    axs[1].set_xlabel(r'Gravity [m/s$^2$]')
    axs[1].grid()
    #axs[1].set_title("Gravity")

    # Plot Pressure
    axs[2].plot(P / 1e9, r_km, color='red')
    axs[2].set_xlabel(r'Pressure [GPa]')
    axs[2].grid()
    #axs[2].set_title("Pressure")

    # Draw and label layers
    for ax in axs:
        # Crust-mantle boundary
        ax.axhline(y=r_crust_base_km, color='darkgray', linestyle='--', linewidth=1)
        ax.text(ax.get_xlim()[1] * 0.95, r_crust_base_km + 2, 'Crust base',
                verticalalignment='bottom', horizontalalignment='right', fontsize=14, color='darkgray')

        # Mantle-core boundary
        ax.axhline(y=r_mantle_base_km, color='gray', linestyle='--', linewidth=1)
        ax.text(ax.get_xlim()[1] * 0.95, r_mantle_base_km + 2, 'Mantle base',
                verticalalignment='bottom', horizontalalignment='right', fontsize=14, color='gray')
        ax.text(ax.get_xlim()[1] * 0.95, 0 + 2, 'Core base', verticalalignment ='bottom',
                horizontalalignment='right', fontsize=14, color='black')

    #fig.suptitle(r"\textbf{Mass, Gravity, and Pressure vs Radius}", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


# ----------------------- Execute -----------------------
if __name__ == "__main__":
    run_model()
