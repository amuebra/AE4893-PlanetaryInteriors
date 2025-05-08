import numpy as np
import matplotlib.pyplot as plt
from burnman import Composite, Layer, Mineral, PerplexMaterial, Composite, Planet
from burnman import minerals
import custom_minerals as cus

# Load mineral endmembers
olivine = minerals.HP_2011_ds62.fo()
salt = minerals.HP_2011_ds62.hlt()
hydrated_silicated = minerals.HP_2011_ds62.hcrd()

print(f"Molar mass of olivine: {olivine.params['molar_mass']}")
print(f"Molar mass of hydrated silicates: {hydrated_silicated.params['molar_mass']}")
ice = cus.ice
ammoniumChlorid = cus.ammonium_chloride
clathrates=cus.methane_clathrate
carbonate = cus.carbonate
print(cus.ice.params['k_0'])


#fo_HP.set_state(4.293e7, 191)
#print(fo_HP.params['V_0'])
#density = fo_HP.params['molar_mass']/fo_HP.params['V_0']
#print(density)

# ----------------------Define composition-----------------------
crust_material = Composite(
    [ice, hydrated_silicated,carbonate, salt, ammoniumChlorid, clathrates ],
    [0.25, 0.10, 0.19, 0.136, 0.064, 0.26]
 )
mantle_material = Composite(
    [ice, hydrated_silicated, olivine],
    [0.77, 0.13, 0.1]
)
core_material = Composite(
    [hydrated_silicated, olivine],
    [0.926, 0.074]
)

# -------------------------------------------------------
# 3. Temperature Setup
# -------------------------------------------------------

# Temperaturen an den Grenzflächen
T_surface = 163.
T_crust_bottom = 180.
T_mantle_bottom = 202.
T_core_center = 278.

# Radien in m
R = 470e3
crust_thickness = 44e3
core_radius = 350e3  # 350 km
mantle_thickness = R-crust_thickness-core_radius

# Radienbereiche
r_core = np.linspace(0, core_radius, 300)
r_mantle = np.linspace(core_radius, core_radius + mantle_thickness, 300)
r_crust = np.linspace(core_radius + mantle_thickness, R, 200)

# Temperaturprofile (linear)
T_core = T_core_center + (T_mantle_bottom - T_core_center) * (r_core - 0) / (core_radius - 0)
T_mantle = T_mantle_bottom + (T_crust_bottom - T_mantle_bottom) * (r_mantle - core_radius) / mantle_thickness
T_crust = T_crust_bottom + (T_surface - T_crust_bottom) * (r_crust - (core_radius + mantle_thickness)) / crust_thickness
print(len(r_core))
print(len(T_core))
# Speichern für BurnMan
layers_data = [
    {
        "name": "core",
        "r_min": 0.0,
        "r_max": core_radius,
        "radii": r_core,
        "temperatures": T_core
    },
    {
        "name": "mantle",
        "r_min": core_radius,
        "r_max": core_radius + mantle_thickness,
        "radii": r_mantle,
        "temperatures": T_mantle
    },
    {
        "name": "crust",
        "r_min": core_radius + mantle_thickness,
        "r_max": R,
        "radii": r_crust,
        "temperatures": T_crust
    }
]



# -------------------------------------------------------
# 4. Build BurnMan Planet Layers
# -------------------------------------------------------

core = Layer(name='core', radii=layers_data[0]["radii"])
core.set_material(core_material)
core.set_temperature_mode(temperature_mode='user-defined', temperatures=T_core)
core.set_pressure_mode(pressure_mode='self-consistent', pressure_top=0.07e9, gravity_bottom=0)
core.make()


mantle = Layer(name='mantle', radii=layers_data[1]["radii"])
mantle.set_material(mantle_material)
mantle.set_temperature_mode(temperature_mode='user-defined', temperatures=T_mantle)
mantle.set_pressure_mode(pressure_mode='self-consistent', pressure_top=0.0169e9, gravity_bottom=0.2410)
mantle.make()

crust = Layer(name='crust', radii=layers_data[2]["radii"])
crust.set_material(crust_material)
crust.set_temperature_mode(temperature_mode='user-defined', temperatures=T_crust)
crust.set_pressure_mode(pressure_mode='self-consistent', pressure_top=0, gravity_bottom=0.2893)
crust.make()

# -------------------------------------------------------
# 5. Build Planet
# -------------------------------------------------------

Ceres = Planet('Ceres', [core, mantle, crust], verbose=True)
Ceres.make()

print(f"Mass = {Ceres.mass:.5e}")
print(f"Moment of Inertia Factor = {Ceres.moment_of_inertia_factor:.4f}")
print('Layer mass fractions:')
for layer in Ceres.layers:
    print(f'{layer.name}: {layer.mass / Ceres.mass:.3f}')


#-------------------------plot--------------------------------------------------------
plt.rcParams['text.usetex'] = True
plt.rcParams["font.family"] = 'serif'
plt.rcParams['font.size'] = 12  # Set default font size

fig = plt.figure(figsize=(8, 5))
ax = [fig.add_subplot(2, 2, i) for i in range(1, 5)]

bounds = np.array([[layer.radii[0]/1.e3, layer.radii[-1]/1.e3]
                   for layer in Ceres.layers])
maxy = [5, 1, 0.5, 300]
for bound, layer in zip(bounds, Ceres.layers):
    for i in range(4):
        ax[i].fill_betweenx([0., maxy[i]],
                            [bound[0], bound[0]],
                            [bound[1], bound[1]], alpha=0.2)
        ax[i].axvline(x=bound[0], color='gray', linestyle='--', linewidth=1)
        ax[i].text(y=ax[i].get_ylim()[1] * 0.9,x=bound[0] + 5,s=layer.name,verticalalignment='top',
            horizontalalignment='left', fontsize=14,color='gray', rotation=90)

ax[0].plot(Ceres.radii / 1.e3, Ceres.density / 1.e3)
ax[0].set_ylabel('Density ($10^3$ kg/m$^3$)')
# ax[0].legend()

# Make a subplot showing the calculated pressure profile
ax[1].plot(Ceres.radii / 1.e3, Ceres.pressure / 1.e9)
ax[1].set_ylabel('Pressure (GPa)')

# Make a subplot showing the calculated gravity profile
ax[2].plot(Ceres.radii / 1.e3, Ceres.gravity)
ax[2].set_ylabel('Gravity (m/s$^2)$')
ax[2].set_xlabel('Radius (km)')

# Make a subplot showing the calculated temperature profile
ax[3].plot(Ceres.radii / 1.e3, Ceres.temperature)
ax[3].set_ylabel('Temperature (K)')
ax[3].set_xlabel('Radius (km)')
ax[3].set_ylim(0.,)

fig.set_layout_engine('tight')
plt.show()