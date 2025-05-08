import burnman
from burnman.utils.chemistry import dictionarize_formula, formula_mass
import burnman.minerals as minerals
import burnman.minerals.SLB_2024 as SLB_2024
print(dir(SLB_2024))

ice_formula = dictionarize_formula('H2O')
molar_mass = formula_mass(ice_formula)
ice_params = {
    'name': 'ice_Ih',
    'formula': ice_formula,
    'equation_of_state': 'hp_tmt',
    'H_0': -285830.0,       # Enthalpy of formation at 298.15 K and 1 bar [J/mol]
    'S_0': 41.0,            # Entropy at 298.15 K [J/mol·K], approximate
    'V_0': 1.962e-05,       # Molar volume [m³/mol] (approx 18.07 cm³/mol)
    'Cp': [75.3, 0.0, 0.0, 0.0],  # Heat capacity [J/mol·K], roughly constant below 273K
    'a_0': 5e-5,            # Thermal expansivity [1/K], approximate
    'k_0': 2.5,             # Thermal conductivity W/m*K
    'K_0': 9.0e9,           # Bulk modulus [Pa] for ice Ih, low value due to softness
    'Kprime_0': 5.1,        # Pressure derivative of K
    'Kdprime_0': 0.0,       # Second pressure derivative (usually small or unknown)
    'n': 5.0,               # Number of atoms per formula unit
    'molar_mass': molar_mass  # Computed above
}
ice = burnman.Mineral(params=ice_params)
# Custom mineral: Muscovite
muscovite_formula = {'K': 1.0, 'Al': 3.0, 'Si': 3.0, 'O': 12.0, 'H': 2.0}
molar_mass = formula_mass(muscovite_formula)
muscovite_params = {
    'name': 'muscovite',
    'formula': muscovite_formula,
    'equation_of_state': 'hp_tmt',
    'H_0': -4082000.0,
    'S_0': 202.0,
    'V_0': 1.514e-04,
    'Cp': [270.0, 0.0, 0.0, 0.0],
    'a_0': 2.5e-5,
    'k_0': 2.0,
    'K_0': 50000000.0,
    'Kprime_0': 4.0,
    'Kdprime_0': 0.0,
    'n': 18.0,
    'molar_mass': molar_mass
}
muscovite = burnman.Mineral(params=muscovite_params)

# Custom mineral: Hydrohalite
hydrohalite_formula = dictionarize_formula('NaCl2H2O')
molar_mass = formula_mass(hydrohalite_formula)
hydrohalite_params = {
    'name': 'hydrohalite',
    'formula': hydrohalite_formula,
    'equation_of_state': 'hp_tmt',
    'H_0': -993000.0,
    'S_0': 143.0,
    'V_0': 5.072e-5,
    'Cp': [115.0, 0.0, 0.0, 0.0],
    'a_0': 4.0e-5,
    'k_0': 0.6,
    'K_0': 2.0e10,
    'Kprime_0': 4.0,
    'Kdprime_0': 0.0,
    'n': 7.0,
    'molar_mass': molar_mass
}
hydrohalite = burnman.Mineral(params=hydrohalite_params)

# Custom mineral: Ammonium Chloride (NH4Cl)
nh4cl_formula = dictionarize_formula('NH4Cl')
molar_mass = formula_mass(nh4cl_formula)
nh4cl_params = {
    'name': 'ammonium_chloride',
    'formula': nh4cl_formula,
    'equation_of_state': 'hp_tmt',
    'H_0': -314300.0,
    'S_0': 94.6,
    'V_0': 3.366e-5,
    'Cp': [96.0, 0.0, 0.0, 0.0],
    'a_0': 4.2e-5,
    'k_0': 0.5,
    'K_0': 2.4e10,
    'Kprime_0': 4.0,
    'Kdprime_0': 0.0,
    'n': 6.0,
    'molar_mass': molar_mass
}
ammonium_chloride = burnman.Mineral(params=nh4cl_params)

# Custom mineral: Methane Clathrate (CH₄·5.75H₂O)
clathrate_formula = {'C': 1.0, 'H': 15.5, 'O': 5.75}
molar_mass = formula_mass(clathrate_formula)
clathrate_params = {
    'name': 'methane_clathrate',
    'formula': clathrate_formula,
    'equation_of_state': 'hp_tmt',
    'H_0': -559000.0,
    'S_0': 260.0,
    'V_0': 1.75e-4,
    'Cp': [140.0, 0.0, 0.0, 0.0],
    'a_0': 4.0e-5,
    'k_0': 0.5,
    'K_0': 7.0e9,
    'Kprime_0': 4.0,
    'Kdprime_0': 0.0,
    'n': 23.0,
    'molar_mass': molar_mass
}
methane_clathrate = burnman.Mineral(params=clathrate_params)

# Define the chemical formula for calcite
carbonate_formula = dictionarize_formula('CaCO3')
molar_mass = formula_mass(carbonate_formula)

# BurnMan parameters for calcite (CaCO₃)
carbonate_params = {
    'name': 'calcite',
    'formula': carbonate_formula,
    'equation_of_state': 'hp_tmt',
    'H_0': -1207000.0,      # Enthalpy of formation [J/mol]
    'S_0': 92.9,            # Entropy [J/mol·K]
    'V_0': 3.693e-5,        # Molar volume [m³/mol], ~36.93 cm³/mol
    'Cp': [82.5, 0.0, 0.0, 0.0],  # Heat capacity [J/mol·K], approximate
    'a_0': 2.5e-5,          # Thermal expansivity [1/K], approximate
    'k_0': 3.0,             # Thermal conductivity [W/m·K], rough estimate
    'K_0': 7.0e10,          # Bulk modulus [Pa], ~70 GPa for calcite
    'Kprime_0': 5.0,        # First pressure derivative of bulk modulus
    'Kdprime_0': 0.0,       # Second pressure derivative (if known)
    'n': 5.0,               # Number of atoms per formula unit (Ca + C + 3O)
    'molar_mass': molar_mass
}

# Create the Mineral object
carbonate = burnman.Mineral(params=carbonate_params)
# Export them
__all__ = ['ice', 'muscovite', 'hydrohalite', 'ammonium_chloride', 'methane_clathrate', 'carbonate']