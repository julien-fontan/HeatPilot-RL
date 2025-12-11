import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# ==========================================
# 1. DÉFINITION DES PARAMÈTRES (CONSTANTES)
# ==========================================
P_REF = 2000.0       # Puissance de référence (kW)

# Paramètres de mise à l'échelle
SCALE = 1.0

# Coefficients simplifiés (a, b, c)
COEFF_A = 1.0    # Confort
COEFF_B = 1.0    # Sobriété Boiler
COEFF_C = 0.5     # Sobriété Pompage

# Paramètres Pompe
P_PUMP_NOMINAL = 15.0 # kW

# Valeurs par défaut
P_DEMAND_VAL = 2000.0    # kW
P_PUMP_VAL = 15.0        # kW
P_EXCESS_VAL = 50.0      # kW (P_boiler - P_demand)

# ==========================================
# 2. GÉNÉRATION DES DONNÉES INITIALES
# ==========================================
# On fait varier la Puissance Fournie (P_supplied) de 0 à 4000 kW
# L'abscisse représente P_supplied.
p_supplied_vals = np.linspace(0, 4000, 1000)

def compute_rewards(a, p_ref, b, c, p_pump_nom, p_demand, p_pump_val, p_excess_val):
    # --- Terme 1 : Confort (Quadratique) ---
    # e = P_demand - P_supplied
    # On ne pénalise que la sous-production (P_supplied < P_demand)
    under_supply = np.maximum(0, p_demand - p_supplied_vals)
    r_confort = -a * (under_supply / p_ref)
    
    # --- Terme 2 : Sobriété Boiler ---
    # Pénalise si P_boiler > P_demand
    # P_excess est fixé par le slider (indépendant de P_supplied sur ce graphe)
    val_boiler = -b * np.maximum(0, p_excess_val / p_ref)
    r_boiler = np.full_like(p_supplied_vals, val_boiler)

    # --- Terme 3 : Sobriété Pompage ---
    # Reward au carré centré sur la valeur nominale
    # + pénalisation linéaire si P_pump > P_nom
    
    diff_norm = (p_pump_val - p_pump_nom) / p_pump_nom
    
    term_quad = diff_norm**2
    term_lin = np.maximum(0, diff_norm)
    
    val_pump = -c * (term_quad + term_lin)
    r_pump = np.full_like(p_supplied_vals, val_pump)

    # --- Total ---
    r_total = r_confort + r_boiler + r_pump
    
    return r_confort, r_boiler, r_pump, r_total

r_confort, r_boiler, r_pump, r_total = compute_rewards(
    SCALE * COEFF_A, 
    P_REF, 
    SCALE * COEFF_B, 
    SCALE * COEFF_C, 
    P_PUMP_NOMINAL, 
    P_DEMAND_VAL, 
    P_PUMP_VAL, 
    P_EXCESS_VAL
)

# ==========================================
# 3. TRACÉ DU GRAPHIQUE
# ==========================================
fig, ax = plt.subplots(figsize=(12, 10))
plt.subplots_adjust(left=0.1, bottom=0.50, right=0.95, top=0.85)

# Tracé des composantes
l_confort, = ax.plot(p_supplied_vals, r_confort, label='Confort (Quadratique)', 
         color='blue', linewidth=2, alpha=0.8)

l_boiler, = ax.plot(p_supplied_vals, r_boiler, label='Sobriété: Boiler (Offset)',
         linestyle='-', color='brown', alpha=0.5)

l_pump, = ax.plot(p_supplied_vals, r_pump, label='Sobriété: Pompage (Offset)',
         linestyle='-', color='purple', alpha=0.5)

l_total, = ax.plot(p_supplied_vals, r_total, label='REWARD GLOBAL', 
         color='black', linewidth=3)

# Ligne verticale P_demand
v_line_dem = ax.axvline(x=P_DEMAND_VAL, color='red', linestyle=':', label='P_demand')

# Affichage de la formule
formula_text = (
    r"$r(t) = - a \left(\frac{\max(0, P_{dem} - P_{sup})}{P_{ref}}\right) "
    r"- b \max\left(0, \frac{P_{boiler} - P_{dem}}{P_{ref}}\right)$"
    "\n"
    r"$ - c \left[ \left(\frac{P_{pump}-P_{nom}}{P_{nom}}\right)^2 + \max\left(0, \frac{P_{pump}-P_{nom}}{P_{nom}}\right) \right]$"
)
fig.text(0.5, 0.92, formula_text, ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

# Mise en forme
ax.set_title(f"Évolution de la Pénalité Globale")
ax.set_xlabel("Puissance Fournie P_supplied (kW)")
ax.set_ylabel("Valeur de la Reward")
ax.grid(True, which='both', linestyle='--', alpha=0.7)
ax.legend(loc='lower left')

# ==========================================
# 4. AJOUT DES SLIDERS
# ==========================================

ax_scale = plt.axes([0.1, 0.42, 0.80, 0.03])

# Colonne 1
ax_a     = plt.axes([0.1, 0.35, 0.35, 0.02])
ax_pref  = plt.axes([0.1, 0.30, 0.35, 0.02])
ax_pdem  = plt.axes([0.1, 0.25, 0.35, 0.02]) 

# Colonne 2
ax_b     = plt.axes([0.60, 0.35, 0.30, 0.02])
ax_c     = plt.axes([0.60, 0.30, 0.30, 0.02])
ax_ppump = plt.axes([0.60, 0.25, 0.30, 0.02])
ax_pexcess = plt.axes([0.60, 0.20, 0.30, 0.02]) # Remplacé P_boiler par P_excess

s_scale = Slider(ax_scale, 'Scale', 0.1, 2.0, valinit=SCALE)

s_coeff_a = Slider(ax_a, r'Coeff $a$', 0.0, 20.0, valinit=COEFF_A)
s_pref  = Slider(ax_pref, r'$P_{ref}$ (kW)', 500.0, 5000.0, valinit=P_REF)
s_pdem  = Slider(ax_pdem, r'$P_{dem}$ (kW)', 500.0, 3500.0, valinit=P_DEMAND_VAL)

s_coeff_b = Slider(ax_b, r'Coeff $b$', 0.0, 20.0, valinit=COEFF_B)
s_pexcess = Slider(ax_pexcess, r'$P_{boiler}-P_{dem}$', 0.0, 1000.0, valinit=P_EXCESS_VAL)
s_coeff_c = Slider(ax_c, r'Coeff $c$', 0.0, 20.0, valinit=COEFF_C)
s_ppump = Slider(ax_ppump, r'$P_{pump}$ (kW)', 5.0, 30.0, valinit=P_PUMP_VAL)

def update(val):
    sc = s_scale.val
    
    a = sc * s_coeff_a.val
    b = sc * s_coeff_b.val
    c = sc * s_coeff_c.val
    
    p_ref_val = s_pref.val
    p_dem_val = s_pdem.val
    pp = s_ppump.val
    p_excess_val = s_pexcess.val
    
    res = compute_rewards(a, p_ref_val, b, c, P_PUMP_NOMINAL, p_dem_val, pp, p_excess_val)
    n_confort, n_boiler, n_pump, n_total = res
    
    l_confort.set_ydata(n_confort)
    l_boiler.set_ydata(n_boiler)
    l_pump.set_ydata(n_pump)
    l_total.set_ydata(n_total)
    
    v_line_dem.set_xdata([p_dem_val, p_dem_val])
    
    ax.set_title(f"Reward Global (Scale={sc:.1f})")
    fig.canvas.draw_idle()

s_scale.on_changed(update)
s_coeff_a.on_changed(update)
s_pref.on_changed(update)
s_pdem.on_changed(update)
s_coeff_b.on_changed(update)
s_coeff_c.on_changed(update)
s_ppump.on_changed(update)
s_pexcess.on_changed(update)

plt.show()