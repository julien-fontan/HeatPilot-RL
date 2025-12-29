import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import config
from mpl_toolkits.mplot3d import Axes3D

# ==========================================
# 1. DÉFINITION DES PARAMÈTRES (CONSTANTES)
# ==========================================
# ====== Remplacement des constantes par les valeurs du config ======
rc = config.REWARD_PARAMS
WEIGHTS = rc.get("weights", {})
PARAMS = rc.get("params", {})

P_REF = PARAMS.get("p_ref", 2000.0)               # kW
P_PUMP_NOMINAL = PARAMS.get("p_pump_nominal", 15.0)  # kW

COEFF_A = WEIGHTS.get("comfort", 10)   # weight for comfort
COEFF_B = WEIGHTS.get("boiler", 20)    # weight for boiler (excess temp)
COEFF_C = WEIGHTS.get("pump", 1)       # weight for pump

MIN_RETURN_TEMP = config.SIMULATION_PARAMS.get("min_return_temp", 40.0)  # °C

# Valeurs par défaut (initiales)
P_DEMAND_VAL = P_REF
P_PUMP_VAL = P_PUMP_NOMINAL
EXCESS_TEMP_VAL = 2.0   # °C d'excès moyen aux nœuds terminaux (proxy)

# Paramètres de mise à l'échelle
SCALE = 1.0

# ==========================================
# 2. GÉNÉRATION DES DONNÉES INITIALES
# ==========================================
# On fait varier la Puissance Fournie (P_supplied) de 0 à 4000 kW
# L'abscisse représente P_supplied.
p_supplied_vals = np.linspace(0, 4000, 1000)

def compute_rewards(w_comfort, p_ref, w_boiler, w_pump, p_pump_nom, p_demand, p_pump_val, excess_temp, p_supplied_vals):
    """
    Calcule les composantes de la reward en suivant la logique de _compute_reward de l'environnement.
    - p_* en kW, excess_temp en °C (moyenne sur nœuds terminaux)
    - p_supplied_vals: array en kW
    """
    # --- Terme 1 : Confort (pénalité sous-production) ---
    under_supply = np.maximum(0.0, p_demand - p_supplied_vals)   # kW
    r_confort = -w_comfort * (under_supply / p_ref)              # shape = p_supplied_vals

    # --- Terme 2 : Sobriété prod (ici on utilise excess_temp proxy) ---
    # env : r_prod = -w_boiler * excess_temp / min_return_temp
    r_boiler_val = -w_boiler * (excess_temp / MIN_RETURN_TEMP)
    r_boiler = np.full_like(p_supplied_vals, r_boiler_val)

    # --- Terme 3 : Pompe ---
    diff_pump = (p_pump_val - p_pump_nom) / p_pump_nom
    term_quad = diff_pump**2
    term_lin = np.maximum(0.0, diff_pump)
    r_pump_val = -w_pump * (term_quad + term_lin)
    r_pump = np.full_like(p_supplied_vals, r_pump_val)

    # --- Total ---
    r_total = r_confort + r_boiler + r_pump

    return r_confort, r_boiler, r_pump, r_total


def plot_3d_reward(w_comfort, p_ref, w_boiler, w_pump, p_pump_nom, p_demand, p_supplied_vals,
                   p_pump_vals=None, excess_temp=EXCESS_TEMP_VAL):
    """
    Crée un plot 3D (surface) du reward en fonction de P_supplied (x)
    et de P_pump (y). Le reward (z) est la reward totale. L'excess_temp
    est un paramètre fixe (contrôlable par le slider `s_pexcess`).
    Retourne (fig3, ax3, p_pump_vals) pour pouvoir rafraîchir la surface.
    """
    if p_pump_vals is None:
        p_pump_vals = np.linspace(5.0, 30.0, 50)

    P_sup_grid, P_pump_grid = np.meshgrid(p_supplied_vals, p_pump_vals)
    R = np.zeros_like(P_sup_grid)

    # Calculer la reward totale pour chaque valeur de P_pump
    for i, pp in enumerate(p_pump_vals):
        _, _, _, r_total_row = compute_rewards(
            w_comfort, p_ref, w_boiler, w_pump, p_pump_nom, p_demand, pp, excess_temp, p_supplied_vals
        )
        R[i, :] = r_total_row

    fig3 = plt.figure(figsize=(10, 6))
    ax3 = fig3.add_subplot(111, projection='3d')
    surf = ax3.plot_surface(P_sup_grid, P_pump_grid, R, cmap='viridis', edgecolor='none', linewidth=0, antialiased=True)
    ax3.set_xlabel('P_supplied (kW)')
    ax3.set_ylabel('P_pump (kW)')
    ax3.set_zlabel('Reward')
    ax3.set_title('Reward surface: P_supplied vs P_pump (Excess Temp={:.1f}°C)'.format(excess_temp))
    fig3.colorbar(surf, ax=ax3, shrink=0.5, aspect=10)

    return fig3, ax3, p_pump_vals

# Recompute initial curves with new compute_rewards
r_confort, r_boiler, r_pump, r_total = compute_rewards(
    COEFF_A,
    P_REF,
    COEFF_B,
    COEFF_C,
    P_PUMP_NOMINAL,
    P_DEMAND_VAL,
    P_PUMP_VAL,
    EXCESS_TEMP_VAL,
    p_supplied_vals
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
s_pexcess = Slider(ax_pexcess, r'Excess Temp (°C)', 0.0, 50.0, valinit=EXCESS_TEMP_VAL)
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
    excess_temp_val = s_pexcess.val

    # Recalcule selon la nouvelle fonction
    n_confort, n_boiler, n_pump, n_total = compute_rewards(
        a, p_ref_val, b, c, P_PUMP_NOMINAL, p_dem_val, pp, excess_temp_val, p_supplied_vals
    )

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

# Affiche aussi le plot 3D (reward en fonction de P_supplied et P_pump)
# On crée la surface initiale et on garde `ax3` et `p_pump_vals` pour mises à jour
fig3, ax3, p_pump_vals = plot_3d_reward(
    COEFF_A, P_REF, COEFF_B, COEFF_C, P_PUMP_NOMINAL, P_DEMAND_VAL, p_supplied_vals
)

def _update_3d_surface(ax, p_pump_vals, w_comfort, p_ref, w_boiler, w_pump, p_pump_nom, p_demand, p_supplied_vals, excess_temp):
    # Reconstruit et redessine la surface 3D (efface l'ancien contenu)
    P_sup_grid, P_pump_grid = np.meshgrid(p_supplied_vals, p_pump_vals)
    R = np.zeros_like(P_sup_grid)
    for i, pp in enumerate(p_pump_vals):
        _, _, _, r_total_row = compute_rewards(
            w_comfort, p_ref, w_boiler, w_pump, p_pump_nom, p_demand, pp, excess_temp, p_supplied_vals
        )
        R[i, :] = r_total_row

    ax.cla()
    surf = ax.plot_surface(P_sup_grid, P_pump_grid, R, cmap='viridis', edgecolor='none', linewidth=0, antialiased=True)
    ax.set_xlabel('P_supplied (kW)')
    ax.set_ylabel('P_pump (kW)')
    ax.set_zlabel('Reward')
    ax.set_title('Reward surface: P_supplied vs P_pump (Excess Temp={:.1f}°C)'.format(excess_temp))
    return surf


def update(val):
    sc = s_scale.val

    a = sc * s_coeff_a.val
    b = sc * s_coeff_b.val
    c = sc * s_coeff_c.val

    p_ref_val = s_pref.val
    p_dem_val = s_pdem.val
    pp = s_ppump.val
    excess_temp_val = s_pexcess.val

    # Recalcule selon la nouvelle fonction
    n_confort, n_boiler, n_pump, n_total = compute_rewards(
        a, p_ref_val, b, c, P_PUMP_NOMINAL, p_dem_val, pp, excess_temp_val, p_supplied_vals
    )

    l_confort.set_ydata(n_confort)
    l_boiler.set_ydata(n_boiler)
    l_pump.set_ydata(n_pump)
    l_total.set_ydata(n_total)

    v_line_dem.set_xdata([p_dem_val, p_dem_val])

    ax.set_title(f"Reward Global (Scale={sc:.1f})")

    # Mise à jour du plot 3D (surface en P_pump)
    _update_3d_surface(ax3, p_pump_vals, a, p_ref_val, b, c, P_PUMP_NOMINAL, p_dem_val, p_supplied_vals, excess_temp_val)

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