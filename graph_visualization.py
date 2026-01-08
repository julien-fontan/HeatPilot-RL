import os
import numpy as np
import matplotlib.pyplot as plt

class DistrictHeatingVisualizer:
    def __init__(self, plots_dir):
        self.plots_dir = plots_dir
        os.makedirs(self.plots_dir, exist_ok=True)

    def plot_dashboard_general(self, data, title_suffix=""):
        """
        Génère le dashboard global. 
        Détecte et trace automatiquement les 'split_node_X' sur un 3ème axe Y.
        """
        t_h = data["time"] / 3600.0
        
        # Identification dynamique des clés de splits
        split_keys = [k for k in data.keys() if k.startswith("split_")]
        
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        # Couleurs définies
        col_temp = '#D32F2F'   # Rouge
        col_flow = '#1976D2'   # Bleu
        col_split_base = ['#8E24AA', '#43A047', '#FF9800', '#795548'] # Violet, Vert, Orange, Marron
        
        col_dem = '#212121'
        col_sup = '#388E3C'
        col_wast = '#FBC02D'

        # --- GRAPHIQUE 1 : Source & Contrôles ---
        ax = axes[0]
        
        # Axe Y1 : Température
        l1, = ax.plot(t_h, data["T_source"], color=col_temp, label="Température", linewidth=1.5, zorder=3)
        ax.set_ylabel("Température (°C)")
        ax.tick_params(axis='y')
        
        # Axe Y2 : Débit (Twinx standard)
        ax2 = ax.twinx()
        l2, = ax2.plot(t_h, data["m_source"], color=col_flow, label="Débit", linewidth=1.5, zorder=2)
        ax2.set_ylabel("Débit (kg/s)")
        ax2.tick_params(axis='y')
        
        lines = [l1, l2]
        
        # Axe Y3 : Fractions Split (Si présentes dans data)
        if split_keys:
            ax3 = ax.twinx()
            # Décalage de l'axe vers la droite ("parasite axis")
            ax3.spines["right"].set_position(("axes", 1.15))
            ax3.set_frame_on(True)
            ax3.patch.set_visible(False)
            
            # Configuration visuelle du 3ème axe
            ax3.set_ylabel("Ouverture vannes (%)")
            ax3.set_ylim(0, 100) # Fixe pour des ratios [0,1]*100
            # ax3.tick_params(axis='y', colors="#555")
            
            # Tracé dynamique pour chaque split trouvé
            for i, key in enumerate(split_keys):
                # Extraction du Node ID pour la légende (split_node_7 -> Node 7)
                label_name = key.replace("split_node_", "Vanne N")
                color = col_split_base[i % len(col_split_base)]
                
                # Style de ligne pointillé pour différencier des variables physiques
                l_split, = ax3.plot(t_h, data[key]*100, color=color, linestyle="--", linewidth=1.5, alpha=0.8, label=label_name)
                lines.append(l_split)

            # S'assurer que l'axe n'est pas coupé à l'export
            # fig.subplots_adjust(right=0.85)
        
        # Légende unifiée
        labels = [l.get_label() for l in lines]
        # ax.legend(lines, labels, loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=len(lines), frameon=False)
        if split_keys:
            ax3.legend(lines, labels, ncol=1, frameon=True, loc="lower right", framealpha=0.95, fontsize=9)
        else:
            ax2.legend(lines, labels, ncol=2, frameon=True, loc="lower right", framealpha=0.95, fontsize=9)

        titre="1. Contrôle à la source"
        if split_keys :
            titre+= " & vannes"

        ax.set_title(titre, fontsize=12, fontweight='bold', loc="left", pad=10)
        ax.grid(True, alpha=0.3)

        # --- GRAPHIQUE 2 : Bilan énergétique ---
        ax = axes[1]
        dem = data["demand_total"] / 1e6 
        sup = data["supplied_total"] / 1e6
        wasted = data["wasted_total"] / 1e6
        
        ax.plot(t_h, dem, color=col_dem, linestyle=':', linewidth=1.5, label="Demande")
        ax.fill_between(t_h, sup, sup+wasted, color=col_wast, alpha=0.6, label="Perdu")
        ax.fill_between(t_h, sup, dem, where=(dem>sup), color='#E53935', alpha=0.3, hatch='///', label="Déficit")
        ax.fill_between(t_h, 0, sup, color=col_sup, alpha=0.5, label="Fourni")
        
        ax.set_ylabel("Puissance (MW)")
        ax.set_xlabel("Temps (heures)")
        ax.set_title("2. Bilan global réseau", fontsize=12, fontweight='bold', loc="left", pad=10)
        ax.legend(loc='lower right', frameon=True, fontsize=9, framealpha=0.95)
        ax.set_ylim(bottom=1.5)
        ax.grid(True, alpha=0.3)

        # Sauvegarde
        plt.tight_layout()
        # Petite correction car tight_layout gère mal les axes décalés (offset spines)
        if split_keys:
            plt.subplots_adjust(right=0.75)

        out_file = os.path.join(self.plots_dir, f"dashboard_GENERAL_{title_suffix}.svg")
        # out_file = os.path.join(self.plots_dir, f"dashboard_GENERAL_{title_suffix}.png")
        plt.savefig(out_file, bbox_inches='tight') # bbox_inches aide aussi pour l'axe décalé
        print(f" Graphique sauvegardé : {out_file}")
        plt.close(fig)
    
    def plot_dashboard_nodes_2cols(self, data, title_suffix=""):
        col_temp, col_flow = '#D32F2F', '#1976D2'
        col_dem, col_sup, col_wast = '#212121', '#388E3C', '#FBC02D'
        t_h = data["time"] / 3600.0
        all_nodes = data["node_ids"]
        main_group = [n for n in [2, 3, 4, 5, 6] if n in all_nodes]
        branch_3 = [n for n in [7, 8] if n in all_nodes]
        branch_5 = [n for n in [9, 10, 11] if n in all_nodes]
        col_left_data = [("Chaîne Principale", main_group)]
        col_right_data = []
        if branch_3: col_right_data.append(("Branche #3", branch_3))
        if branch_5: col_right_data.append(("Branche #5", branch_5))
        if not col_right_data and not col_left_data: return
        subplot_width = 6
        subplot_height = 3
        left_nodes = col_left_data[0][1] if col_left_data else []
        right_nodes = []
        for g in col_right_data:
            right_nodes.extend(g[1])
        n_left = len(left_nodes)
        n_right = len(right_nodes)
        n_rows = max(n_left, n_right, 1)
        fig_width = subplot_width * 2
        fig_height = subplot_height * n_rows
        fig, axs = plt.subplots(n_rows, 2, figsize=(fig_width, fig_height), sharex=True)
        if n_rows == 1:
            axs = np.array([axs])
        def plot_node(ax, nid):
            p_dem = data[f"node_{nid}_p_dem"] / 1000.0
            p_sup = data[f"node_{nid}_p_sup"] / 1000.0
            m_in = data[f"node_{nid}_m_in"]
            ax.plot(t_h, p_dem, color=col_dem, linestyle=':', linewidth=1.5, label="Demande")
            ax.plot(t_h, p_sup, color=col_sup, label="Fourni", linewidth=2, alpha=0.9)
            ax.fill_between(t_h, p_sup, p_dem, where=(p_dem > p_sup), color='red', alpha=0.15)
            ax.set_ylabel("Puissance (kW)", fontsize=9)
            ax2 = ax.twinx()
            ax2.plot(t_h, m_in, color=col_flow, linewidth=2, alpha=1)
            ax2.grid(False)
            ax2.set_ylabel("Débit (kg/s)", fontsize=9)
            ax2.tick_params(axis='y', labelcolor=col_flow, labelsize=9)
            ax2.spines['right'].set_visible(True)
            ax.grid(alpha=0.5)
            ax2.text(0.015, 0.88, f"Nœud {nid}", transform=ax.transAxes, fontsize=9, fontweight='bold', 
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2), zorder=10)
            l1, lab1 = ax.get_legend_handles_labels()
            l2, lab2 = ax2.get_legend_handles_labels()
            ax2.legend(l1+l2, lab1+lab2, loc='upper right', fontsize=9, ncol=3, frameon=True)
        for i in range(n_rows):
            if i < n_left:
                plot_node(axs[i, 0], left_nodes[i])
            else:
                axs[i, 0].axis('off')
            if i < n_right:
                plot_node(axs[i, 1], right_nodes[i])
            else:
                axs[i, 1].axis('off')
        axs[-1, 0].set_xlabel("Temps (heures)")
        axs[-1, 1].set_xlabel("Temps (heures)")
        out_file = os.path.join(self.plots_dir, f"dashboard_NODES_{title_suffix}.svg")
        fig.tight_layout()
        plt.savefig(out_file)
        print(f" Graphique Nœuds sauvegardé : {out_file}")
        plt.close(fig)
