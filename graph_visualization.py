import os
import numpy as np
import matplotlib.pyplot as plt

class DistrictHeatingVisualizer:
    def __init__(self, plots_dir):
        self.plots_dir = plots_dir
        os.makedirs(self.plots_dir, exist_ok=True)

    def plot_dashboard_general(self, data, title_suffix="", plot_fraction_splits=False, fraction_splits_1=None, fraction_splits_2=None, fraction_labels=("Split 1", "Split 2")):
        t_h = data["time"] / 3600.0
        fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
        col_temp, col_flow = '#D32F2F', '#1976D2'
        col_dem, col_sup, col_wast = '#212121', '#388E3C', '#FBC02D'
        # 1. Source
        ax = axes[0]
        ax.plot(t_h, data["T_source"], color=col_temp, label="T Source (°C)", linewidth=2)
        ax.set_ylabel("Température (°C)")
        ax2 = ax.twinx()
        ax2.plot(t_h, data["m_source"], color=col_flow, label="Débit (kg/s)", linewidth=2)
        ax2.set_ylabel("Débit (kg/s)")
        ax2.spines['right'].set_visible(True)
        # Ajout d'un troisième axe y pour les fraction_splits si demandé
        if plot_fraction_splits and (fraction_splits_1 is not None or fraction_splits_2 is not None):
            ax3 = ax.twinx()
            # Décale le troisième axe pour éviter la superposition
            ax3.spines["right"].set_position(("axes", 1.12))
            colors = ["#8E24AA", "#43A047"]
            lines3 = []
            labs3 = []
            if fraction_splits_1 is not None:
                l1, = ax3.plot(t_h, fraction_splits_1, color=colors[0], linestyle="-", label=fraction_labels[0], linewidth=1.7)
                lines3.append(l1)
                labs3.append(fraction_labels[0])
            if fraction_splits_2 is not None:
                l2, = ax3.plot(t_h, fraction_splits_2, color=colors[1], linestyle="--", label=fraction_labels[1], linewidth=1.7)
                lines3.append(l2)
                labs3.append(fraction_labels[1])
            ax3.set_ylabel("Fraction split", color="#555")
            ax3.tick_params(axis='y', labelcolor="#555")
            ax3.spines['right'].set_visible(True)
        else:
            ax3 = None
            lines3 = []
            labs3 = []
        ax.set_title("1. Contrôle à la source", fontsize=12, fontweight='bold', pad=10, loc="left")
        lines1, lab1 = ax.get_legend_handles_labels()
        lines2, lab2 = ax2.get_legend_handles_labels()
        # Combine les légendes
        all_lines = lines1 + lines2 + lines3
        all_labs = lab1 + lab2 + labs3
        ax.legend(all_lines, all_labs, frameon=True, loc="lower center", ncol=2)
        ax.grid()
        # 2. Bilan
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
        ax.set_title("2. Bilan global", fontsize=12, fontweight='bold', pad=10, loc="left")
        ax.legend(loc='upper right', frameon=True)
        ax.grid()
        plt.tight_layout()
        out_file = os.path.join(self.plots_dir, f"dashboard_GENERAL_{title_suffix}.svg")
        plt.savefig(out_file)
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
            ax.legend(l1+l2, lab1+lab2, loc='upper right', fontsize=8, ncol=3, frameon=True)
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
