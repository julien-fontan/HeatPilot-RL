# HeatPilot-RL : Contrôle de Réseau de Chaleur par Apprentissage par Renforcement

Bienvenue dans **HeatPilot-RL**. Ce projet est un environnement de recherche simulant la physique d'un réseau de chaleur urbain (District Heating Network) couplé à un agent d'intelligence artificielle (Reinforcement Learning) dont le but est d'optimiser la production de chaleur.

Ce document a pour but d'expliquer **pédagogiquement** comment fonctionne le simulateur, quelles sont les équations physiques utilisées, et comment l'IA apprend à piloter le réseau.

---

## 1. Vue d'ensemble du problème

Un réseau de chaleur transporte de l'eau chaude depuis une **centrale de production** (source) vers plusieurs **consommateurs** (bâtiments) via des canalisations isolées.

**Le défi :**
Les consommateurs ont une demande de puissance variable (douche le matin, chauffage le soir). L'eau met du temps à voyager dans les tuyaux (retard thermique).
- Si on chauffe trop : on gaspille de l'énergie (pertes thermiques, coût de pompage).
- Si on ne chauffe pas assez : les consommateurs ont froid (inconfort).

**L'objectif de l'IA :**
Anticiper les demandes et piloter la température et le débit à la source pour satisfaire les besoins tout en minimisant la consommation énergétique.

---

## 2. Modélisation Physique (Le Simulateur)

Le cœur du code se trouve dans `district_heating_model.py`. Nous utilisons une approche 1D (unidimensionnelle) basée sur la méthode des volumes finis.

### A. Hypothèses Simplificatrices
1.  **Incompressibilité :** L'eau est considérée incompressible. Le débit massique se propage instantanément dans tout le réseau (pas d'ondes de pression).
2.  **Mélange Parfait :** Aux jonctions, les flux se mélangent instantanément.
3.  **Isolation constante :** Les pertes thermiques dépendent linéairement de la différence de température avec l'extérieur.

### B. L'Équation de la Conduite (Pipe)
Chaque tuyau est divisé en petits segments (cellules) de longueur $dx$. L'évolution de la température $T$ dans une cellule suit l'équation d'advection-diffusion avec pertes :

$$
\frac{\partial T}{\partial t} = \underbrace{-v \frac{\partial T}{\partial x}}_{\text{Transport}} - \underbrace{\frac{4 h}{\rho c_p D} (T - T_{ext})}_{\text{Pertes}} + \underbrace{\alpha \frac{\partial^2 T}{\partial x^2}}_{\text{Diffusion (négligée)}}
$$

Où :
*   $v$ : Vitesse du fluide ($m/s$).
*   $h$ : Coefficient de perte thermique ($W/m^2/K$).
*   $D$ : Diamètre du tuyau.
*   $T_{ext}$ : Température du sol.

**Discrétisation numérique (Code : `Pipe.compute_derivatives`)**
Nous utilisons un schéma "Upwind" (décentré amont) pour la stabilité numérique. Pour une cellule $i$, la variation de température est :

$$
\frac{dT_i}{dt} = - \frac{\dot{m}}{\rho A dx} (T_i - T_{i-1}) - \lambda (T_i - T_{ext})
$$

### C. Les Nœuds (Nodes)
Les nœuds connectent les tuyaux. Ils assurent :
1.  **La conservation de la masse (Loi de Kirchhoff) :** $\sum \dot{m}_{in} = \sum \dot{m}_{out}$.
2.  **Le bilan thermique (Mélange) :** La température résultante à un nœud est la moyenne pondérée par les débits des températures arrivantes.

$$
T_{node} = \frac{\sum (\dot{m}_{in} \cdot T_{in})}{\sum \dot{m}_{in}}
$$

### D. Les Consommateurs
Chaque consommateur demande une puissance $P_{demand}(t)$ (en Watts).
Le réseau fournit une puissance $P_{supplied}$ calculée par :

$$
P_{supplied} = \dot{m} \cdot c_p \cdot (T_{inlet} - T_{return})
$$

Le modèle calcule la chute de température nécessaire au nœud pour extraire cette puissance, sans descendre sous une température de retour minimale (`MIN_RETURN_TEMP`).

---

## 3. L'Environnement RL (Gymnasium)

Le fichier `district_heating_gym_env.py` fait le lien entre la physique et l'IA.

### L'Agent (Le Cerveau)
L'agent contrôle la centrale de production. À chaque pas de temps $\Delta t$ (ex: 10 minutes), il prend une décision.

### Espace d'Observation (Ce que l'agent voit)
L'agent reçoit un vecteur contenant :
1.  La température actuelle aux nœuds consommateurs (feedback retardé).
2.  La température de départ actuelle à la source.
3.  Le débit massique actuel.
4.  La demande de puissance actuelle des consommateurs (prévision parfaite à t).

### Espace d'Action (Ce que l'agent fait)
L'agent modifie :
1.  **Target Temperature :** La température de consigne à la source (ex: 70°C à 110°C).
2.  **Target Flow :** Le débit massique (ex: 1 kg/s à 30 kg/s).
3.  **Splits (Vannes) :** La répartition du débit aux intersections (quel % d'eau va à gauche ou à droite).

### Fonction de Récompense (Reward)
C'est la "note" donnée à l'agent. On veut maximiser le score, donc minimiser les pénalités (coûts).

$$
Reward = - ( w_1 \cdot \text{EcartConfort} + w_2 \cdot \text{EnergieThermique} + w_3 \cdot \text{EnergiePompage} )
$$

*   **EcartConfort :** Différence absolue entre la puissance demandée et fournie ($|P_{dem} - P_{sup}|$).
*   **EnergieThermique :** Coût du gaz/électricité pour chauffer l'eau ($\dot{m} c_p \Delta T$).
*   **EnergiePompage :** Coût électrique pour pousser l'eau (proportionnel au débit $\dot{m}$).

---

## 4. Architecture du Code

Voici comment les fichiers interagissent :

```text
.
├── config.py                       # Le "Cerveau des paramètres". Tout est ici (Topologie, Physique, RL).
├── district_heating_model.py       # Le "Moteur Physique". Contient les classes Pipe et Network.
├── district_heating_gym_env.py     # L'interface "Gym". Traduit la physique en états/récompenses pour l'IA.
├── train_agent.py                  # Le "Professeur". Lance l'entraînement PPO (Stable-Baselines3).
├── evaluate_agent.py               # L' "Examinateur". Charge un modèle entraîné et trace les courbes.
├── run_district_heating_simulation.py # Simulation "Sans IA". Pour tester la physique seule.
├── graph_utils.py                  # Outils pour gérer la topologie du graphe (tri topologique).
└── utils.py                        # Fonctions utilitaires (génération de profils aléatoires).
```

---

## 5. Comment lancer le projet ?

### Pré-requis
Assurez-vous d'avoir installé les dépendances :
```bash
pip install numpy scipy matplotlib gymnasium stable-baselines3
```

### Étape 1 : Vérifier la physique
Lancez une simulation simple sans IA pour voir comment le réseau réagit thermiquement.
```bash
python run_district_heating_simulation.py
```
*Cela générera des graphiques montrant l'équilibre puissance fournie vs demandée.*

### Étape 2 : Entraîner l'IA
Lancez l'apprentissage. L'agent va faire des essais/erreurs pendant des milliers d'épisodes.
```bash
python train_agent.py
```
*Le modèle sera sauvegardé dans `ppo_heat_network_final.zip` et des checkpoints dans `./checkpoints/`.*

### Étape 3 : Évaluer et Visualiser
Une fois l'entraînement fini, regardez comment l'agent se comporte sur un scénario de test.
```bash
python evaluate_agent.py
```
*Cela générera `evaluation_results.png` avec les courbes de température, débit et récompense.*

---

## 6. Topologie du Réseau (Exemple)

La topologie est définie dans `config.py` via la liste `EDGES`.
Exemple simple linéaire :
```text
[Source 1] ====> [Noeud 2] ====> [Consommateur 3]
```

Exemple avec branchement :
```text
                 /===> [Consommateur 4]
[Source 1] ===> [Noeud 2]
                 \===> [Consommateur 3]
```
Le fichier `graph_utils.py` analyse automatiquement ces connexions pour savoir dans quel ordre calculer les débits.

---

## 7. Pistes d'amélioration (Pour aller plus loin)
*   **Délais de transport variables :** Actuellement, le débit change instantanément partout. Ajouter l'inertie hydraulique.
*   **Prévision météo :** Ajouter la température extérieure future dans les observations de l'agent.
*   **Topologie complexe :** Tester sur des réseaux maillés (boucles), ce qui nécessiterait un solveur hydraulique plus complexe (ex: Hardy Cross).
