# HeatPilot-RL : Contrôle de réseau de chaleur par apprentissage par renforcement

Ce projet est un environnement de recherche simulant la physique d'un réseau de chaleur urbain (District Heating Network) couplé à un agent d'intelligence artificielle (Reinforcement Learning) dont le but est d'optimiser la production de chaleur.

Ce document a pour but d'expliquer :
- comment fonctionne le simulateur,
- quelles sont les équations et hypothèses physiques,
- quels ordres de grandeur issus de réseaux réels ont guidé le dimensionnement,
- comment l'IA apprend à piloter le réseau,
- et [à remplir] comment analyser les performances de l’agent entraîné.

---

## 1. Vue d'ensemble du problème

Un réseau de chaleur transporte de l'eau chaude depuis une **centrale de production** (source) vers plusieurs **consommateurs** (bâtiments) via des canalisations isolées.

**Le défi :**  
Les consommateurs ont une demande de puissance variable (douche le matin, chauffage le soir). L'eau met du temps à voyager dans les tuyaux (retard thermique).
- Si on chauffe trop : on gaspille de l'énergie (pertes thermiques, coût de pompage).
- Si on ne chauffe pas assez : les consommateurs ont froid (inconfort).

**L'objectif de l'IA :**  
Anticiper les demandes et piloter la température et le débit à la source pour satisfaire les besoins tout en minimisant la consommation énergétique (chaudière + pompage).

---

## 2. Contexte de dimensionnement et ordres de grandeur

Cette section explique **les choix numériques du modèle** et montre qu’ils sont cohérents avec les données disponibles sur de vrais réseaux de chaleur.

### 2.1. Puissance du réseau “type”

Le réseau simulé représente un **petit réseau de chaleur** d’ordre de grandeur :

- **Puissance installée** : ~2 MW thermiques.
- **Nombre de sous-stations** : typiquement 3 à 8 nœuds consommateurs. Dans `config.py`, le graphe `EDGES` décrit 1 source et plusieurs nœuds consommateurs raccordés.

Ce choix est cohérent avec les données suivantes (voir le document d’annexe sur les réseaux de chaleur) :
- En France, ~1 000 réseaux de chaleur totalisent ~23,8 GW, soit ~24 MW de puissance moyenne par réseau. Un réseau “type” de 2 MW est donc représentatif d’un **petit réseau municipal ou de quartier**.
- Les chaufferies biomasse ou gaz de cette taille (0,5–3 MW) sont fréquentes dans les petites communes et quartiers résidentiels.

### 2.2. Températures aller / retour

Dans `config.py`, on utilise typiquement :
- Température de départ cible : **60–110 °C** (`CONTROL_LIMITS["temp_min"/"temp_max"]`),
- Température minimale de retour : **MIN_RETURN_TEMP = 40 °C**.

Ces valeurs sont basées sur :
- Les régimes classiques des réseaux eau chaude : **70–110 °C** en départ, **40–70 °C** en retour, avec ΔT = 20–40 K.
- Les exigences sanitaires (ECS à ≥ 60 °C côté secondaire) et les recommandations de réseaux basse température (70/40 °C, 80/50 °C, etc.).

**Justification :**
- 40 °C en retour est un compromis :
  - suffisamment chaud pour rester réaliste par rapport aux retours de sous-stations (radiateurs ~40 °C),
  - suffisamment froid pour laisser un ΔT significatif (20–40 K) et donc un débit raisonnable.
- Le contrôle permet de monter la température jusqu’à 110 °C pour représenter les situations de pointe (hiver rigoureux ou sur-dimensionnement volontaire de la consigne).

### 2.3. Débits et puissance de pompage

Dans `config.py`, le débit massique contrôlable est limité à :

- `flow_min = 1 kg/s`, `flow_max = 30 kg/s`.

Pour une puissance de 2 MW et un ΔT de 30 K (par exemple 80/50 °C), le débit massique requis est :

$$
\dot m \approx \frac{P}{c_p \Delta T}
       \approx \frac{2{\,}000}{4{,}18 \times 30}
       \approx 16\ \text{kg/s}
$$

Ce débit (16 kg/s) est **à l’intérieur de la plage 1–30 kg/s**, ce qui laisse de la marge pour :

- réduire le débit (optimisation de pompage),
- augmenter le débit en cas de ΔT plus faible (exploitation dégradée).

Dans le modèle, la puissance de pompage est donnée par :

$$
P_{\text{pompe}}(t) = 1000 \cdot \dot m(t)
$$

**Hypothèse implicite :**
- On suppose une **hauteur manométrique et un rendement global constants**, tels que :

$$
P = \dot m g H / \eta \approx 1000 \cdot \dot m
$$

Par exemple, pour $\dot{m} = 16\ \text{kg/s}$, on obtient $P_{\text{pompe}} \approx 16\ \text{kW}$, ce qui correspond à l’ordre de grandeur attendu pour un réseau de quelques MW avec ~100–150 kPa de pertes de charge et un rendement pompe + moteur ~60 %.

**On ne cherche pas ici à modéliser précisément l’hydraulique**, mais à obtenir :
- une **pénalisation énergétique croissante avec le débit**,
- des ordres de grandeur plausibles pour comparer coût thermique et coût de pompage.

### 2.4. Puissance demandée par bâtiment / nœud

`POWER_PROFILE_CONFIG` définit :

- `p_min = 100 kW`, `p_max = 300 kW` par nœud consommateur,
- pas de temps de changement de demande : `step_time = 7200 s` (2 h).

Ordres de grandeur :
- Un immeuble résidentiel de 2 000–5 000 m², avec 70–100 W/m² de puissance de chauffage de pointe, représente **140–500 kW**.
- Les valeurs 100–300 kW par nœud sont donc typiques d’un immeuble ou petit groupe d’immeubles.

Avec 4–6 nœuds de 100–300 kW :
- puissance appelée cumulée maximale ≈ 0,4–1,8 MW,  
  compatible avec la puissance de production cible (~2 MW).

### 2.5. Volumes d’eau et inertie thermique (choix de la discrétisation)

Les chaudières et réseaux réels présentent des volumes d’eau importants (plusieurs milliers de litres), voir le document sur les volumes internes :

- Chaudières biomasse 250–2500 kW : ratio typique 2–5 L/kW → une chaudière de 2 MW contient 4 000–10 000 L d’eau.
- La boucle réseau (canalisations) ajoute des centaines de m³.

Dans le modèle :

- Chaque tuyau est discrétisé en **20 à 60 segments** (`PIPE_GENERATION`), avec un pas spatial `dx = 0.01 m`.
- On ne cherche pas à reproduire le *volume absolu* du réseau réel, mais à obtenir :
  - un **temps de transport** non négligeable,
  - une **inertie thermique distribuée** suffisante pour que les actions de l’agent aient un effet différé et non instantané.

Ce choix permet de tester les capacités d’anticipation de l’agent RL sans alourdir le modèle au point de rendre l’intégration ODE trop coûteuse.

---

## 3. Modélisation Physique (Le Simulateur)

Le cœur du code se trouve dans `district_heating_model.py`. Nous utilisons une approche 1D (unidimensionnelle) basée sur la méthode des volumes finis.

### 3.1. Hypothèses Simplificatrices

1. **Incompressibilité :**  
   L'eau est considérée incompressible. Le débit massique se propage instantanément dans tout le réseau (pas d’ondes de pression, pas de dynamique transitoire hydrauliques).
   - Justification :  
     Pour de l’eau liquide à faible compressibilité, les vitesses typiques (0,5–2 m/s) et les variations de pression restent modérées. À l’échelle de temps thermique (secondes à minutes) la dynamique de pression est très rapide → on peut la considérer quasi-statique.

2. **Mélange parfait aux nœuds :**  
   Aux jonctions, les flux se mélangent instantanément.
   - Justification :  
     Les volumes de mélange en chambre de sous-station ou collecteur sont faibles devant la longueur des canalisations, et les temps de mélange volumique sont négligeables à l’échelle de quelques secondes.

3. **Isolation uniforme :**  
   Les pertes thermiques linéiques sont modélisées par un coefficient `heat_loss_coeff` constant le long de chaque tuyau, tiré aléatoirement dans un intervalle raisonnable.
   - Justification :  
     Dans les réseaux réels, les pertes sont de l’ordre de 5–15 % de l’énergie transportée sur une année. Un coefficient constant par conduit permet de représenter ce phénomène sans introduire une dépendance trop fine au voisinage (sol, nappe, etc.).

4. **Conduction longitudinale négligée (facultative) :**  
   Le terme diffusif peut être activé, mais est généralement nul ou très faible. Le transport par **advection** domine dans des conduites avec Re élevés.

5. **Pas de dynamique chaudière détaillée :**  
   La chaudière est modélisée comme un **producteur imposant une température de départ** et recevant un débit. Pas de bilans détaillés de combustion ou d’inertie du corps de chauffe.
   - Justification :  
     Pour le RL, l’essentiel est la relation entrée-sortie (T_in, ṁ → P_boiler) et les contraintes sur les rampes, plus que la transitoire interne de la chaudière.

### 3.2. L'Équation de la Conduite (Pipe)

Chaque tuyau est divisé en petits segments (cellules) de longueur $dx$. L'évolution de la température $T$ suit :

$$
\frac{\partial T}{\partial t}
= -v \frac{\partial T}{\partial x}
- \frac{4 h}{\rho c_p D}(T - T_{ext})
+ \alpha \frac{\partial^2 T}{\partial x^2}
$$

Où :
- $v$ : vitesse du fluide (m/s),
- $h$ : coefficient de perte thermique (W/m²/K),
- $D$ : diamètre du tuyau,
- $T_{ext}$ : température du sol,
- $\alpha = \lambda / (\rho c_p)$ : diffusivité thermique effective (souvent négligée).

**Discrétisation numérique (`Pipe.compute_derivatives`) :**

Pour une cellule i, on utilise un schéma **upwind** 1er ordre :

$$
\frac{dT_i}{dt}
= - \frac{\dot{m}}{\rho A dx}(T_i - T_{i-1})
  - \lambda (T_i - T_{ext})
$$

avec :

- $\dot{m}$ : débit massique (kg/s),
- $A$ : aire de section intérieure du tube,
- $\lambda = \frac{4 h}{\rho c_p D}$.

**Pourquoi un schéma upwind ?**
- Il est **numériquement stable** pour l’advection pure,
- Il respecte la direction physique du flux,
- Il évite les oscillations non physiques en présence de fronts de température marqués.

Le terme de diffusion peut être ajouté via un schéma à 3 points (Laplacien 1D).

### 3.3. Représentation du réseau et des nœuds

Le graphe du réseau est défini par `EDGES` dans `config.py` et représenté via `Graph` (`graph_utils.py`).

Les nœuds assurent :

1. **Conservation de la masse :**

$$
\sum \dot{m}_{in} = \sum \dot{m}_{out}
$$

Cette conservation est imposée par le routage des débits via les **fractions de split** aux nœuds de branchement.

2. **Mélange thermique :**

$$
T_{node} = \frac{\sum (\dot{m}_{in} T_{in})}{\sum \dot{m}_{in}}
$$

Calculé dans `_solve_nodes_temperature`.

3. **Soutirage de puissance aux nœuds consommateurs :**  
Pour chaque nœud consommateur, une puissance demandée $P_{demand}(t)$ est générée (profil en escalier).  
Le réseau fournit une puissance :

$$
P_{supplied} = \dot{m}_{in} c_p (T_{inlet} - T_{return})
$$

Dans `_apply_node_power_consumption`, on impose une chute de température locale :

$$
T_{\text{out}} = \max\left(T_{\text{in}} - \frac{P_{\text{supplied}}}{\dot{m}_{\text{in}} c_p},\ T_{\text{min return}}\right)
$$

avec $T_{\text{min return}} =$ `MIN_RETURN_TEMP` (40 °C), bornant la température de retour.

**Justification de $T_{\text{min return}}$ :**
- Physiquement, on ne peut pas extraire une puissance infinie : la température de retour ne peut pas descendre en dessous de la température ambiante du bâtiment / des retours des émetteurs.
- Cela évite des débits nuls accompagnés de chutes de température irréalistes, et donne une puissance maximale soutirable $P_{\max} = \dot{m}_{in} c_p (T_{in} - T_{\text{min return}})$.

---

## 4. L'Environnement RL (Gymnasium)

Le fichier `district_heating_gym_env.py` fait le lien entre la physique et l'IA.

### 4.1. Agent et horizon temporel

- Un épisode représente **une journée** : `t_max_day = 24h`.
- Pas de contrôle : `dt = 0.1 s` dans la config actuelle (à adapter selon la granularité souhaitée).  
  Cela permet d’approcher des contraintes thermiques “en 10 secondes” discutées dans le document sur les chaudières (gradients max par 10s de l’ordre de 0,5–2,5 °C pour l’eau chaude).

L’agent contrôle à chaque pas :
- la température de départ,
- le débit massique,
- la répartition des débits aux nœuds de branchement.

### 4.2. Espace d’Observation (ce que l’agent voit)

L’observation est un vecteur contenant :

1. Température actuelle aux nœuds consommateurs (feedback retardé par le transport).
2. Température actuelle de départ à la source.
3. Débit massique actuel.
4. Demande de puissance courante pour chaque consommateur.

Ces informations suffisent pour :
- Estimer l’état thermique actuel du réseau (via les températures de retour),
- Connaître la demande instantanée,
- Adapter T_in et ṁ pour anticiper les variations de charge.

### 4.3. Espace d’Action (ce que l’agent contrôle)

L’action est un vecteur continu :

1. **Target Temperature** : consigne de température à la source, dans `[temp_min, temp_max] = [60, 110] °C`.
2. **Target Flow** : débit massique cible, dans `[flow_min, flow_max] = [1, 30] kg/s`.
3. **Splits** : pour chaque nœud de branchement, un scalaire `0–1` représentant la fraction de débit allouée au premier enfant (le reste allant au second).

**Ramping / contraintes dynamiques :**

Pour coller aux limites de gradients de température et de débit réelles :

- **Température** :
  - Montée maximale par pas : `max_temp_rise_per_dt = 0.5 °C`  
    → sur 10 s, cela donne un gradient de l’ordre de 0,5–2,5 °C selon `dt`, compatible avec les gradients permis pour des chaudières eau chaude (5–15 K/min).
  - Descente plus rapide autorisée : `max_temp_drop_per_dt = 5 °C` (simule l’ouverture de mélange / baisse de consigne).

- **Débit** :
  - Variation max : `max_flow_delta_per_dt = 1 kg/s par pas`.

Ces bornes sont choisies pour :
- Empêcher l’agent de générer des variations irréalistes (chocs thermiques ou hydrauliques),
- Rester dans les ordres de grandeur observés (voir la section sur “Variation maximale de température en 10 s”).

### 4.4. Fonction de Coût / Récompense

La récompense est construite comme :

$$
Reward = - \left( w_1 \cdot \text{ÉcartConfort}
                + w_2 \cdot P_{\text{chaudière}}
                + w_3 \cdot P_{\text{pompe}} \right)
$$

Dans le code :

$$
\text{ÉcartConfort} = \sum_i \left|P_{\text{supplied},i} - P_{\text{demand},i}\right|
$$

$$
P_{\text{chaudière}} = \dot{m} \, c_p \, \bigl(T_{\text{inlet}} - T_{\min,\text{return}}\bigr)
$$

$$
P_{\text{pompe}} = 1000 \cdot \dot{m}
$$

Les poids sont actuellement :

- `1.0e-4` pour l’écart de puissance,
- `1.0e-5` pour la puissance chaudière,
- `1.0e-2` pour la puissance de pompage.

**Justification :**
- L’objectif principal est de **respecter la demande** (écart-confort minimal),
- La pénalisation de l’énergie chaudière est plus faible mais incite à **baisser la consigne** si possible,
- La pénalisation du pompage est relativement forte par rapport à la chaudière pour encourager :
  - l’optimisation de ΔT (retours plus froids),
  - des débits réduits quand c’est possible.

**Remarque :**
Ces poids peuvent être ajustés pour refléter des coûts économiques ou environnementaux plus précis (prix gaz vs électricité, contenu CO₂, etc.).

---

## 5. Architecture du Code

Voici comment les fichiers interagissent :

```text
.
├── config.py                       # Paramètres globaux : topologie, physique, RL.
├── district_heating_model.py       # Moteur physique : classes Pipe et DistrictHeatingNetwork.
├── graph_utils.py                  # Structure de graphe et topologie (tri topologique, rôles des noeuds).
├── district_heating_gym_env.py     # Environnement Gym : interface physique <-> RL.
├── train_agent.py                  # Entraînement PPO (Stable-Baselines3) avec callbacks de sauvegarde.
├── evaluate_agent.py               # Évaluation d'un agent entraîné + génération de graphiques.
├── run_district_heating_simulation.py # Simulation déterministe "sans IA" (température/débit imposés).
└── utils.py                        # Fonctions utilitaires (profils temporels, etc.).
```

---

## 6. Comment lancer le projet ?

### 6.1. Pré-requis

Assurez-vous d'avoir installé les dépendances :

```bash
pip install numpy scipy matplotlib gymnasium stable-baselines3 s3fs
```

### 6.2. Étape 1 : Vérifier la physique

Lancez une simulation simple sans IA pour voir comment le réseau réagit thermiquement :

```bash
python run_district_heating_simulation.py
```

Cela :
- crée un réseau avec des conduites et pertes générées aléatoirement mais reproductibles,
- applique des profils de puissance aux nœuds consommateurs,
- calcule et trace :

  - la puissance totale demandée,
  - la puissance totale effectivement soutirée,
  - la puissance fournie par la chaudière,
  - la puissance de pompage (si vous ajoutez son tracé).

**[TODO FIGURE 1]**  
À cet endroit, vous pouvez insérer un graphique `power_balance.png` illustrant :
- $P_{demand\_tot}$,
- $P_{supplied\_tot}$,
- $P_{boiler}$,
en fonction du temps. Il est déjà en partie généré dans `run_district_heating_simulation.py`.

### 6.3. Étape 2 : Entraîner l'IA

Lancez l'apprentissage. L'agent va faire des essais/erreurs pendant plusieurs épisodes :

```bash
python train_agent.py
```

- Le modèle final sera sauvegardé dans `ppo_heat_network_final.zip`.
- Des checkpoints intermédiaires seront stockés dans `./checkpoints/`.
- Optionnellement, les checkpoints peuvent être envoyés vers S3 si `use_s3_checkpoints=True` dans `config.py`.

### 6.4. Étape 3 : Évaluer et Visualiser

Une fois l'entraînement fini, regardez comment l'agent se comporte sur un scénario de test :

```bash
python evaluate_agent.py
```

- Cela génère `evaluation_results.png` avec :
  - température de départ,
  - températures aux nœuds consommateurs,
  - débit,
  - récompense instantanée.

**[TODO FIGURE 2]**  
Insérer ici un exemple d’`evaluation_results.png` montrant :
- le suivi des températures noeuds vs consigne,
- le débit piloté,
- l’évolution de la récompense.

---

## 7. Topologie du Réseau (Exemple)

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

Le fichier `graph_utils.py` analyse automatiquement ces connexions pour :
- identifier le noeud d’entrée (source),
- repérer les nœuds de branchement,
- déterminer un **ordre topologique** de calcul des débits.

---

## 8. Analyse prévue des performances de l’agent RL (à compléter)

Cette section décrit la structure d’analyse à mettre en place une fois que vous aurez suffisamment de modèles entraînés.

### 8.1. Métriques globales

À calculer sur un épisode de test :

- **Énergie totale demandée** vs fournie (intégrale de $P_{demand\_tot}$ et $P_{supplied\_tot}$).
- **Énergie chaudière** (intégrale de $P_{boiler}(t)$).
- **Énergie de pompage** (intégrale de $P_{pump}(t)$).
- **Écart confort moyen et max** :

$$
\frac{1}{T}\int \left|P_{\text{supplied}}(t) - P_{\text{demand}}(t)\right| \,\mathrm{d}t
$$

- **Récompense cumulée** sur l’épisode.

**[TODO FIGURE 3]**  
Graphique : `energy_breakdown.png` montrant, pour un épisode :
- énergie demandée (kWh),
- énergie fournie,
- énergie chaudière,
- énergie pompage,
pour une politique RL vs une politique de référence (par ex. température et débit constants).

### 8.2. Profils temporels

Analyser :

- la trajectoire de la température de départ,
- le débit massique,
- les températures de retour aux nœuds,
- la réponse du réseau à des marches de demande.

**[TODO FIGURE 4]**  
Graphique multi-panneaux :
- (a) T_in(t) et T_consommateurs(t),
- (b) ṁ(t),
- (c) P_demand_tot(t) et P_supplied_tot(t),
- (d) Reward(t).

Ces courbes sont en grande partie déjà disponibles via `evaluate_agent.py`. Il suffira éventuellement de compléter le script pour exporter des données au format `.npz` ou `.csv` et tracer avec un notebook.

### 8.3. Comparaison avec des stratégies baselines

Comparer la politique RL avec :

1. **Stratégie naïve** :  
   - T_in fixe (par ex. 90 °C),
   - ṁ fixe (débit max ou débit dimensionné pour ΔT=20K),
   - splits uniformes.
2. **Stratégie “loi d’eau” simplifiée** :  
   - T_in modulée en fonction d’une température extérieure synthétique,
   - ṁ constant.

Pour chaque stratégie, calculer :
- les mêmes métriques globales,
- l’écart de performance relatif (gain en % sur la consommation d’énergie pour un même confort).

**[TODO TABLEAU 1]**  
Insérer un tableau récapitulatif :

| Stratégie        | Énergie chaudière (kWh) | Énergie pompe (kWh) | Écart confort moyen (kW) | Récompense cumulée |
|------------------|-------------------------|----------------------|--------------------------|--------------------|
| Naïve            | …                       | …                    | …                        | …                  |
| Loi d’eau        | …                       | …                    | …                        | …                  |
| RL (HeatPilot)   | …                       | …                    | …                        | …                  |

### 8.4. Robustesse et généralisations

Pistes d’analyse supplémentaires (à implémenter plus tard) :

- **Robustesse à des profils de demande non