J'ai changé le nombre de steps à partir duquel on fait une itération : 1 journée (5h pour voi un changement)
Parallèlement, j'ai modifié le gamma (0.99 -> 0.9995) pour prendre en compte les anciens épisodes (voir calcul)

J'ai désactivé les rampes
tinit_70_sans_rampe

J'ai modifié la température initiale pour mettre tout à 70 au lieu 50, résultat : cela n'a rien changé, je remets donc tout à à 50
tinit_70_sans_rampe

J'ai modifié l'entropie : ??
& le reward (passé de 25 à 150 le confort) (pompe : 3.5, chaudière : 14)
Résultat : au début ça oscillait bêtement, puis plus tard dans l'entrainement il s'est mis à maxer la pompe et diminuer le chauffage pour respecter le confort (il a donc fallu rééquilibrer la pompe)

Par rapport à l'étape précédente : confort à 10, reste à 0.5 et 5
-> obtient test5

Pourquoi a arrêté d'apprendre et désapprend ? Approx kl démarre vers 0.007 et vient s'écraser à 0.005 très vite

à 656k, ep_rew_mean=-5.39e4
à 950400, -5.23

Poids de test 5 pas assez élevés ? rapports de 10 entre chacuns
commence à -3.25e4

mai2014
180806



In order to benchmark the performance of the RL agent a rule-based
approach to operation is used which will be referred to as hysteresis in the
following. Hysteresis strategies are commonly applied in district heating
systems and consist mainly of two rules or thresholds: A lower threshold of
the SOC value of the heat storage where the HP starts to operate in order
to increase the SOC as well as an upper threshold of the SOC value where
the HP stops operating. In this work, the lower and upper thresholds are
20% and 100% respectively. When active for hysteresis operation