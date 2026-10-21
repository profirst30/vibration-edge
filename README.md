# Vibrations : décider sur le capteur pour économiser la radio

**Un capteur sans pile, alimenté par récupération d'énergie, ne peut pas envoyer son signal vibratoire brut. Il doit calculer sur place et n'envoyer que l'essentiel. Ce dépôt mesure ce que coûte et ce que rapporte chaque choix, sur de vrais roulements usés jusqu'à la panne.**

Projet personnel. Page de démo : https://profirst30.github.io/vibration-edge/

## Le problème

En surveillance vibratoire de machines, les signes d'usure classiques sont un niveau global qui monte, de nouvelles fréquences et des à-coups de plus en plus fréquents. Les capteurs sans fil alimentés par récupération d'énergie (chaleur, vibration, lumière) émettent en radio LPWAN avec un budget d'énergie très serré.

Un enregistrement vibratoire d'une seconde pèse des dizaines de kilo-octets, alors qu'une trame LoRaWAN en porte au plus quelques centaines d'octets. La question est donc : **que calculer sur le nœud, et qu'envoyer ?**

## Ce que fait le dépôt

1. **Indicateurs sur le nœud** (`src/features.py`). Pour chaque enregistrement d'1 s à 20 kHz, 12 indicateurs, qui suivent les trois signes d'usure :
   - le niveau global : RMS et crête ;
   - les nouvelles fréquences : niveau dans 8 bandes de 1,25 kHz, par FFT ;
   - les à-coups : facteur de crête et kurtosis.
2. **Deux détecteurs** (`src/detect.py`). Chacun n'apprend que sur les premiers 10 % de la vie de son capteur, comme un nœud qui apprend son « normal » après l'installation.
   - **z-score** : écart de chaque indicateur à sa moyenne de départ.
   - **autoencodeur** : 271 paramètres, exporté en ONNX, puis quantifié en INT8 avec ONNX Runtime.

   Pour les deux, la règle d'alerte est la même. Le seuil vaut 1,5 fois le score le plus haut vu pendant l'apprentissage, et l'alerte se déclenche après 3 enregistrements de suite au-dessus du seuil (30 min). Ces réglages sont fixés à l'avance : ils n'ont pas été ajustés sur les pannes.
3. **Trames** (`src/payload.py`, `docs/decoder.js`).
   - Une trame d'indicateurs fait 26 octets : version, état, puis les 12 indicateurs sur 16 bits en log.
   - Une trame d'état fait 2 octets.
   - Le décodeur JavaScript suit le format `decodeUplink` de The Things Stack. Un test vérifie qu'il décode exactement comme Python.
4. **Budget radio** (`src/radio.py`). Le temps d'antenne LoRa est calculé avec la formule de la fiche technique Semtech SX1276, en EU868, à 125 kHz, pour SF7 et SF12. Trois stratégies sont comparées :
   - envoyer le signal brut ;
   - envoyer les indicateurs toutes les 10 minutes ;
   - n'envoyer qu'une trame par alerte, plus un battement de cœur par jour.

   Le nombre d'alertes utilisé vient des détections réelles.
5. **Coût du calcul** (`src/run.py`). Le temps mesuré sur ce PC, sur un thread, pour calculer les indicateurs et pour exécuter le modèle en FP32 et en INT8.

## Résultats

<!-- results -->
| essai | roulement | cassé à la fin | z-score | autoencodeur FP32 | autoencodeur INT8 |
|---|---:|---|---|---|---|
| test1 | 1 | non | 83,4 h avant la fin (2 déclenchements) | 11,9 h avant la fin (2 déclenchements) | 11,9 h avant la fin (2 déclenchements) |
| test1 | 2 | non | 91,4 h avant la fin (14 déclenchements) | 91,5 h avant la fin (13 déclenchements) | 91,5 h avant la fin (14 déclenchements) |
| test1 | 3 | **oui** | 230,9 h avant la fin (14 déclenchements) | 274,1 h avant la fin (23 déclenchements) | 274,1 h avant la fin (23 déclenchements) |
| test1 | 4 | **oui** | 150,4 h avant la fin (3 déclenchements) | 119,8 h avant la fin (1 déclenchement) | 119,9 h avant la fin (1 déclenchement) |
| test2 | 1 | **oui** | 71,8 h avant la fin (2 déclenchements) | 72,5 h avant la fin (3 déclenchements) | 72,5 h avant la fin (3 déclenchements) |
| test2 | 2 | non | 33,8 h avant la fin (7 déclenchements) | 36,0 h avant la fin (7 déclenchements) | 36,0 h avant la fin (7 déclenchements) |
| test2 | 3 | non | 18,8 h avant la fin (1 déclenchement) | 25,3 h avant la fin (2 déclenchements) | 25,2 h avant la fin (2 déclenchements) |
| test2 | 4 | non | 46,2 h avant la fin (1 déclenchement) | 55,2 h avant la fin (6 déclenchements) | 55,2 h avant la fin (6 déclenchements) |
| test3 | 1 | non | 58,8 h avant la fin (1 déclenchement) | 301,7 h avant la fin (5 déclenchements) | 301,7 h avant la fin (5 déclenchements) |
| test3 | 2 | non | 25,5 h avant la fin (1 déclenchement) | 26,8 h avant la fin (1 déclenchement) | 26,8 h avant la fin (1 déclenchement) |
| test3 | 3 | **oui** | 58,8 h avant la fin (1 déclenchement) | 827,3 h avant la fin (80 déclenchements) | 827,3 h avant la fin (74 déclenchements) |
| test3 | 4 | non | 729,1 h avant la fin (100 déclenchements) | 729,1 h avant la fin (244 déclenchements) | 729,1 h avant la fin (245 déclenchements) |

| stratégie | SF | octets par message | trames par jour | temps d'antenne par jour (s) | part du rapport cyclique de 1 % |
|---|---:|---:|---:|---:|---:|
| signal brut | 7 | 40 960 | 26 640 | 9 804,534 | 1 134,78 % |
| indicateurs | 7 | 26 | 144 | 11,833 | 1,37 % |
| alerte seule | 7 | 2 | 1,4 | 0,066 | 0,01 % |
| signal brut | 12 | 40 960 | 115 776 | 323 204,678 | 37 407,95 % |
| indicateurs | 12 | 26 | 144 | 284,295 | 32,9 % |
| alerte seule | 12 | 2 | 1,4 | 1,649 | 0,19 % |

Sur AMD Ryzen 7 5825U with Radeon Graphics, un thread : indicateurs d'un enregistrement 924,6 µs ; autoencodeur FP32 29,0 µs, INT8 38,2 µs (271 paramètres, 240 multiplications-additions ; fichier ONNX de 1 945 octets en FP32, 5 841 en INT8).
<!-- /results -->

Tous les chiffres ci-dessus sont écrits par `python -m src.run` à partir des données : aucun n'est tapé à la main.

## Ce que disent les chiffres

- **Les quatre roulements qui cassent déclenchent une alerte bien avant la fin de l'essai, avec les deux détecteurs.**
- **Des capteurs de roulements sains déclenchent aussi une alerte en fin d'essai.** Les quatre roulements sont sur le même arbre, donc la vibration de celui qui casse se propage aux autres. Un nœud signale « la machine change », pas forcément « mon roulement casse ».
- **Le roulement 4 de l'essai 3 alerte très tôt sans être noté comme cassé.** Le z-score montre que c'est la bande 7,5-8,75 kHz qui monte. Sans rapport d'inspection, impossible de dire si c'est une fausse alerte ou un vrai changement non documenté.
- **Le z-score fait aussi bien que l'autoencodeur, avec moins de déclenchements parasites sur l'essai 3.** Sur ces données, un réseau de neurones n'apporte rien de plus qu'une statistique simple. C'est un résultat, pas un échec : la méthode la plus sobre gagne.
- **L'INT8 donne presque les mêmes alertes que le FP32** : même première alerte à un enregistrement près, et un nombre de déclenchements très proche. Mais sur ce processeur, il est plus lent et son fichier est plus gros : pour 271 paramètres, les nœuds de quantification et leurs échelles pèsent plus que les poids. L'INT8 se justifie sur un microcontrôleur sans unité de calcul flottant ou avec un accélérateur, pas ici.
- **Le calcul coûteux, c'est la FFT, pas le modèle.** Sur un nœud alimenté par la récupération d'énergie, le vrai levier est la taille et la fréquence des enregistrements, pas le réseau.
- **La radio tranche.**
  - Le signal brut est impossible : il dépasse de loin le rapport cyclique autorisé.
  - Les indicateurs passent en SF7, mais prennent une grosse part du rapport cyclique en SF12, quand le capteur est loin de la passerelle.
  - L'alerte seule ne coûte presque rien.

  C'est l'argument pour décider sur le capteur.

## Limites

- **Aucun microcontrôleur testé** : le temps de calcul est mesuré sur un PC portable. La puissance récoltée n'est pas modélisée : elle dépend du récupérateur et de son installation. Les stratégies sont donc comparées en temps d'antenne, sans parler d'énergie.
- **Banc d'essai de laboratoire** : 2000 tr/min constants et une charge fixe. Une machine industrielle change de régime, et un z-score figé déclencherait à chaque changement.
- **Pannes datées seulement « à la fin de l'essai »** d'après la documentation IMS. Il n'y a pas de date de début de défaut, donc les « heures avant la fin » mesurent une avance sur la casse, pas une précision de détection.
- **Un seul accéléromètre par roulement** (l'axe x pour l'essai 1). Les essais 2 et 3 contiennent quelques enregistrements quasi nuls, machine arrêtée : ces instants sont exclus, et leur nombre est indiqué dans `results/results.json`.
- **L'autoencodeur est entraîné sur les périodes d'apprentissage des 12 capteurs réunies**, et chaque capteur garde sa propre normalisation.

## Reproduire

```sh
python -m venv .venv && .venv/Scripts/activate        # source .venv/bin/activate sous Linux
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
# Données : https://phm-datasets.s3.amazonaws.com/NASA/4.+Bearings.zip (1 Go), à extraire dans data/
#   (IMS.7z puis 1st_test.rar, 2nd_test.rar, 3rd_test.rar ; sous Windows, tar.exe sait les ouvrir)
python -m src.features       # indicateurs des 9 464 enregistrements -> data/features.npz
python -m src.run            # détection, modèle INT8, radio, coût -> results/, docs/, ce README
python -m pytest tests
```

La page de démo (`docs/index.html`) lit `docs/results.json` et `docs/series.json`. Pour la voir en local : `python -m http.server -d docs`.

Données : J. Lee, H. Qiu, G. Yu, J. Lin et Rexnord Technical Services (2007), IMS, University of Cincinnati, « Bearing Data Set », NASA Prognostics Data Repository, NASA Ames Research Center. Données publiques mises à disposition par la NASA, qui demande de citer la source ; aucune licence plus précise n'est indiquée.
