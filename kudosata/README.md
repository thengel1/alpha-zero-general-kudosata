# AlphaZero Kudosata — Cluster training

## Objectif

Cette version permet de lancer l’apprentissage AlphaZero de Kudosata sur le cluster de calcul.

Le cluster utilise uniquement :

- moteur Kudosata C++ exposé à Python
- pybind11
- PyTorch
- self-play
- MCTS
- Arena
- checkpoints

---

## Dépendances(strict minim) :

Créer un environnement :

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
````

Installer les dépendances :

```bash
pip install numpy==1.24.4
pip install torch==1.11.0
pip install tqdm==4.64.0
pip install coloredlogs==15.0.1
pip install pybind11==3.0.4
```

---

## Compilation du binding C++

Depuis la racine du projet :

```bash
cd kudosata
mkdir -p build
cd build
cmake ..
make
cd ../..
```

Tester le binding :

```bash
export PYTHONPATH=$(pwd)
python -c "from kudosata import openxum_kudosata as k; print('binding ok')"
```

---

## Fichiers cluster ajoutés

À la racine du projet :

```text
cluster_selfplay_worker.py
cluster_merge_examples.py
cluster_train_from_examples.py
```

Rôle :

```text
cluster_selfplay_worker.py
    génère des épisodes self-play indépendants

cluster_merge_examples.py
    fusionne les exemples générés par les workers

cluster_train_from_examples.py
    entraîne le réseau avec les exemples fusionnés
```

---

## Préparer le checkpoint de départ

Les workers utilisent un checkpoint figé :

```bash
cp temp/best.pth.tar temp/selfplay_current.pth.tar
```

---

## Générer du self-play

Exemple avec un worker :

```bash
export PYTHONPATH=$(pwd)

python cluster_selfplay_worker.py \
  --worker-id 0 \
  --num-eps 10 \
  --checkpoint ./temp \
  --checkpoint-file selfplay_current.pth.tar \
  --output-dir ./temp/cluster_selfplay
```

Chaque worker produit :

```text
temp/cluster_selfplay/examples_worker_<id>.pkl
```

---

## Fusionner les exemples

```bash
python cluster_merge_examples.py \
  --input-dir ./temp/cluster_selfplay \
  --output-file ./temp/cluster_train_examples.pkl \
  --maxlen 200000
```

---

## Entraîner avec les exemples fusionnés

```bash
python cluster_train_from_examples.py \
  --examples-file ./temp/cluster_train_examples.pkl \
  --checkpoint ./temp \
  --previous selfplay_current.pth.tar \
  --output best.pth.tar
```

---

## Pipeline cluster

```text
1. Installer les dépendances
2. Compiler le moteur C++
3. Copier le checkpoint de départ
4. Lancer les workers self-play
5. Fusionner les exemples
6. Entraîner le réseau
```

---

## Remarques importantes

* Chaque worker charge son propre jeu, son propre réseau et son propre MCTS.
* Les épisodes sont indépendants.
* Le self-play est la partie la plus coûteuse et la plus parallélisable.


```
