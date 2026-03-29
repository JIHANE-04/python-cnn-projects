# 🧬 CNN Détection de Tumeurs — MobileNetV2 Transfer Learning

> 📁 Dossier : `CNN_MobileNet_Tumeurs/` — Fichier : `CNN_MobileNet_Tumeurs.ipynb`

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![MobileNetV2](https://img.shields.io/badge/Transfer_Learning-MobileNetV2-red)
![Google Colab](https://img.shields.io/badge/Google-Colab-yellow)

Version améliorée de la détection de tumeurs cérébrales, exploitant **MobileNetV2** (Transfer Learning). Atteint une précision supérieure en seulement **10 époques**, contre 15 pour le CNN from scratch.

---

## 🎯 Objectif

Démontrer la **supériorité du Transfer Learning** sur un CNN from scratch pour la détection de tumeurs cérébrales, en utilisant MobileNetV2 pré-entraîné sur ImageNet comme extracteur de features.

---

## 🗂️ Dataset — `Dataset.zip`

| Élément | Détail |
|---------|--------|
| Fichier | `Dataset.zip` → `MyDrive/Projet_CNN/Dataset.zip` |
| Extraction | `/content/dataset` |
| Nombre de classes | **2 classes** |
| Classes | `Tumeur` (1) · `Sain` (0) |
| Images d'entraînement (80%) | **6 622 images** |
| Images de validation (20%) | **1 655 images** |
| Images de test | **1 816 images** |
| **Total** | **~10 093 images** |

> Même dataset que le CNN from scratch, permettant une comparaison directe des performances.

---

## ⚙️ Paramètres du modèle

| Paramètre | Valeur |
|-----------|--------|
| Taille des images | **224 × 224 px** |
| Batch size | 32 |
| Époques | **10** |
| Optimiseur | Adam |
| Split validation | 80% train / 20% validation |
| Loss | Binary Crossentropy |
| Activation sortie | Sigmoid (1 neurone) |
| Type de classification | Binaire (0 = Sain / 1 = Tumeur) |

> ✅ MobileNetV2 fonctionne de manière optimale avec des images **224×224** (taille native d'ImageNet).

---

## 🏗️ Architecture — MobileNetV2 (totalement gelé)

```
Input (224 × 224 × 3)
        ↓
MobileNetV2 (poids ImageNet — base_model.trainable = False)
  └── Toutes les couches GELÉES
  └── Extraction de features sans réentraînement
        ↓
GlobalAveragePooling2D
        ↓
Dropout(0.2)
        ↓
Dense(1, sigmoid)   ← 0 = Sain / 1 = Tumeur
```

**Pourquoi geler tout MobileNetV2 ?**
Le dataset (~10 000 images) est relativement modeste. Geler entièrement la base évite l'overfitting et permet d'obtenir d'excellentes performances en ne réentraînant que le classificateur final (3 couches seulement).

---

## 🔄 Pipeline complet

**Étape 1 — Connexion & extraction**
```python
drive.mount('/content/drive')
zipfile.ZipFile('Dataset.zip').extractall('/content/dataset')
```

**Étape 2 — Data Augmentation** *(train uniquement)*
| Transformation | Valeur |
|----------------|--------|
| Rotation | ±20° |
| Décalage horizontal | 10% |
| Décalage vertical | 10% |
| Cisaillement | 10% |
| Zoom | 10% |
| Miroir horizontal | ✅ |
| fill_mode | nearest |
| Normalisation | pixels ÷ 255 |

**Étape 3 — Chargement des générateurs**
- `train_generator` → 6 622 images (80% du train, class_mode='binary')
- `validation_generator` → 1 655 images (20% du train)
- `test_generator` → 1 816 images (**shuffle=False** — important pour la matrice de confusion)

**Étape 4 — Entraînement**
```python
history = model.fit(train_generator, epochs=10, validation_data=validation_generator)
```

> 💡 Seulement **10 époques** suffisent grâce au Transfer Learning !

---

## 📊 Évaluation & Visualisations

- ✅ **Accuracy & Loss** finales sur le dataset de test (1 816 images)
- ✅ **Courbes d'apprentissage** (accuracy et loss sur 10 époques)
- ✅ **Matrice de confusion** 2×2 (heatmap Seaborn)
- ✅ **Rapport de classification** (précision, rappel, F1-score)

---

## ⚖️ Comparaison CNN from Scratch vs MobileNetV2

| Critère | CNN From Scratch | MobileNetV2 |
|---------|-----------------|-------------|
| Époques | 15 | **10** |
| Taille image | 150×150 | **224×224** |
| Paramètres entraînables | Beaucoup | **Très peu** |
| Précision attendue | ~80–85% | **~90–95%** |
| Temps d'entraînement | Plus long | **Plus rapide** |
| Risque overfitting | Élevé | Faible |

👉 Voir [CNN_Tumeurs](../CNN_Tumeurs/) pour la version from scratch.

---

## 🚀 Utilisation

1. Ouvrez `CNN_MobileNet_Tumeurs.ipynb` sur **Google Colab**
2. Placez `Dataset.zip` dans `MyDrive/Projet_CNN/`
3. Exécutez les cellules dans l'ordre *(Runtime → Run all)*

---

## 📦 Librairies

```python
tensorflow  keras  numpy  matplotlib  seaborn  sklearn
```
