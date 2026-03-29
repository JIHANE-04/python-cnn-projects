# 🧬 CNN Détection de Tumeurs Cérébrales — CNN from Scratch

> 📁 Dossier : `CNN_Tumeurs/` — Fichier : `CNN_Tumeurs.ipynb`

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![CNN](https://img.shields.io/badge/CNN-From_Scratch-purple)
![Google Colab](https://img.shields.io/badge/Google-Colab-yellow)
![Accuracy](https://img.shields.io/badge/Accuracy-95.04%25-brightgreen)

Projet de détection de tumeurs cérébrales sur images IRM, en construisant un **CNN from scratch** (sans Transfer Learning). Le modèle effectue une **classification binaire** : tumeur / sain. Ce projet sert de baseline à comparer avec la version MobileNetV2.

---

## 🏆 Résultats obtenus

| Métrique | Valeur |
|----------|--------|
| 🎯 Accuracy (test) | **95.04%** |
| 📉 Loss (test) | **0.1054** |
| F1-Score (no tumor) | **0.95** |
| F1-Score (tumor) | **0.95** |
| F1-Score macro | **0.95** |

### Rapport de classification détaillé

| Classe | Précision | Rappel | F1-Score | Support |
|--------|-----------|--------|----------|---------|
| no tumor | 0.91 | 1.00 | 0.95 | 910 |
| tumor | 1.00 | 0.90 | 0.95 | 906 |
| **accuracy** | | | **0.95** | **1816** |

---

## 🎯 Objectif

Concevoir et entraîner un réseau de neurones convolutif **entièrement from scratch** pour détecter la présence ou l'absence d'une tumeur cérébrale sur des images IRM, sans utiliser de modèle pré-entraîné.

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

---

## ⚙️ Paramètres du modèle

| Paramètre | Valeur |
|-----------|--------|
| Taille des images | 150 × 150 px |
| Batch size | 32 |
| Époques | 15 |
| Optimiseur | Adam |
| Split validation | 80% train / 20% validation |
| Loss | Binary Crossentropy |
| Activation sortie | Sigmoid (1 neurone) |
| Type de classification | Binaire (0 = Sain / 1 = Tumeur) |

---

## 🏗️ Architecture CNN from Scratch

```
Input (150 × 150 × 3)
        ↓
Conv2D(32 filtres, 3×3, relu)
MaxPooling2D(2×2)
        ↓
Conv2D(64 filtres, 3×3, relu)
MaxPooling2D(2×2)
        ↓
Conv2D(128 filtres, 3×3, relu)
MaxPooling2D(2×2)
        ↓
Flatten
        ↓
Dense(128, relu)
        ↓
Dropout(0.5)
        ↓
Dense(1, sigmoid)   ← 0 = Sain / 1 = Tumeur
```

**Progression des filtres (32 → 64 → 128) :**
- **32 filtres** : détectent les formes simples (bords, contours)
- **64 filtres** : capturent des formes plus complexes (motifs, textures)
- **128 filtres** : identifient des caractéristiques fines spécifiques aux tumeurs

**Dropout(0.5)** : désactive aléatoirement 50% des neurones pour éviter l'overfitting.

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
- `train_generator` → 6 622 images (80%, shuffle=True)
- `validation_generator` → 1 655 images (20%, shuffle=True)
- `test_generator` → 1 816 images (**shuffle=False** pour la matrice de confusion)

**Étape 4 — Entraînement**
```python
history = model.fit(train_generator, epochs=15, validation_data=validation_generator)
```

---

## 📊 Évaluation & Visualisations

- ✅ **Accuracy & Loss** finales sur le dataset de test (1 816 images)
- ✅ **Courbes d'apprentissage** (accuracy et loss sur 15 époques)
- ✅ **Matrice de confusion** 2×2 (heatmap Seaborn)
- ✅ **Rapport de classification** (précision, rappel, F1-score)
- ✅ **Test visuel** sur 5 images aléatoires avec label prédit

---

## ⚖️ Comparaison avec MobileNetV2

| Critère | CNN From Scratch | MobileNetV2 |
|---------|-----------------|-------------|
| Époques | 15 | **10** |
| Taille image | 150×150 | **224×224** |
| Paramètres entraînables | Beaucoup | **Très peu** |
| Accuracy obtenue | **95.04%** | **~95%+** |
| Temps d'entraînement | Plus long | **Plus rapide** |

👉 Voir [CNN_MobileNet_Tumeurs](../CNN_MobileNet_Tumeurs/) pour la version Transfer Learning.

---

## 🚀 Utilisation

1. Ouvrez `CNN_Tumeurs.ipynb` sur **Google Colab**
2. Placez `Dataset.zip` dans `MyDrive/Projet_CNN/`
3. Exécutez les cellules dans l'ordre *(Runtime → Run all)*

---

## 📦 Librairies

```python
tensorflow  keras  numpy  matplotlib  seaborn  sklearn
```

