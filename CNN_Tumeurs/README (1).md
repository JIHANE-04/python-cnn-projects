# 🧬 CNN Détection de Tumeurs Cérébrales — CNN from Scratch

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![CNN](https://img.shields.io/badge/CNN-From_Scratch-purple)
![Google Colab](https://img.shields.io/badge/Google-Colab-yellow)

Projet de détection de tumeurs cérébrales à partir d'images IRM, en utilisant un **CNN construit from scratch** (sans Transfer Learning). Le modèle effectue une **classification binaire** : tumeur / sain.

---

## 🎯 Objectif

Concevoir et entraîner un réseau de neurones convolutif **from scratch** capable de **détecter la présence ou l'absence d'une tumeur cérébrale** sur des images IRM.

---

## 🗂️ Dataset

- **Fichier** : `Dataset.zip` (stocké sur Google Drive dans `Projet_CNN/`)
- **Structure** : dossiers `train/` et `test/`, deux classes
- **Classes** : `Tumeur` (1) / `Sain` (0)
- **Type** : Classification binaire

---

## ⚙️ Paramètres du modèle

| Paramètre | Valeur |
|-----------|--------|
| Taille des images | 150 × 150 px |
| Batch size | 32 |
| Époques | 15 |
| Optimiseur | Adam |
| Split validation | 20% |
| Loss | Binary Crossentropy |
| Sortie | Sigmoid (1 neurone) |

---

## 🏗️ Architecture CNN (from scratch)

```
Input (150 × 150 × 3)
        ↓
Conv2D(32, 3×3, relu) → MaxPooling2D(2×2)
        ↓
Conv2D(64, 3×3, relu) → MaxPooling2D(2×2)
        ↓
Conv2D(128, 3×3, relu) → MaxPooling2D(2×2)
        ↓
Flatten
        ↓
Dense(128, relu)
        ↓
Dropout(0.5)
        ↓
Dense(1, sigmoid)   ← 0 = Sain / 1 = Tumeur
```

**Pourquoi cette architecture ?**
- **3 blocs Conv + Pooling** : progression des filtres (32 → 64 → 128) pour capturer des caractéristiques de plus en plus complexes
- **Dropout(0.5)** : évite l'overfitting en désactivant 50% des neurones
- **Sigmoid** : sortie entre 0 et 1 pour la classification binaire

---

## 🔄 Pipeline d'entraînement

**1. Connexion Google Drive & extraction**
```python
drive.mount('/content/drive')
zipfile.ZipFile(chemin_zip).extractall('/content/dataset')
```

**2. Data Augmentation (train uniquement)**
- Rotation : ±20°
- Décalages horizontal/vertical : 10%
- Cisaillement : 10%
- Zoom : 10%
- Miroir horizontal
- Normalisation : pixels ÷ 255

**3. Générateurs**
- `train_generator` → 80% du dossier train
- `validation_generator` → 20% du dossier train
- `test_generator` → dossier test séparé (sans augmentation)

**4. Entraînement**
```python
history = model.fit(train_generator, epochs=15, validation_data=validation_generator)
```

---

## 📊 Évaluation

- **Accuracy & Loss** finales sur le dataset de test
- **Courbes d'apprentissage** (accuracy et loss par époque)
- **Matrice de confusion** (heatmap Seaborn)
- **Rapport de classification** (précision, rappel, F1-score)
- **Test visuel** sur images aléatoires

---

## 🔬 Comparaison avec MobileNetV2

Ce projet est la **version CNN from scratch**. Pour voir les performances avec Transfer Learning (MobileNetV2), consultez le projet [CNN_MobileNet_Tumeurs](../CNN_MobileNet_Tumeurs/).

---

## 🚀 Utilisation

1. Ouvrez le notebook sur **Google Colab**
2. Placez `Dataset.zip` dans `MyDrive/Projet_CNN/`
3. Exécutez les cellules dans l'ordre

---

## 📦 Librairies requises

```python
tensorflow, keras, numpy, matplotlib, seaborn, sklearn
```
