# 😷 CNN Détection de Masque — MobileNetV2 + Webcam

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![MobileNetV2](https://img.shields.io/badge/Transfer_Learning-MobileNetV2-red)
![Google Colab](https://img.shields.io/badge/Google-Colab-yellow)
![OpenCV](https://img.shields.io/badge/OpenCV-cv2-green)

Projet de détection du port du masque en temps réel via **webcam**, basé sur **MobileNetV2** avec Transfer Learning. Le modèle classifie chaque image capturée comme **Avec masque** ou **Sans masque**.

---

## 🎯 Objectif

Entraîner un CNN capable de **détecter automatiquement si une personne porte un masque** sur son visage, avec une démonstration en temps réel via la webcam de l'ordinateur.

---

## 🗂️ Dataset

- **Fichier** : `mask_dataset.zip` (stocké sur Google Drive dans `Projet_CNN/`)
- **Structure** : dossiers `train/` et `test/` avec sous-dossiers par classe
- **Classes** : `with_mask` / `without_mask`
- **Split** : 80% entraînement / 20% validation

---

## ⚙️ Paramètres du modèle

| Paramètre | Valeur |
|-----------|--------|
| Taille des images | 96 × 96 px |
| Batch size | 32 |
| Époques | 15 |
| Optimiseur | Adam |
| Split validation | 20% |
| Loss | Categorical Crossentropy |

---

## 🏗️ Architecture

```
MobileNetV2 (poids ImageNet)
  └── 80 premières couches gelées
        ↓
GlobalAveragePooling2D
        ↓
Dropout(0.5)
        ↓
Dense(N_classes, activation='softmax')   ← Avec masque / Sans masque
```

---

## 🔄 Pipeline d'entraînement

**1. Connexion Google Drive & extraction**
```python
drive.mount('/content/drive')
zipfile.ZipFile(chemin_zip).extractall('/content/dataset_mask/...')
```

**2. Data Augmentation**
- Rotation : ±20°
- Zoom : 20%
- Miroir horizontal
- Normalisation : pixels ÷ 255

**3. Générateurs**
- `train_gen` → 80% du dataset
- `val_gen` → 20% du dataset (validation)

**4. Entraînement**
```python
history = model.fit(train_gen, epochs=15, validation_data=val_gen)
```

---

## 📷 Fonctionnalité Webcam (temps réel)

Ce projet inclut une démonstration en temps réel via la **webcam du navigateur** :
- Capture d'une image via JavaScript dans Google Colab
- Traitement de l'image avec **OpenCV**
- Prédiction du modèle en temps réel
- Affichage du résultat avec rectangle et texte sur l'image

```python
from IPython.display import display, Javascript
import cv2
```

---

## 📊 Évaluation

- **Accuracy & Loss** finales sur le dataset de validation
- **Courbes d'apprentissage** par époque
- **Matrice de confusion** (heatmap Seaborn)
- **Rapport de classification** (précision, rappel, F1-score)
- **Test webcam** en temps réel

---

## 🚀 Utilisation

1. Ouvrez le notebook sur **Google Colab**
2. Placez `mask_dataset.zip` dans `MyDrive/Projet_CNN/`
3. Exécutez les cellules dans l'ordre
4. Autorisez l'accès à la webcam quand demandé

---

## 📦 Librairies requises

```python
tensorflow, keras, numpy, matplotlib, seaborn, sklearn, opencv-python
```
