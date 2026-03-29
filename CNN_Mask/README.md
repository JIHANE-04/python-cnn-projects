# 😷 CNN Détection de Masque — MobileNetV2 + Webcam temps réel

> 📁 Dossier : `CNN_Mask/` — Fichier : `CNN_Mask.ipynb`

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![MobileNetV2](https://img.shields.io/badge/Transfer_Learning-MobileNetV2-red)
![OpenCV](https://img.shields.io/badge/OpenCV-cv2-green)
![Google Colab](https://img.shields.io/badge/Google-Colab-yellow)

Projet de détection du port du masque basé sur **MobileNetV2** avec Transfer Learning, enrichi d'une démonstration en **temps réel via webcam**. Le modèle classifie chaque image comme **avec masque** ou **sans masque**.

---

## 🎯 Objectif

Entraîner un CNN capable de **détecter automatiquement si une personne porte un masque**, avec une interface de test en temps réel via la webcam du navigateur dans Google Colab.

---

## 🗂️ Dataset — `mask_dataset.zip`

| Élément | Détail |
|---------|--------|
| Fichier | `mask_dataset.zip` → `MyDrive/Projet_CNN/mask_dataset.zip` |
| Extraction | `/content/dataset_mask/data/data/data/data` |
| Nombre de classes | **2 classes** |
| Classes | `with_mask` · `without_mask` |
| Images d'entraînement (80%) | **6 043 images** |
| Images de validation (20%) | **1 510 images** |
| **Total** | **~7 553 images** |

> Le dataset est divisé automatiquement en 80% entraînement / 20% validation via `validation_split=0.2`.

---

## ⚙️ Paramètres du modèle

| Paramètre | Valeur |
|-----------|--------|
| Taille des images | 96 × 96 px |
| Batch size | 32 |
| Époques | 15 |
| Optimiseur | Adam |
| Split validation | 80% train / 20% validation |
| Loss | Categorical Crossentropy |
| Activation sortie | Softmax (2 classes) |

---

## 🏗️ Architecture — MobileNetV2 Transfer Learning

```
Input (96 × 96 × 3)
        ↓
MobileNetV2 (poids ImageNet)
  └── 80 premières couches GELÉES
  └── Couches suivantes ENTRAÎNABLES
        ↓
GlobalAveragePooling2D
        ↓
Dropout(0.5)
        ↓
Dense(2, activation='softmax')   ← with_mask / without_mask
```

---

## 🔄 Pipeline complet

**Étape 1 — Connexion & extraction**
```python
drive.mount('/content/drive')
zipfile.ZipFile('mask_dataset.zip').extractall('/content/dataset_mask/...')
```

**Étape 2 — Data Augmentation** *(train uniquement)*
| Transformation | Valeur |
|----------------|--------|
| Rotation | ±20° |
| Zoom | 20% |
| Miroir horizontal | ✅ |
| Normalisation | pixels ÷ 255 |

**Étape 3 — Chargement des générateurs**
- `train_gen` → 6 043 images (80%)
- `val_gen` → 1 510 images (20%)

**Étape 4 — Entraînement**
```python
history = model.fit(train_gen, epochs=15, validation_data=val_gen)
```

---

## 📷 Fonctionnalité Webcam (temps réel)

Ce projet inclut une démonstration unique : la **détection en temps réel via webcam** directement dans Google Colab.

**Fonctionnement :**
1. Capture d'une image via JavaScript dans le navigateur
2. Décodage Base64 de l'image capturée
3. Traitement avec **OpenCV** (redimensionnement, normalisation)
4. Prédiction instantanée du modèle
5. Affichage du résultat avec rectangle et texte superposés sur l'image

```python
from IPython.display import display, Javascript
from google.colab.output import eval_js
from base64 import b64decode
import cv2
```

---

## 📊 Évaluation & Visualisations

- ✅ **Accuracy & Loss** finales sur la validation
- ✅ **Courbes d'apprentissage** (accuracy et loss sur 15 époques)
- ✅ **Matrice de confusion** 2×2 (heatmap Seaborn)
- ✅ **Rapport de classification** (précision, rappel, F1-score)
- ✅ **Test webcam** en temps réel

---

## 🚀 Utilisation

1. Ouvrez `CNN_Mask.ipynb` sur **Google Colab**
2. Placez `mask_dataset.zip` dans `MyDrive/Projet_CNN/`
3. Exécutez les cellules dans l'ordre
4. Autorisez l'accès à la **webcam** quand demandé par le navigateur

---

## 📦 Librairies

```python
tensorflow  keras  numpy  matplotlib  seaborn  sklearn  opencv-python
```
