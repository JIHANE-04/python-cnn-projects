# 🎭 CNN Reconnaissance des Émotions — MobileNetV2 + Fine-Tuning

> 📁 Dossier : `CNN_Emotions/` — Fichier : `CNN_EMOTIONS.ipynb`

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![MobileNetV2](https://img.shields.io/badge/Transfer_Learning-MobileNetV2-red)
![Google Colab](https://img.shields.io/badge/Google-Colab-yellow)

Projet de classification des émotions faciales basé sur **MobileNetV2** avec **Fine-Tuning partiel**. Le modèle est capable de reconnaître **5 émotions humaines** à partir d'images de visages.

---

## 🎯 Objectif

Entraîner un CNN capable de **détecter et classifier automatiquement les émotions** présentes sur des visages humains, en exploitant le Transfer Learning avec MobileNetV2 pré-entraîné sur ImageNet.

---

## 🗂️ Dataset — `emotions3.zip`

| Élément | Détail |
|---------|--------|
| Fichier | `emotions3.zip` → `MyDrive/Projet_CNN/emotions3.zip` |
| Extraction | `/content/dataset_emotions` |
| Nombre de classes | **5 émotions** |
| Classes | `colere` · `joie` · `neutre` · `peur` · `tristesse` |
| Images d'entraînement (80%) | **33 904 images** |
| Images de validation (20%) | **8 475 images** |
| Images de test | **5 341 images** |
| **Total** | **~47 720 images** |

> Le dataset est automatiquement détecté et extrait depuis Google Drive. Les dossiers `train/` et `test/` sont détectés dynamiquement par le code.

---

## ⚙️ Paramètres du modèle

| Paramètre | Valeur |
|-----------|--------|
| Taille des images | 96 × 96 px |
| Batch size | 32 |
| Époques | 15 |
| Learning rate | 1e-4 (Adam) |
| Split validation | 80% train / 20% validation |
| Loss | Categorical Crossentropy (label smoothing = 0.1) |
| Activation sortie | Softmax (multi-classes) |

---

## 🏗️ Architecture — MobileNetV2 + Fine-Tuning

```
Input (96 × 96 × 3)
        ↓
MobileNetV2 (poids ImageNet)
  ├── Couches 0–79  : GELÉES  (détectent contours, textures, formes simples)
  └── Couches 80+   : ENTRAÎNABLES (Fine-Tuning — adaptées aux émotions)
        ↓
GlobalAveragePooling2D
        ↓
Dense(256, activation='relu')
        ↓
Dropout(0.5)
        ↓
Dense(5, activation='softmax')   ← 5 émotions
```

**Pourquoi le Fine-Tuning ?**
Le dataset contient plus de 47 000 images, ce qui est suffisant pour dégeler une partie de MobileNetV2 et l'adapter aux visages humains, améliorant ainsi la précision par rapport au Transfer Learning classique (modèle totalement gelé).

---

## 🔄 Pipeline complet

**Étape 1 — Connexion & extraction**
```python
drive.mount('/content/drive')
zipfile.ZipFile('emotions3.zip').extractall('/content/dataset_emotions')
```

**Étape 2 — Data Augmentation** *(train uniquement)*
| Transformation | Valeur |
|----------------|--------|
| Rotation | ±20° |
| Zoom | 20% |
| Décalage horizontal | 15% |
| Décalage vertical | 15% |
| Variation luminosité | 0.7 → 1.3 |
| Miroir horizontal | ✅ |
| Normalisation | pixels ÷ 255 |

**Étape 3 — Chargement des générateurs**
- `train_generator` → 33 904 images (80% du train)
- `validation_generator` → 8 475 images (20% du train)
- `test_generator` → 5 341 images (dossier test, sans augmentation)

**Étape 4 — Entraînement**
```python
history = model.fit(train_generator, epochs=15, validation_data=validation_generator)
```

---

## 📊 Évaluation & Visualisations

- ✅ **Accuracy & Loss** finales sur le dataset de test
- ✅ **Courbes d'apprentissage** (accuracy et loss sur 15 époques)
- ✅ **Matrice de confusion** 5×5 (heatmap Seaborn)
- ✅ **Rapport de classification** (précision, rappel, F1-score par émotion)
- ✅ **Test visuel** sur 5 images aléatoires (vrai label vs prédiction, couleur verte/rouge)

---

## 🚀 Utilisation

1. Ouvrez `CNN_EMOTIONS.ipynb` sur **Google Colab**
2. Placez `emotions3.zip` dans `MyDrive/Projet_CNN/`
3. Exécutez les cellules dans l'ordre *(Runtime → Run all)*
4. Le modèle se connecte, extrait, s'entraîne et évalue automatiquement

---

## 📦 Librairies

```python
tensorflow  keras  numpy  matplotlib  seaborn  sklearn
```
