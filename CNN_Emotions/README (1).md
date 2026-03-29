# 🎭 CNN Reconnaissance des Émotions — MobileNetV2 + Fine-Tuning

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![MobileNetV2](https://img.shields.io/badge/Transfer_Learning-MobileNetV2-red)
![Google Colab](https://img.shields.io/badge/Google-Colab-yellow)

Projet de classification des émotions faciales basé sur **MobileNetV2** avec **Fine-Tuning**. Le modèle est capable de reconnaître plusieurs émotions humaines à partir d'images.

---

## 🎯 Objectif

Entraîner un CNN capable de **détecter et classifier automatiquement les émotions** présentes sur des visages humains, en s'appuyant sur le Transfer Learning avec MobileNetV2.

---

## 🗂️ Dataset

- **Fichier** : `emotions3.zip` (stocké sur Google Drive dans `Projet_CNN/`)
- **Structure** : dossiers `train/` et `test/`, un sous-dossier par émotion
- **Chargement** : montage Google Drive → extraction automatique

---

## ⚙️ Paramètres du modèle

| Paramètre | Valeur |
|-----------|--------|
| Taille des images | 96 × 96 px |
| Batch size | 32 |
| Époques | 15 |
| Learning rate | 1e-4 (Adam) |
| Split validation | 20% |
| Loss | Categorical Crossentropy (label smoothing 0.1) |

---

## 🏗️ Architecture

Le modèle repose sur **MobileNetV2** pré-entraîné sur ImageNet, avec Fine-Tuning partiel :

```
MobileNetV2 (poids ImageNet)
  └── 80 premières couches gelées
  └── Couches suivantes entraînables (Fine-Tuning)
        ↓
GlobalAveragePooling2D
        ↓
Dense(256, activation='relu')
        ↓
Dropout(0.5)
        ↓
Dense(N_classes, activation='softmax')   ← Classification multi-classes
```

---

## 🔄 Pipeline d'entraînement

**1. Connexion Google Drive & extraction du dataset**
```python
drive.mount('/content/drive')
zipfile.ZipFile(chemin_zip).extractall(dossier_extraction)
```

**2. Data Augmentation (train uniquement)**
- Rotation : ±20°
- Zoom : 20%
- Décalages horizontal/vertical : 15%
- Variation de luminosité : 0.7 → 1.3
- Miroir horizontal
- Normalisation : pixels divisés par 255

**3. Chargement des générateurs**
- `train_generator` → 80% des données d'entraînement
- `validation_generator` → 20% des données d'entraînement
- `test_generator` → dossier test séparé (sans augmentation)

**4. Entraînement**
```python
history = model.fit(train_generator, epochs=15, validation_data=validation_generator)
```

---

## 📊 Évaluation

Le modèle est évalué sur le dataset de test avec :
- **Accuracy & Loss** finales
- **Courbes d'apprentissage** (accuracy et loss par époque)
- **Matrice de confusion** (heatmap Seaborn)
- **Rapport de classification** (précision, rappel, F1-score par classe)
- **Test visuel** sur 5 images aléatoires (vrai label vs prédiction)

---

## 🚀 Utilisation

1. Ouvrez le notebook sur **Google Colab**
2. Placez `emotions3.zip` dans `MyDrive/Projet_CNN/`
3. Exécutez les cellules dans l'ordre
4. Le modèle se connecte au Drive, extrait le dataset et s'entraîne automatiquement

---

## 📦 Librairies requises

```python
tensorflow, keras, numpy, matplotlib, seaborn, sklearn
```
*(toutes disponibles nativement sur Google Colab)*
