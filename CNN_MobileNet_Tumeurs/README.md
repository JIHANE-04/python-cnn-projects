# 🧬 CNN Détection de Tumeurs — MobileNetV2 Transfer Learning

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![MobileNetV2](https://img.shields.io/badge/Transfer_Learning-MobileNetV2-red)
![Google Colab](https://img.shields.io/badge/Google-Colab-yellow)

Projet de détection de tumeurs cérébrales basé sur **MobileNetV2** (Transfer Learning). Version améliorée du [CNN from scratch](../CNN_Tumeurs/), atteignant une précision élevée en seulement **10 époques**.

---

## 🎯 Objectif

Exploiter la puissance du **Transfer Learning** avec MobileNetV2 pour détecter la présence ou l'absence d'une tumeur cérébrale sur des images IRM, en comparaison avec le CNN from scratch.

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
| Taille des images | 224 × 224 px |
| Batch size | 32 |
| Époques | 10 |
| Optimiseur | Adam |
| Split validation | 20% |
| Loss | Binary Crossentropy |
| Sortie | Sigmoid (1 neurone) |

> ✅ MobileNetV2 fonctionne de manière optimale avec des images **224×224** (taille native d'ImageNet).

---

## 🏗️ Architecture

```
MobileNetV2 (poids ImageNet — totalement gelé)
  └── base_model.trainable = False
        ↓
GlobalAveragePooling2D
        ↓
Dropout(0.2)
        ↓
Dense(1, sigmoid)   ← 0 = Sain / 1 = Tumeur
```

**Pourquoi geler tout MobileNetV2 ?**
- Le dataset tumeurs est relativement petit
- Les poids ImageNet sont déjà excellents pour extraire des features
- On évite l'overfitting en ne réentraînant que la tête de classification

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
- `fill_mode='nearest'`
- Normalisation : pixels ÷ 255

**3. Générateurs**
- `train_generator` → 80% du dossier train
- `validation_generator` → 20% du dossier train
- `test_generator` → dossier test séparé (shuffle=False)

**4. Entraînement**
```python
history = model.fit(train_generator, epochs=10, validation_data=validation_generator)
```

> 💡 Seulement **10 époques** suffisent grâce au Transfer Learning (vs 15 pour le CNN from scratch) !

---

## 📊 Évaluation

- **Accuracy & Loss** finales sur le dataset de test
- **Courbes d'apprentissage** (accuracy et loss par époque)
- **Matrice de confusion** (heatmap Seaborn)
- **Rapport de classification** (précision, rappel, F1-score)

---

## ⚖️ Comparaison CNN from scratch vs MobileNetV2

| Critère | CNN From Scratch | MobileNetV2 |
|---------|-----------------|-------------|
| Époques | 15 | 10 |
| Taille image | 150×150 | 224×224 |
| Paramètres entraînables | Beaucoup | Très peu |
| Précision attendue | ~80-85% | ~90-95% |
| Temps d'entraînement | Plus long | Plus rapide |

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
