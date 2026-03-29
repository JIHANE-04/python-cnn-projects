# 🧠 Projets Deep Learning — CNN & Transfer Learning

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Keras](https://img.shields.io/badge/Keras-MobileNetV2-red)
![Google Colab](https://img.shields.io/badge/Google-Colab-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

Collection de **4 projets de Deep Learning** réalisés en **Python** sur **Google Colab**, dans le cadre du module **Python Avancé** (GSEIR-3 — ENSA Oujda). Chaque projet applique des réseaux de neurones convolutifs (CNN) à un domaine différent.

---

## 📁 Projets

| # | Projet | Type CNN | Dataset | Classes | Accuracy |
|---|--------|----------|---------|---------|----------|
| 1 | [🎭 CNN Émotions](./CNN_Emotions/) | MobileNetV2 + Fine-Tuning | emotions3.zip | 5 émotions | **76.76%** |
| 2 | [😷 CNN Masque](./CNN_Mask/) | MobileNetV2 + Transfer Learning | mask_dataset.zip | Avec/Sans masque | **99.07%** |
| 3 | [🧬 CNN Tumeurs](./CNN_Tumeurs/) | CNN from scratch | Dataset.zip | Tumeur / Sain | **95.04%** |
| 4 | [🧬 CNN Tumeurs MobileNet](./CNN_MobileNet_Tumeurs/) | MobileNetV2 | Dataset.zip | Tumeur / Sain | **~95%+** |

---

## 🏆 Résultats globaux

| Projet | Accuracy Test | F1-Score | Époques | Dataset |
|--------|--------------|----------|---------|---------|
| 🎭 CNN Émotions | **76.76%** | 0.77 | 15 | 47 720 images — 5 classes |
| 😷 CNN Masque | **99.07%** | 1.00 | 15 | 7 553 images — 2 classes |
| 🧬 CNN Tumeurs (scratch) | **95.04%** | 0.95 | 15 | 10 093 images — 2 classes |
| 🧬 CNN Tumeurs MobileNet | **~95%+** | — | 10 | 10 093 images — 2 classes |

---

## 🛠️ Stack technique commune

- **Python 3.10** — Google Colab
- **TensorFlow / Keras** — Deep Learning
- **MobileNetV2** — Transfer Learning (ImageNet)
- **ImageDataGenerator** — Data Augmentation
- **Scikit-learn** — Évaluation (matrice de confusion, rapport)
- **Matplotlib / Seaborn** — Visualisation des résultats
- **Google Drive** — Stockage des datasets

---

## 🗂️ Datasets (Google Drive)

Les datasets sont stockés sur Google Drive dans le dossier `Projet_CNN/` :

| Fichier | Utilisé par | Taille |
|---------|-------------|--------|
| `emotions3.zip` | CNN Émotions | ~47 720 images |
| `mask_dataset.zip` | CNN Masque | ~7 553 images |
| `Dataset.zip` | CNN Tumeurs & CNN MobileNet Tumeurs | ~10 093 images |

> 📌 Pour utiliser ces notebooks, montez votre Google Drive et placez les fichiers zip dans `MyDrive/Projet_CNN/`.

---

## 📁 Structure du dépôt

```
deep-learning-CNN/
├── CNN_Emotions/
│   ├── CNN_EMOTIONS.ipynb
│   └── README.md
├── CNN_Mask/
│   ├── CNN_Mask.ipynb
│   └── README.md
├── CNN_Tumeurs/
│   ├── CNN_Tumeurs.ipynb
│   └── README.md
├── CNN_MobileNet_Tumeurs/
│   ├── CNN_MobileNet_Tumeurs.ipynb
│   └── README.md
└── README.md   ← ce fichier
```

---

## 👩‍💻 Auteur

**Jihane Bouras** — GSEIR-3, ENSA Oujda — Année 2024/2025

