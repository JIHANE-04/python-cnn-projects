# 🧠 Projets Deep Learning — CNN & Transfer Learning

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Keras](https://img.shields.io/badge/Keras-MobileNetV2-red)
![Google Colab](https://img.shields.io/badge/Google-Colab-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

Collection de 4 projets de Deep Learning réalisés en **Python** sur **Google Colab**, dans le cadre du module **Python Avancé** (GSEIR-3 — ENSA Oujda). Chaque projet applique des réseaux de neurones convolutifs (CNN) à un domaine différent.

---

## 📁 Projets

| # | Projet | Type CNN | Dataset | Classes |
|---|--------|----------|---------|---------|
| 1 | [🎭 CNN Émotions](./CNN_Emotions/) | MobileNetV2 + Fine-Tuning | emotions3.zip | Émotions faciales |
| 2 | [😷 CNN Masque](./CNN_Mask/) | MobileNetV2 + Transfer Learning | mask_dataset.zip | Avec/Sans masque |
| 3 | [🧬 CNN Tumeurs](./CNN_Tumeurs/) | CNN from scratch | Dataset.zip | Tumeur / Sain |
| 4 | [🧬 CNN Tumeurs MobileNet](./CNN_MobileNet_Tumeurs/) | MobileNetV2 | Dataset.zip | Tumeur / Sain |

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

| Fichier | Utilisé par |
|---------|-------------|
| `emotions3.zip` | CNN Émotions |
| `mask_dataset.zip` | CNN Masque |
| `Dataset.zip` | CNN Tumeurs & CNN MobileNet Tumeurs |

> 📌 Pour utiliser ces notebooks, montez votre Google Drive et placez les fichiers zip dans `MyDrive/Projet_CNN/`.

---

## 👩‍💻 Auteur

**Jihane Bouras** — GSEIR-3, ENSA Oujda — Année 2024/2025
