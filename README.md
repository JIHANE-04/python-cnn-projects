# 🧠 Projets Deep Learning — CNN & Transfer Learning

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Keras](https://img.shields.io/badge/Keras-MobileNetV2-red)
![Google Colab](https://img.shields.io/badge/Google-Colab-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

Collection de **4 projets de Deep Learning** réalisés en Python sur Google Colab, dans le cadre du module **Python Avancé** (filière GSEIR-3 — ENSA Oujda). Chaque projet applique des réseaux de neurones convolutifs (CNN) à un domaine différent : reconnaissance d'émotions, détection de masque, et diagnostic de tumeurs cérébrales.

---

## 📁 Projets

| # | Projet | Type CNN | Dataset | Classes | Images totales |
|---|--------|----------|---------|---------|----------------|
| 1 | [🎭 CNN Émotions](./CNN_Emotions/) | MobileNetV2 + Fine-Tuning | emotions3.zip | 5 émotions | ~47 720 |
| 2 | [😷 CNN Masque](./CNN_Mask/) | MobileNetV2 + Transfer Learning | mask_dataset.zip | 2 classes | ~7 553 |
| 3 | [🧬 CNN Tumeurs](./CNN_Tumeurs/) | CNN from Scratch | Dataset.zip | 2 classes | ~10 093 |
| 4 | [🧬 CNN Tumeurs MobileNet](./CNN_MobileNet_Tumeurs/) | MobileNetV2 | Dataset.zip | 2 classes | ~10 093 |

---

## 🛠️ Stack technique commune

| Outil | Rôle |
|-------|------|
| Python 3.10 | Langage principal |
| TensorFlow / Keras | Framework Deep Learning |
| MobileNetV2 | Modèle pré-entraîné (ImageNet) |
| ImageDataGenerator | Chargement + Data Augmentation |
| Scikit-learn | Matrice de confusion, rapport de classification |
| Matplotlib / Seaborn | Courbes d'apprentissage, visualisation |
| Google Colab | Environnement d'exécution (GPU gratuit) |
| Google Drive | Stockage des datasets |

---

## 🗂️ Datasets

Les datasets sont stockés sur Google Drive dans le dossier `MyDrive/Projet_CNN/` :

| Fichier zip | Projet(s) | Classes |
|-------------|-----------|---------|
| `emotions3.zip` | CNN Émotions | colere, joie, neutre, peur, tristesse |
| `mask_dataset.zip` | CNN Masque | with_mask, without_mask |
| `Dataset.zip` | CNN Tumeurs & CNN MobileNet Tumeurs | Tumeur, Sain |

> 📌 Pour utiliser les notebooks, montez votre Google Drive et placez les fichiers zip dans `MyDrive/Projet_CNN/`. Le code gère l'extraction automatiquement.

---

## 👩‍💻 Auteure

**Jihane Bouras** — GSEIR-3, ENSA Oujda — Année académique 2024/2025
