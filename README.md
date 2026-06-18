# MedGemma Chest X-Ray Disease Classification

[![GitHub stars](https://img.shields.io/github/stars/poornanandnaik24/MedGemma-Chest-X-Ray-Disease-Classification.svg)](https://github.com/poornanandnaik24/MedGemma-Chest-X-Ray-Disease-Classification/stargazers)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📌 Project Overview
This repository contains a comprehensive pipeline for automated chest X-ray disease classification leveraging **MedGemma**, a medical-domain Large Language Model. The project bridges state-of-the-art deep learning with an accessible web interface, providing tools for both model inference and user interaction. It includes a frontend application and configurations for seamless deployment on Hugging Face Spaces.

## ✨ Key Features
* **Advanced Medical AI:** Utilizes MedGemma for robust disease classification and analysis of thoracic datasets.
* **Interactive Frontend:** A dedicated web interface (`frontend/`) to easily upload X-ray images and view classification results.
* **Hugging Face Integration:** Ready-to-deploy application structure (`hf_space_app/`) for hosting the model and interface on Hugging Face Spaces.
* **Apple Silicon / MPS Support:** Jupyter notebooks optimized for Metal Performance Shaders (MPS) to accelerate local LLM inference on macOS.

## 📂 Repository Structure
```text
📦 MedGemma-Chest-X-Ray-Disease-Classification
 ┣ 📂 .github/workflows           # CI/CD pipelines and GitHub actions
 ┣ 📂 frontend                    # Web application interface source code
 ┣ 📂 hf_space_app                # Application files configured for Hugging Face Spaces
 ┣ 📜 mps llm ver2-medgemma.ipynb # Core Jupyter Notebook for MedGemma model execution
 ┣ 📜 .gitignore                  # Git ignore rules
 ┗ 📜 README.md                   # Project documentation