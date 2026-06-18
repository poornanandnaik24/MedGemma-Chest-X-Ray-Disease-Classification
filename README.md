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


🛠️ Tech Stack
Machine Learning & NLP: Python, PyTorch (MPS enabled), Jupyter Notebook, Hugging Face Transformers

Model: MedGemma

Frontend: Web development framework (contained in frontend/)

Deployment: Hugging Face Spaces

🚀 Getting Started
Prerequisites
Python 3.8+

Node.js & npm (for the frontend)

Jupyter environment

Installation
Clone the repository:

Bash
git clone [https://github.com/poornanandnaik24/MedGemma-Chest-X-Ray-Disease-Classification.git](https://github.com/poornanandnaik24/MedGemma-Chest-X-Ray-Disease-Classification.git)
cd MedGemma-Chest-X-Ray-Disease-Classification
Model Setup (Jupyter):

Open mps llm ver2-medgemma.ipynb in your Jupyter environment.

Install the required Python dependencies (e.g., transformers, torch).

Ensure your Hugging Face credentials are set up if downloading gated models.

Frontend Setup:

Bash
cd frontend
npm install
npm start
Hugging Face Space Deployment:

Navigate to the hf_space_app/ directory.

Push the contents of this directory to your Hugging Face Space repository to initialize the live app.

🧠 Usage
Run the local backend/model inference via the provided Jupyter Notebook.

Launch the frontend application to access the user interface.

Upload a Chest X-Ray image through the UI to generate a classification report utilizing the MedGemma model pipeline.

👨‍💻 Author & Citation
Poornanand Purushottam Naik Department of Computer Science and Engineering National Institute of Technology Karnataka (NITK), Surathkal If you use this project or code in your research, please consider giving the repository a star.

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
