# 💳 Credit Card Fraud Detection

A machine learning web application that predicts potentially fraudulent credit card transactions using a trained Random Forest classifier.

## Features
- Fraudulent transaction classification
- Flask-based web interface
- Pre-trained machine learning model
- Jupyter Notebook for experimentation

## Tech Stack
- Python
- Flask
- Scikit-learn
- Random Forest
- HTML, CSS, JavaScript
- Joblib

## Project Structure

```text
credit-card-fraud-detection/
├── templates/
│   └── index.html
├── CreditCard.ipynb
├── fraud_model.joblib
├── main.py
├── requirements.txt
├── train_model.py
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/sumi-0648/credit-card-fraud-detection-.git
cd credit-card-fraud-detection-
```

Create a virtual environment and install dependencies:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Run the application:

```bash
python main.py
```

Open `http://127.0.0.1:5000` in your browser.

## Dataset

The original `creditcard.csv` dataset is not included because it exceeds GitHub's standard 100 MB file limit. Obtain it from an authorized source if required for model training.

## Disclaimer

This project is intended for educational purposes. It is not a production-grade financial fraud detection system.

## Author

**Perumal Sumithra**

GitHub: [@sumi-0648](https://github.com/sumi-0648)