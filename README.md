# 🌱 Green AI: Energy-Aware Model Selector

## 📌 Introduction
Machine learning models are usually selected based only on accuracy, while their **energy consumption and environmental impact** are ignored.  
This project focuses on **Green AI**, where the goal is to build models that are not only accurate but also **energy efficient and sustainable**.

The Energy-Aware Model Selector automatically evaluates multiple machine learning models and recommends the best one by balancing **performance and energy usage**.

---

## 🎯 Problem Statement
Modern ML pipelines often choose complex models that:
- Consume high computational power
- Increase carbon footprint
- Are inefficient for real-world deployment

**Core Question:**  
> How can we select a machine learning model that provides good accuracy while minimizing energy consumption?

---

## 🧠 Proposed Solution
This project:
- Trains multiple ML models on the same dataset
- Tracks energy consumption during training and inference
- Measures standard ML performance metrics
- Uses a multi-objective scoring strategy
- Recommends the most energy-efficient model

---

## 🏗️ System Architecture

```
Dataset
  ↓
Data Preprocessing
  ↓
Train Multiple Models
  ↓
Energy & Performance Monitoring
  ↓
Scoring & Comparison Engine
  ↓
Best Model Recommendation
  ↓
Visualization Dashboard
```

---

## ⚙️ Tech Stack

### 🔹 Programming Language
- Python

### 🔹 Libraries & Tools
- **Machine Learning**: Scikit-learn, XGBoost
- **Energy Monitoring**: CodeCarbon, psutil
- **Data Processing**: Pandas, NumPy
- **Visualization**: Streamlit, Matplotlib

---

## 🤖 Models Implemented
- Linear Regression
- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost
- Simple Neural Network (baseline)

---

## 📊 Evaluation Metrics

### 🔸 Performance Metrics
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

### 🔸 Energy & Efficiency Metrics
- Energy consumption (kWh)
- Training time (seconds)
- Inference latency (milliseconds)

---

## 🧮 Model Selection Strategy
A weighted scoring approach is used:

```
Final Score = α × Performance − β × Energy Consumption
```

Where:
- **α** controls importance of accuracy
- **β** controls importance of energy efficiency

---

## 🖥️ Dashboard Features
The Streamlit dashboard provides:
- Accuracy vs Energy comparison plots
- Model-wise performance table
- Highlighted recommended model
- Energy usage insights

---

## 🚀 How to Run the Project

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/yash-755/green-ai-energy-aware-model-selector.git
cd green-ai-energy-aware-model-selector
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Train Models & Track Energy
```bash
python src/train_models.py
```
This step trains multiple models while monitoring their energy consumption using CodeCarbon.

### 4️⃣ Launch the Dashboard
```bash
streamlit run dashboard/app.py
```

---

## 📁 Project Structure
```
green-ai-energy-aware-model-selector/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── data_preprocessing.py
│   ├── train_models.py
│   ├── energy_monitor.py
│   ├── model_selector.py
│   └── evaluation.py
│
├── experiments/
│   └── experiment_logs.csv
│
├── models/
│   └── trained_models/
│
├── dashboard/
│   └── app.py
│
├── reports/
│
├── requirements.txt
├── .gitignore
├── README.md
└── LICENSE
```

---

## 🔬 Research Motivation
This project follows **Green AI principles**, focusing on reducing unnecessary computation and encouraging responsible machine learning practices without sacrificing performance.

---

## 🚧 Future Improvements
- Model pruning and quantization
- GPU-level power monitoring
- Dataset-aware energy prediction
- AutoML-based model selection
- Carbon footprint estimation

---

## 👤 Author
**Yash Uttam**  

---

## 📜 License
This project is licensed under the **MIT License**.
