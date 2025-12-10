# 🏦 Home Credit Risk Prediction

A production-ready credit default prediction model using **LightGBM** with **Cost-Sensitive Learning** to identify potential loan defaulters in highly imbalanced datasets.

## 📊 Project Overview

This project addresses the critical challenge of credit risk assessment in banking by building a machine learning model that predicts loan default probability. The model emphasizes **interpretability** and **regulatory compliance**, making it suitable for real-world financial applications.

### Key Highlights
- ✅ **ROC-AUC Score:** 0.7462
- ✅ **Feature Reduction:** 463 → 20 high-impact features
- ✅ **Cost-Sensitive Learning:** Handles severe class imbalance
- ✅ **Model Explainability:** SHAP & LIME integration
- ✅ **Production-Ready:** Strict data leakage prevention

## 🎯 Problem Statement

Credit default prediction faces two major challenges:
1. **Severe Class Imbalance:** Only ~8% of applicants default
2. **High Cost of Misclassification:** Missing a defaulter is more costly than rejecting a good applicant

This project uses **Cost-Sensitive Learning** to prioritize recall while maintaining acceptable precision.

## 🚀 Methodology

### 1. Data Pipeline
- Integrated multiple credit bureau tables (application, bureau, previous loans)
- Created comprehensive customer risk profiles
- Generated 463 initial features through feature engineering

### 2. Feature Selection (Multi-Stage)
```
Stage 1: Mutual Information Filtering
    ↓
Stage 2: Correlation Removal (>0.95)
    ↓
Stage 3: Permutation Importance Ranking
    ↓
Stage 4: SHAP-Based Selection
    ↓
Result: 20 High-Impact Features
```

### 3. Model Training
- **Algorithm:** LightGBM Classifier
- **Class Weighting:** `scale_pos_weight = 11.39`
- **Threshold Optimization:** 0.5 → 0.6760
- **Validation:** Stratified Train-Test Split

### 4. Model Interpretation
- **SHAP Values:** Global feature importance
- **LIME:** Local instance-level explanations
- Ensures audit-readiness for regulatory compliance

## 📈 Results

| Metric | Score |
|--------|-------|
| ROC-AUC | 0.7462 |
| F1-Score | 0.3010 |
| Precision | 0.2475 |
| Recall | 0.3839 |
| Optimal Threshold | 0.6760 |

### Top 5 Risk Drivers
1. **EXT_SOURCE_3** - External credit score (source 3)
2. **EXT_SOURCE_2** - External credit score (source 2)
3. **AMT_ANNUITY** - Loan annuity amount
4. **EXT_SOURCE_1** - External credit score (source 1)
5. **AMT_CREDIT_SUM_DEBT** - Current total debt

## 📁 Tech Stack

- **Language:** Python 3.8+
- **ML Framework:** LightGBM, Scikit-learn
- **Interpretability:** SHAP, LIME
- **Data Processing:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn

## 🔧 Installation

```bash
# Clone the repository
git clone https://github.com/calvinnnleo/home-credit-risk-prediction.git
cd home-credit-risk-prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 💻 Usage

### Training the Model
```python
from src.model_training import train_model

# Load processed data
X_train, y_train = load_data('data/final/train.csv')

# Train with cost-sensitive learning
model = train_model(
    X_train, 
    y_train, 
    scale_pos_weight=11.39
)
```

### Making Predictions
```python
from src.model_training import predict

# Load test data
X_test = load_data('data/final/test.csv')

# Predict with optimized threshold
predictions = predict(model, X_test, threshold=0.6760)
```

### Model Interpretation
```python
from src.model_interpretation import explain_prediction

# SHAP explanation
explain_prediction(model, X_test[0], method='shap')

# LIME explanation
explain_prediction(model, X_test[0], method='lime')
```

## 📊 Key Features

### Cost-Sensitive Learning
Addresses class imbalance by assigning higher weight to minority class:
```python
scale_pos_weight = (n_negative / n_positive) = 11.39
```

### Threshold Optimization
Instead of default 0.5, optimized threshold to 0.6760 for better business value:
- Reduces false positives
- Maintains acceptable recall
- Maximizes F1-Score

### Data Leakage Prevention
- Stratified Train-Test Split before any feature engineering
- Feature selection performed only on training data
- No information leak from test set during model development

## 🎓 Lessons Learned

1. **Class imbalance requires domain-specific solutions** - Generic SMOTE may not always be optimal; cost-sensitive learning proved more effective
2. **Feature selection is crucial** - Reducing from 463 to 20 features improved interpretability without sacrificing performance
3. **Threshold tuning matters** - Default 0.5 threshold is rarely optimal for imbalanced datasets
4. **Explainability is non-negotiable in finance** - SHAP/LIME integration ensures regulatory compliance

## 📝 Future Improvements

- [ ] Experiment with ensemble methods (Stacking, Blending)
- [ ] Incorporate deep learning for automatic feature extraction
- [ ] Deploy model as REST API using FastAPI
- [ ] Build real-time monitoring dashboard
- [ ] A/B testing framework for production deployment

## 🙏 Acknowledgments

- Home Credit for providing the dataset
- Kaggle community for inspiration and discussions
- Open-source contributors of LightGBM, SHAP, and LIME

---

⭐ If you find this project useful, please consider giving it a star!
