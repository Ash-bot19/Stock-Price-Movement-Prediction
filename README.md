# 📈 Stock Price Movement Prediction Project  
---
## **Project Overview**  

**Project Title**: Stock Price Movement Prediction (AAPL)  
**Dataset**: Yahoo Finance – Apple Inc. (`AAPL`) historical OHLCV data  

This project demonstrates the implementation of a machine learning pipeline for predicting **next-day stock price movement**. It includes data collection, exploratory data analysis (EDA), feature engineering, model training, and evaluation using various ML models.  


---

## **Objectives**  

1. **Collect and Preprocess Data**: Download stock price data (Open, High, Low, Close, Volume) using Yahoo Finance API.  
2. **Exploratory Data Analysis (EDA)**: Perform statistical analysis and visualize patterns in the data.  
3. **Feature Engineering**: Create additional features like Open-Close difference, High-Low spread, quarter-end flag, etc.  
4. **Target Creation**: Generate binary target variable for next-day price movement (Up = 1, Down = 0).  
5. **Model Development**: Train and evaluate classification models (Logistic Regression, SVM, XGBoost).  
6. **Evaluation Metrics**: Assess models using ROC-AUC scores, confusion matrix, and visualization.  

---

## **Project Structure**  

### **1. Data Collection & Preprocessing**  

- **Data Source**: `yfinance` API (2018–2025 AAPL stock data)  
- **Preprocessing Steps**:
  - Removed hierarchical column structure  
  - Checked for missing values  
  - Extracted day, month, year  
  - Created target variable for stock movement  

```python
import yfinance as yf
import numpy as np

df = yf.download("AAPL", start="2018-01-01", end="2025-01-01")
df['open-close'] = df['Open'] - df['Close']
df['low-high'] = df['Low'] - df['High']
df['target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
## **2. Exploratory Data Analysis (EDA)**
```

- **Line Plot**: Closing price trend over time  
- **Histograms & Boxplots**: Distribution of OHLCV features  
- **Correlation Heatmap**: Identified highly correlated features  
- **Target Balance**: Checked distribution of upward vs downward movements  

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(15,5))
plt.plot(df['Close'])
plt.title('Apple Close price.', fontsize=15)
plt.ylabel('Price in dollars.')
plt.show()
```
### **3. Feature Engineering**

- Added date-related features (day, month, year, quarter-end flag).
- Created financial indicators:
  - `open-close` (daily return strength)
  - `low-high` (intraday volatility)
- Constructed target variable:
  -`1` → Next day close > current close
  -`0` → Next day close ≤ current close

### **4. Model Development**
Trained three classification models:
```python
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier

models = [
    LogisticRegression(),
    SVC(kernel='poly', probability=True),
    XGBClassifier()
]
```
### **5. Evaluation & Results**
Metric Used: ROC-AUC Score

| Model                   | Train AUC | Validation AUC |
| ----------------------- | --------- | -------------- |
| Logistic Regression     | 0.51      | 0.48           |
| SVC (Polynomial Kernel) | 0.52      | 0.45           |
| XGBoost Classifier      | 0.96      | 0.55           |


- **Best Model:** ✅ XGBoost, though validation performance indicates **room for improvement.**
- **Confusion Matrix** was also plotted for classification insights.

### **6. Future Improvements**
- Add **technical indicators** (SMA, EMA, RSI, MACD, Bollinger Bands).
- Apply **hyperparameter tuning** (GridSearchCV, Bayesian optimization).
- Incorporate **sentiment analysis** from news & social media.
- Explore **deep learning models** (LSTM, GRU) for time-series prediction.
- Perform **feature importance analysis** for better interpretability.

### **Advanced Tasks**
- ✅ Feature scaling using `StandardScaler`
- ✅ Yearly grouped analysis of OHLC trends
- ✅ Pie chart for class balance check
- ✅ CTAS-like approach: Derived new features from raw OHLC data for modeling

### **Reports**
- **EDA Plots:** Close price trend, OHLC histograms, boxplots
- **Target Balance:** Distribution of up vs down movements
- **Model Comparison:** ROC-AUC results across ML models

### **Conclusion**

This project demonstrates how machine learning can be applied to financial market prediction using historical stock price data. Although initial results (AUC ≈ 0.55 with XGBoost) are modest, incorporating technical indicators, sentiment data, and advanced models will significantly enhance predictive power.
