# 📈 Stock Price Movement Prediction Project --P1  

## **Project Overview**  

**Project Title**: Stock Price Movement Prediction (AAPL)  
**Dataset**: Yahoo Finance – Apple Inc. (`AAPL`) historical OHLCV data  

This project demonstrates the implementation of a machine learning pipeline for predicting **next-day stock price movement**. It includes data collection, exploratory data analysis (EDA), feature engineering, model training, and evaluation using various ML models.  

![Stock_project](https://github.com/yourusername/Stock-Price-Movement-Prediction/blob/main/stock.jpg)  

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
