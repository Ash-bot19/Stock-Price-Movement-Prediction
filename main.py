import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import ConfusionMatrixDisplay, roc_auc_score
from sklearn.utils import resample

# ===== Step 1: Download & preprocess =====
df = yf.download("AAPL", start="2018-01-01", end="2025-01-01")
df.columns = df.columns.get_level_values(0)

df['day'] = df.index.day
df['month'] = df.index.month
df['year'] = df.index.year
df['is_quarter_end'] = np.where(df['month']%3==0, 1, 0)

# Features
df['open-close'] = df['Open'] - df['Close']
df['low-high'] = df['Low'] - df['High']
df['target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)

# ===== Step 2: Check imbalance =====
plt.pie(df['target'].value_counts(), labels=df['target'].value_counts().index, autopct='%1.1f%%')
plt.title("Class Distribution Before Balancing")
plt.show()

# Drop NaN from shift
df.dropna(inplace=True)

# ===== Step 3: Balance dataset =====
df_majority = df[df['target'] == 1]
df_minority = df[df['target'] == 0]

df_minority_upsampled = resample(df_minority, 
                                 replace=True, 
                                 n_samples=len(df_majority), 
                                 random_state=42)

df_balanced = pd.concat([df_majority, df_minority_upsampled])

# Shuffle the balanced dataset
df_balanced = df_balanced.sample(frac=1, random_state=42)

# ===== Step 4: Features & scaling =====
features = df_balanced[['open-close', 'low-high', 'is_quarter_end']]
target = df_balanced['target']

scaler = StandardScaler()
features = scaler.fit_transform(features)

X_train, X_valid, Y_train, Y_valid = train_test_split(
    features, target, test_size=0.1, random_state=2022
)

# ===== Step 5: Train models =====
models = [
    LogisticRegression(),
    SVC(kernel='poly', probability=True),
    XGBClassifier(use_label_encoder=False, eval_metric='logloss')
]

for model in models:
    model.fit(X_train, Y_train)
    print(f'{model.__class__.__name__}:')
    print('Training ROC-AUC:', roc_auc_score(Y_train, model.predict_proba(X_train)[:, 1]))
    print('Validation ROC-AUC:', roc_auc_score(Y_valid, model.predict_proba(X_valid)[:, 1]))
    print()

# ===== Step 6: Confusion Matrix for LogisticRegression =====
ConfusionMatrixDisplay.from_estimator(models[0], X_valid, Y_valid)
plt.title("Confusion Matrix - Logistic Regression (Balanced Data)")
plt.show()
