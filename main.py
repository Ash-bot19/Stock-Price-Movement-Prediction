import yfinance as yf
import pandas as pd
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from transformers import pipeline

# 1. Download stock data
df = yf.download("AAPL", start="2020-01-01", end="2024-12-31")

# 2. Target column: 1 if price goes up next day, else 0
df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

# 3. Technical Indicators
df['SMA_10'] = df['Close'].rolling(window=10).mean()

def compute_rsi(data, window=14):
    delta = data.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

df['RSI_14'] = compute_rsi(df['Close'])

# MACD
ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
df['MACD'] = ema_12 - ema_26

# Calculate EMA 20
df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()

# Correct way to get rolling std as a Series
df['Rolling_STD_20'] = df['Close'].rolling(window=20).std()

# Bollinger Bands
df['BB_upper'] = df['EMA_20'] + 2 * df['Rolling_STD_20']
df['BB_lower'] = df['EMA_20'] - 2 * df['Rolling_STD_20']


# Drop rows with NaNs created by rolling/EMA
df.dropna(inplace=True)

# 4. Feature selection
features = ['SMA_10', 'RSI_14', 'MACD', 'Volume', 'BB_upper', 'BB_lower']
X = df[features]
y = df['Target']

# 5. Time-based split
split_index = int(0.8 * len(df))
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

# 6. Train RandomForest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

print("🔍 Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# 7. Train XGBoost
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

print("🚀 XGBoost Accuracy:", accuracy_score(y_test, y_pred_xgb))
print(classification_report(y_test, y_pred_xgb))

# 8. Hyperparameter tuning for XGBoost
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1]
}

grid = GridSearchCV(XGBClassifier(eval_metric='logloss'), param_grid, cv=3, scoring='accuracy', verbose=1)
grid.fit(X_train, y_train)

print("✅ Best XGBoost Params:", grid.best_params_)
print("📈 Best CV Score:", grid.best_score_)

# 9. Sentiment Analysis
sentiment_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert")
headline = "Apple beats quarterly earnings expectations"
sentiment = sentiment_pipeline(headline)
print("📰 Headline Sentiment:", sentiment)
