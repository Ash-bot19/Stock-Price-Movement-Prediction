# Stock-Price-Movement-Prediction-using-News-Sentiment-and-Technical-Indicators

main.py file 
1️⃣ Checked class imbalance

I added this to visually confirm the imbalance in your target labels:

plt.pie(df['target'].value_counts(), 
        labels=df['target'].value_counts().index, 
        autopct='%1.1f%%')
plt.title("Class Distribution Before Balancing")
plt.show()


This showed that class 1 (up) was much more frequent than class 0 (down).

2️⃣ Dropped NaNs from shift(-1)

Your original target creation left the last row with NaN.

df.dropna(inplace=True)

3️⃣ Balanced the dataset with oversampling

Instead of training on the imbalanced data, I upsampled the minority class (class 0) to have the same number of samples as the majority class.

from sklearn.utils import resample

df_majority = df[df['target'] == 1]
df_minority = df[df['target'] == 0]

df_minority_upsampled = resample(
    df_minority, 
    replace=True, 
    n_samples=len(df_majority), 
    random_state=42
)

df_balanced = pd.concat([df_majority, df_minority_upsampled])
df_balanced = df_balanced.sample(frac=1, random_state=42)  # shuffle

4️⃣ Trained models on balanced data

I replaced your original features and target definitions so they came from the balanced dataset instead of the original:

features = df_balanced[['open-close', 'low-high', 'is_quarter_end']]
target = df_balanced['target']

5️⃣ Kept the rest of your training code the same

I still used:

Logistic Regression

SVC (poly)

ipynb file:
We’ll likely see a strong imbalance — most values are 1 because over the last years, Apple’s price has generally gone up more often than down (daily data).
If, for example, 85–90% of days are "up" (1), then predicting always 1 gives you:

High ROC-AUC (because probabilities align with imbalance)

Confusion matrix like yours (no true negatives)


XGBClassifier

But now all models trained on equal numbers of up and down days, so the confusion matrix is no longer biased toward 1.

✅ Key takeaway:
The main change was balancing the dataset before training, which forced the model to learn patterns for both classes instead of just guessing “up” every time.
