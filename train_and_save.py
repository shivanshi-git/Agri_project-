import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
# import xgboost as xgb
import pickle
import os
import warnings
# --- DATA LOADING AND PREPARATION ---
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


warnings.filterwarnings('ignore')

# --- DATA LOADING AND EXPLORATION ---
df = pd.read_csv(r'C:\Users\shiva\OneDrive\Desktop\TRY PART 2\IMAGE-NUMERIC-MODEL\image\Crop_recommendation.csv') # Assuming the CSV is in the same directory

# Separate features and target label
features = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
target = df['label']

# --- Perform Label Encoding on the target variable ---
le = LabelEncoder()
target = le.fit_transform(target)

# Splitting into train and test data
Xtrain, Xtest, Ytrain, Ytest = train_test_split(features, target, test_size=0.2, random_state=2)

# --- Continue with the rest of your model training code ---
# ... (the rest of your script follows here)

print("Dataset Head:")
print(df.head())
print("\nDataset Info:")
print(df.info())
print("\nValue Counts for 'label':")
print(df['label'].value_counts())

# Correlation heatmap
numeric_df = df.select_dtypes(include=["number"])
plt.figure(figsize=(10, 8))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix of Features")
plt.show()

# --- MODEL TRAINING AND EVALUATION ---
features = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
target = df['label']

Xtrain, Xtest, Ytrain, Ytest = train_test_split(features, target, test_size=0.2, random_state=2)

acc = []
model = []

# --- 1. Decision Tree Classifier ---
print("\n--- Training Decision Tree ---")
DecisionTree = DecisionTreeClassifier(criterion="entropy", random_state=2, max_depth=5)
DecisionTree.fit(Xtrain, Ytrain)
predicted_values = DecisionTree.predict(Xtest)
x = accuracy_score(Ytest, predicted_values)
acc.append(x)
model.append('Decision Tree')
print(f"Decision Tree's Accuracy: {x*100:.2f}%")
print(classification_report(Ytest, predicted_values))
score_dt = cross_val_score(DecisionTree, features, target, cv=5)
print(f"Decision Tree Cross-Validation Scores: {score_dt}")

# --- 2. Gaussian Naive Bayes ---
print("\n--- Training Naive Bayes ---")
NaiveBayes = GaussianNB()
NaiveBayes.fit(Xtrain, Ytrain)
predicted_values = NaiveBayes.predict(Xtest)
x = accuracy_score(Ytest, predicted_values)
acc.append(x)
model.append('Naive Bayes')
print(f"Naive Bayes's Accuracy: {x*100:.2f}%")
print(classification_report(Ytest, predicted_values))
score_nb = cross_val_score(NaiveBayes, features, target, cv=5)
print(f"Naive Bayes Cross-Validation Scores: {score_nb}")

# --- 3. Support Vector Machine (SVM) ---
print("\n--- Training SVM ---")
norm = MinMaxScaler().fit(Xtrain)
X_train_norm = norm.transform(Xtrain)
X_test_norm = norm.transform(Xtest)
SVM = SVC(kernel='poly', degree=3, C=1)
SVM.fit(X_train_norm, Ytrain)
predicted_values = SVM.predict(X_test_norm)
x = accuracy_score(Ytest, predicted_values)
acc.append(x)
model.append('SVM')
print(f"SVM's Accuracy: {x*100:.2f}%")
print(classification_report(Ytest, predicted_values))
score_svm = cross_val_score(SVM, features, target, cv=5)
print(f"SVM Cross-Validation Scores: {score_svm}")

# --- 4. Logistic Regression ---
print("\n--- Training Logistic Regression ---")
LogReg = LogisticRegression(random_state=2)
LogReg.fit(Xtrain, Ytrain)
predicted_values = LogReg.predict(Xtest)
x = accuracy_score(Ytest, predicted_values)
acc.append(x)
model.append('Logistic Regression')
print(f"Logistic Regression's Accuracy: {x*100:.2f}%")
print(classification_report(Ytest, predicted_values))
score_lr = cross_val_score(LogReg, features, target, cv=5)
print(f"Logistic Regression Cross-Validation Scores: {score_lr}")

# --- 5. Random Forest ---
print("\n--- Training Random Forest ---")
RF = RandomForestClassifier(n_estimators=20, random_state=0)
RF.fit(Xtrain, Ytrain)
predicted_values = RF.predict(Xtest)
x = accuracy_score(Ytest, predicted_values)
acc.append(x)
model.append('RF')
print(f"Random Forest's Accuracy: {x*100:.2f}%")
print(classification_report(Ytest, predicted_values))
score_rf = cross_val_score(RF, features, target, cv=5)
print(f"Random Forest Cross-Validation Scores: {score_rf}")

# # --- 6. XGBoost ---
# print("\n--- Training XGBoost ---")
# XB = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
# XB.fit(Xtrain, Ytrain)
# predicted_values = XB.predict(Xtest)
# x = accuracy_score(Ytest, predicted_values)
# acc.append(x)
# model.append('XGBoost')
# print(f"XGBoost's Accuracy: {x*100:.2f}%")
# print(classification_report(Ytest, predicted_values))
# score_xb = cross_val_score(XB, features, target, cv=5)
# print(f"XGBoost Cross-Validation Scores: {score_xb}")

# --- ACCURACY COMPARISON ---
print("\n--- Accuracy Comparison ---")
accuracy_models = dict(zip(model, acc))
for k, v in accuracy_models.items():
    print(f"{k} --> {v:.4f}")

plt.figure(figsize=[10, 5], dpi=100)
plt.title('Accuracy Comparison')
plt.xlabel('Accuracy')
plt.ylabel('Algorithm')
sns.barplot(x=acc, y=model, palette='viridis')
plt.show()

# --- MODEL SAVING ---
# Create a directory for models if it doesn't exist
os.makedirs('models', exist_ok=True)

# Save the best-performing model (XGBoost)
# Or save all models as shown in the original notebook
with open('models/DecisionTree.pkl', 'wb') as file:
    pickle.dump(DecisionTree, file)
print("Saved DecisionTree.pkl")

with open('models/NBClassifier.pkl', 'wb') as file:
    pickle.dump(NaiveBayes, file)
print("Saved NBClassifier.pkl")

with open('models/SVMClassifier.pkl', 'wb') as file:
    pickle.dump(SVM, file)
print("Saved SVMClassifier.pkl")

with open('models/LogisticRegression.pkl', 'wb') as file:
    pickle.dump(LogReg, file)
print("Saved LogisticRegression.pkl")

with open('models/RandomForest.pkl', 'wb') as file:
    pickle.dump(RF, file)
print("Saved RandomForest.pkl")

# with open('models/XGBoost.pkl', 'wb') as file:
#     pickle.dump(XB, file)
# print("Saved XGBoost.pkl")

# --- FLASK API CODE ---
# This part of the code should be saved as a separate Python file, e.g., 'app.py'
# to be run independently as a web server.

"""
Save the following code as 'app.py'
"""
