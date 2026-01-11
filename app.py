import streamlit as st
import numpy as np
import pandas as pd
import joblib

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

# --------------------------------------------
# APP TITLE
# --------------------------------------------
st.title("🍷 Wine Dataset – Multiclass Classification with PCA")

st.markdown("""
This Streamlit application demonstrates **all steps** of the ML pipeline:
- Data Loading & Exploration  
- Preprocessing & Scaling  
- PCA Analysis  
- Model Training & Evaluation  
- Final Prediction  
""")

# ============================================
# a) DATA LOADING & EXPLORATION
# ============================================
st.header("a) Data Loading, Cleaning & Exploration")

wine = load_wine()
X = wine.data
y = wine.target

st.write("**Shape of X:**", X.shape)
st.write("**Shape of y:**", y.shape)

df = pd.DataFrame(X, columns=wine.feature_names)
df["target"] = y

st.subheader("First 5 Rows of Dataset")
st.dataframe(df.head())

st.subheader("Summary Statistics")
st.dataframe(df.describe())

st.subheader("Class Distribution")
class_counts = df["target"].value_counts()
st.write(class_counts)

if class_counts.max() - class_counts.min() < 10:
    st.success("Dataset is Balanced")
else:
    st.warning("Dataset is Imbalanced")

# ============================================
# b) PREPROCESSING & STRATIFIED SPLIT
# ============================================
st.header("b) Preprocessing, Scaling & Stratified Split")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

st.write("Training set shape:", X_train.shape)
st.write("Testing set shape:", X_test.shape)

# ============================================
# c) PCA ANALYSIS
# ============================================
st.header("c) PCA Analysis")

pca_full = PCA()
pca_full.fit(X_train)

cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)

components_95 = np.argmax(cumulative_variance >= 0.95) + 1
components_99 = np.argmax(cumulative_variance >= 0.99) + 1

st.write("Components needed for **95% variance:**", components_95)
st.write("Components needed for **99% variance:**", components_99)

# Apply PCA (95%)
pca = PCA(n_components=0.95, random_state=42)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

st.subheader("Explained Variance Ratio (95% PCA)")
st.write(pca.explained_variance_ratio_)

# ============================================
# d) MODEL TRAINING & EVALUATION
# ============================================
st.header("d) Model Training, Evaluation & Comparison")

models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel="rbf", probability=True, random_state=42)
}

results = {}

for name, model in models.items():
    model.fit(X_train_pca, y_train)
    y_pred = model.predict(X_test_pca)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    results[name] = acc

    st.subheader(f"{name}")
    st.write("Accuracy:", acc)
    st.write("Confusion Matrix:")
    st.dataframe(cm)

# Best model
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]

st.success(f"✅ Best Performing Model: **{best_model_name}**")
st.write("Justification: Highest test accuracy among all classifiers.")

# ============================================
# e) MODEL DEPLOYMENT & PREDICTION
# ============================================
st.header("e) Model Deployment – Wine Class Prediction")

st.write("Enter wine chemical properties to predict the wine class:")

feature_names = wine.feature_names
user_input = []

for feature in feature_names:
    value = st.number_input(feature, value=0.0)
    user_input.append(value)

if st.button("Predict Wine Class"):
    input_array = np.array(user_input).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    input_pca = pca.transform(input_scaled)

    prediction = best_model.predict(input_pca)[0]

    st.success(f"🍇 Predicted Wine Class: **{prediction}**")
