import pandas as pd
import streamlit as st
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Hide warnings
warnings.filterwarnings('ignore')

# Streamlit App Title
st.title("Prediksi Penyakit Jantung dengan Decision Tree")

# File Upload (Using Streamlit File Uploader)
uploaded_file = st.file_uploader("Pilih file CSV", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        # Validate the existence of 'target' column
        if 'target' not in df.columns:
            st.error("Kolom 'target' tidak ditemukan dalam dataset. Pastikan file CSV memiliki kolom 'target'.")
        else:
            # Sidebar Navigation
            st.sidebar.header("Navigasi")
            options = st.sidebar.radio(
                "Pilih Halaman:",
                ["Dataset", "Model", "Confusion Matrix", "Visualisasi Pohon Keputusan"]
            )

            # Dataset Overview
            if options == "Dataset":
                st.subheader("Dataset")
                st.write("### Informasi Data")
                st.write(df.head())
                st.write("### Deskripsi Data")
                st.write(df.describe())
                st.write("### Info Data")
                st.write(df.info())
                st.write("### Nilai yang Hilang")
                st.write(df.isnull().sum())

            # Model Overview
            elif options == "Model":
                st.subheader("Model Decision Tree")

                # Data Preparation
                X = df.drop(columns=['target'])
                Y = df['target']

                # Normalization
                scaler = MinMaxScaler()
                X_scaled = scaler.fit_transform(X)

                # Train-Test Split
                X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=0)

                # Model Initialization and Training
                dt = DecisionTreeClassifier(criterion='entropy', random_state=0, max_depth=6)
                dt.fit(X_train, y_train)

                # Predictions and Evaluation
                dt_predicted = dt.predict(X_test)
                dt_acc_score = accuracy_score(y_test, dt_predicted)

                st.write(f"### Akurasi Model Decision Tree: {dt_acc_score * 100:.2f} %")
                st.write("### Laporan Klasifikasi")
                st.text(classification_report(y
