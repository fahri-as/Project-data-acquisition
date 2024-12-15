import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set up the Streamlit app
st.title("Crime Data Analysis App")

# Section 1: Upload Dataset
st.header("1. Upload Dataset")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file:
    # Load the dataset
    dataset = pd.read_csv(uploaded_file)

    st.write("Dataset Preview (First 5 Rows):")
    st.write(dataset.head())

    # Button to display all rows of the uploaded dataset
    if st.button("Show All Rows of Uploaded Dataset"):
        st.write(dataset)
else:
    st.warning("Please upload a dataset to proceed.")

if uploaded_file:
    # Section 2: Preprocess Data
    st.header("2. Preprocess Data")

    # Pastikan kolom-kolom yang diperlukan ada
    required_cols = ['crime_id', 'crime_type', 'crime_date', 'location', 'suspect_name',
                     'victim_name', 'witness_name', 'crime_description', 'arrest_date', 'sentence_length']
    for col in required_cols:
        if col not in dataset.columns:
            st.warning(f"Column '{col}' not found in the dataset.")
            st.stop()

    # Drop rows dengan missing values yang penting untuk clustering
    preprocessed_data = dataset.dropna(subset=['crime_type', 'crime_date', 'location', 'arrest_date', 'sentence_length']).copy()

    # Convert date columns to datetime
    preprocessed_data['crime_date'] = pd.to_datetime(preprocessed_data['crime_date'], errors='coerce')
    preprocessed_data['arrest_date'] = pd.to_datetime(preprocessed_data['arrest_date'], errors='coerce')

    # Drop rows jika ada yg tidak bisa dikonversi ke datetime
    preprocessed_data = preprocessed_data.dropna(subset=['crime_date', 'arrest_date'])

    # Create a new feature: time_to_arrest (in days)
    preprocessed_data['time_to_arrest'] = (preprocessed_data['arrest_date'] - preprocessed_data['crime_date']).dt.days

    # Pilih fitur yang relevan untuk clustering
    # - crime_type (categorical)
    # - location (categorical)
    # - sentence_length (numeric)
    # - time_to_arrest (numeric)
    features = ['crime_type', 'location', 'sentence_length', 'time_to_arrest']

    # Isi kolom-kolom yang tidak digunakan dalam clustering (non-core features)
    # Misalnya: suspect_name, victim_name, witness_name, crime_description
    # Jika kategorikal/teks: isi dengan "Unknown"
    # Jika numerik: isi dengan median
    all_cols = preprocessed_data.columns.tolist()
    non_core_cols = [col for col in all_cols if col not in features and col not in ['crime_date', 'arrest_date', 'crime_id']]

    for col in non_core_cols:
        if preprocessed_data[col].dtype == 'object':
            # Isi dengan "Unknown" untuk teks
            preprocessed_data[col] = preprocessed_data[col].fillna("Unknown")
        else:
            # Isi dengan median untuk numerik
            preprocessed_data[col] = preprocessed_data[col].fillna(preprocessed_data[col].median())

    # Encode categorical features
    cat_cols = ['crime_type', 'location']
    for ccol in cat_cols:
        le = LabelEncoder()
        preprocessed_data[ccol] = le.fit_transform(preprocessed_data[ccol])

    # Pastikan tidak ada missing value di fitur yang digunakan
    preprocessed_data = preprocessed_data.dropna(subset=features)

    # Scale features
    scaler = StandardScaler()
    preprocessed_data[features] = scaler.fit_transform(preprocessed_data[features])

    st.write("Preprocessed Data (First 5 Rows):")
    st.write(preprocessed_data.head())

    if st.button("Show All Rows of Preprocessed Dataset"):
        st.write(preprocessed_data)

    # Section 3: Clustering
    st.header("3. Clustering with K-Means")

    # Apply KMeans
    kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
    preprocessed_data['cluster'] = kmeans.fit_predict(preprocessed_data[features])

    st.write("Cluster Analysis Results (First 5 Rows):")
    st.write(preprocessed_data[['crime_id', 'crime_type', 'location', 'sentence_length', 'time_to_arrest', 'cluster']].head())

    pca = PCA(n_components=2, random_state=42)
    pca_result = pca.fit_transform(preprocessed_data[features])
    preprocessed_data['pca_x'] = pca_result[:,0]
    preprocessed_data['pca_y'] = pca_result[:,1]

    fig, ax = plt.subplots(figsize=(8,6))
    sns.scatterplot(data=preprocessed_data, x='pca_x', y='pca_y', hue='cluster', palette='viridis', ax=ax)
    ax.set_title("Clusters Visualization in PCA Space")
    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    st.pyplot(fig)

    st.write("Dengan PCA, kita bisa melihat apakah ada pemisahan klaster yang lebih jelas dalam ruang 2 dimensi yang mewakili variasi terbesar dalam data.")

    # Section 4: Visualization Data
    st.header("4. Visualization Data")

    # Visualization: Distribution of Crime Types by Cluster
    st.subheader("Crime Type Distribution per Cluster")
    if 'crime_type' in preprocessed_data.columns and 'cluster' in preprocessed_data.columns:
        cluster_type_counts = preprocessed_data.groupby(['cluster', 'crime_type']).size().reset_index(name='count')
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=cluster_type_counts, x='crime_type', y='count', hue='cluster', ax=ax, palette='viridis')
        ax.set_title("Crime Type Count per Cluster")
        ax.set_xlabel("Crime Type (Encoded)")
        ax.set_ylabel("Count")
        st.pyplot(fig)
    
    # Visualization: Boxplot Sentence Length by Cluster
    st.subheader("Sentence Length Distribution by Cluster")
    if 'sentence_length' in preprocessed_data.columns and 'cluster' in preprocessed_data.columns:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.boxplot(data=preprocessed_data, x='cluster', y='sentence_length', ax=ax)
        ax.set_title("Sentence Length Distribution by Cluster")
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Sentence Length (Scaled)")
        st.pyplot(fig)

    # Visualization: Boxplot Time to Arrest by Cluster
    st.subheader("Time to Arrest Distribution by Cluster")
    if 'time_to_arrest' in preprocessed_data.columns and 'cluster' in preprocessed_data.columns:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.boxplot(data=preprocessed_data, x='cluster', y='time_to_arrest', ax=ax)
        ax.set_title("Time to Arrest Distribution by Cluster")
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Time to Arrest (Days, Scaled)")
        st.pyplot(fig)
