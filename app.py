import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

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

# Proceed if dataset is uploaded
if uploaded_file:
    # Section 2: Preprocess Data
    st.header("2. Preprocess Data")
    st.write("Removing rows with missing values.")
    preprocessed_data = dataset.dropna()
    st.write("Preprocessed Data Preview (First 5 Rows):")
    st.write(preprocessed_data.head())

    # Button to display all rows of the preprocessed dataset
    if st.button("Show All Rows of Preprocessed Dataset"):
        st.write(preprocessed_data)

    # Section 3: Analyze Data with Clustering
    st.header("3. Analyze Data")
    st.write("Clustering data based on 'crime_type'.")

    if 'crime_type' in preprocessed_data.columns:
        # Encode categorical 'crime_type'
        le = LabelEncoder()
        preprocessed_data['crime_type_encoded'] = le.fit_transform(preprocessed_data['crime_type'])

        # Apply KMeans clustering
        kmeans = KMeans(n_clusters=3, random_state=42)
        preprocessed_data['cluster'] = kmeans.fit_predict(preprocessed_data[['crime_type_encoded']])

        st.write("Cluster Analysis Results (First 5 Rows):")
        st.write(preprocessed_data[['crime_type', 'cluster']].head())

        # Button to display all rows of the clustering result
        if st.button("Show All Rows of Clustering Result"):
            st.write(preprocessed_data[['crime_type', 'cluster']])
        
        # Display cluster distribution
        st.bar_chart(preprocessed_data['cluster'].value_counts())
    else:
        st.warning("The dataset does not have a 'crime_type' column.")

    # Section 4: Visualize Data
    st.header("4. Visualize Data")

    # Visualization 1: Crime Type Distribution
    st.subheader("Crime Type Distribution")
    if 'crime_type' in preprocessed_data.columns:
        crime_type_counts = preprocessed_data['crime_type'].value_counts()
        fig, ax = plt.subplots()
        crime_type_counts.plot(kind='bar', ax=ax, color="skyblue", edgecolor="black")
        ax.set_title("Crime Type Distribution")
        ax.set_xlabel("Crime Type")
        ax.set_ylabel("Count")
        st.pyplot(fig)

    # Visualization 2: Crime Distribution by Location
    st.subheader("Crime Distribution by Location")
    if 'location' in preprocessed_data.columns:
        location_counts = preprocessed_data['location'].value_counts().head(10)
        fig, ax = plt.subplots()
        location_counts.plot(kind='bar', ax=ax, color="orange", edgecolor="black")
        ax.set_title("Top 10 Locations with Most Crimes")
        ax.set_xlabel("Location")
        ax.set_ylabel("Count")
        st.pyplot(fig)

    # Visualization 3: Crime Trend Over Time
    st.subheader("Crime Trend Over Time")
    if 'crime_date' in preprocessed_data.columns:
        # Convert 'crime_date' to datetime
        preprocessed_data['crime_date'] = pd.to_datetime(preprocessed_data['crime_date'], errors='coerce')
        crime_trend = preprocessed_data.groupby(preprocessed_data['crime_date'].dt.to_period("M")).size()
        fig, ax = plt.subplots()
        crime_trend.plot(kind='line', ax=ax, color="green")
        ax.set_title("Crime Trend Over Time")
        ax.set_xlabel("Date")
        ax.set_ylabel("Number of Crimes")
        st.pyplot(fig)

    # Visualization 4: Sentence Length by Crime Type
    st.subheader("Sentence Length by Crime Type")
    if 'sentence_length' in preprocessed_data.columns and 'crime_type' in preprocessed_data.columns:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.boxplot(data=preprocessed_data, x='crime_type', y='sentence_length', ax=ax)
        ax.set_title("Sentence Length Distribution by Crime Type")
        ax.set_xlabel("Crime Type")
        ax.set_ylabel("Sentence Length (Years)")
        st.pyplot(fig)
