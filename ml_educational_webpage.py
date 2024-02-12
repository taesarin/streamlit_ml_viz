import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn import tree
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np

# Function to load data
def load_data(file):
    data = pd.read_csv(file)
    st.write("Input Data Preview (Top 5 Rows):")
    st.dataframe(data.head())  # Displaying the top 5 rows of the dataframe
    return data

# Select target variable
def select_target_variable(data):
    target = st.selectbox('Select the target variable', data.columns)
    return target

# Linear Regression Model Training and Plotting
def linear_regression(data, target):
    feature = st.selectbox('Select the feature for regression', [col for col in data.columns if col != target], key='lr_feature')
    X = data[[feature]].values
    y = data[target].values
    model = LinearRegression()
    model.fit(X, y)
    predictions = model.predict(X)
    
    # Sorting values for plotting
    sorted_indices = np.argsort(X.squeeze())
    sorted_X = X[sorted_indices].squeeze()
    sorted_predictions = predictions[sorted_indices]
    
    fig = px.scatter(x=sorted_X, y=y[sorted_indices], labels={'x': feature, 'y': target})
    fig.add_scatter(x=sorted_X, y=sorted_predictions, mode='lines', name='Trend Line')
    return fig

# Decision Tree Model Training and Plotting
def decision_tree(data, target):
    features = [col for col in data.columns if col != target]
    X = data[features].values
    y = data[target].values
    model = DecisionTreeClassifier()
    model.fit(X, y)
    
    # Correcting class_names format
    class_names = np.unique(y).astype(str)  # Convert class names to string
    
    fig, ax = plt.subplots(figsize=(12, 8))
    tree.plot_tree(model, filled=True, ax=ax, feature_names=features, class_names=class_names)
    return fig

# K-Means Clustering and Plotting
def k_means_clustering(data, n_clusters):
    X = data.select_dtypes(include=[np.number]).values  # Ensure clustering is done on numeric columns
    model = KMeans(n_clusters=n_clusters)
    labels = model.fit_predict(X)
    
    # Applying a distinct color palette
    fig = px.scatter(x=X[:, 0], y=X[:, 1], color=labels, 
                     color_continuous_scale=px.colors.qualitative.Bold, labels={'x': 'Feature 1', 'y': 'Feature 2'})
    fig.update_traces(marker=dict(size=12, line=dict(width=2, color='DarkSlateGrey')))
    return fig

# Streamlit UI
st.title('Basic Machine Learning Visualization')

# Navigation
page = st.sidebar.selectbox('Choose a model', ['Home', 'Linear Regression', 'Decision Tree', 'K-Means Clustering'])

if page == 'Home':
    st.header('Welcome to the ML Educational Webpage')
    st.write('Select a model from the sidebar to get started.')

elif page == 'Linear Regression':
    st.header('Linear Regression')
    file = st.file_uploader('Upload your CSV data file', type=['csv'])
    if file is not None:
        data = load_data(file)
        target = select_target_variable(data)
        if st.button('Run Model'):
            fig = linear_regression(data, target)
            st.plotly_chart(fig)

elif page == 'Decision Tree':
    st.header('Decision Tree')
    file = st.file_uploader('Upload your CSV data file', type=['csv'])
    if file is not None:
        data = load_data(file)
        target = select_target_variable(data)
        if st.button('Run Model'):
            fig = decision_tree(data, target)
            st.pyplot(fig)

elif page == 'K-Means Clustering':
    st.header('K-Means Clustering')
    file = st.file_uploader('Upload your CSV data file', type=['csv'])
    if file is not None:
        data = load_data(file)
        n_clusters = st.number_input('Enter the number of clusters', min_value=2, max_value=10, value=3)
        if st.button('Run Model'):
            fig = k_means_clustering(data, n_clusters)
            st.plotly_chart(fig)
