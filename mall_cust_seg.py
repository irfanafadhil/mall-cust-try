import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import streamlit as st

# Import dataset
df = pd.read_csv('D:\Streamlit\datasets\Mall_Customers.csv')

# Rename important columns
df.rename(index=str, columns={
    'Annual Income (k$)' : 'Income',
    'Spending Score (1-100)' : 'Score'
}, inplace=True)

# Select datasets
X = df.drop(['CustomerID','Gender'], axis=1)

st.header("Used Datasets")
st.write(X)


# Determine the optimal number of cluster
# Elbow method
cluster = []
for i in range(1,11):
    km = KMeans(n_clusters=i).fit(X)
    cluster.append(km.inertia_)

fig, ax = plt.subplots(figsize=(12,8))
sns.lineplot(x=list(range(1,11)), y=cluster, ax=ax)
ax.set_title('Elbow Point')
ax.set_xlabel('Cluster')
ax.set_ylabel('Inertia')

# Elbow arrow
ax.annotate('Possible elbow point', xy=(3,140000), xytext=(2,100000), xycoords='data',
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='blue', lw=2))
ax.annotate('Possible elbow point', xy=(5,75000), xytext=(4,50000), xycoords='data',
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='blue', lw=2))

st.set_option('deprecation.showPyplotGlobalUse', False)
elbow_plot = st.pyplot()

st.sidebar.subheader("k numbers")
cluster = st.sidebar.slider("Select cluster number :", 2,10,3,1)

# Perform K-Means
def k_means(n_clust):
    kmean = KMeans(n_clusters=n_clust).fit(X)
    X['Label'] = kmean.labels_

    # Plot result
    plt.figure(figsize=(10,8))
    sns.scatterplot(x=X['Income'], y=X['Score'], hue=X['Label'], markers=True, size=X['Label'], palette=sns.color_palette('hls', n_clust))

    for label in X['Label']:
        plt.annotate(label,
                    (X[X['Label']==label]['Income'].mean(),
                    X[X['Label']==label]['Score'].mean()),
                    horizontalalignment = 'center',
                    verticalalignment = 'center',
                    size = 20, weight='bold',
                    color= 'black')
    
    st.header("Clustering Result")
    st.pyplot()
    st.write(X)

k_means(cluster)