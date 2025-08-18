import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import os

sns.set(style="whitegrid", palette="muted")
os.makedirs("results", exist_ok=True)
df = pd.read_csv("data/Mall_Customers.csv")

print("First 5 rows of dataset:")
print(df.head())
print("\nDataset Info:")
print(df.info())
print("\nMissing Values:")
print(df.isnull().sum())

X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=X, x='Annual Income (k$)', y='Spending Score (1-100)',
    s=80, color='purple', edgecolor='black'
)
plt.title('Customer Data Distribution', fontsize=16, fontweight='bold')
plt.xlabel('Annual Income (k$)', fontsize=12)
plt.ylabel('Spending Score (1-100)', fontsize=12)
plt.savefig("results/Customer_data_distribution.png")
plt.show()

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
sns.lineplot(x=range(1, 11), y=wcss, marker='o', color='blue')
plt.title('Elbow Method - Optimal K', fontsize=16, fontweight='bold')
plt.xlabel('Number of Clusters', fontsize=12)
plt.ylabel('WCSS (Within Cluster Sum of Squares)', fontsize=12)
plt.xticks(range(1, 11))
plt.savefig("results/Elbow_Method_Optimal_K.png")
plt.show()

kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X)
df['Cluster'] = y_kmeans

plt.figure(figsize=(10, 7))
palette = sns.color_palette("husl", 5)
for i in range(5):
    sns.scatterplot(
        x=X.values[y_kmeans == i, 0],
        y=X.values[y_kmeans == i, 1],
        s=120, color=palette[i], label=f'Cluster {i+1}', edgecolor='black'
    )

sns.scatterplot(
    x=kmeans.cluster_centers_[:, 0], y=kmeans.cluster_centers_[:, 1],
    s=400, color='yellow', edgecolor='black', marker='*', label='Centroids'
)

plt.title('Customer Segments - KMeans Clustering', fontsize=18, fontweight='bold')
plt.xlabel('Annual Income (k$)', fontsize=14)
plt.ylabel('Spending Score (1-100)', fontsize=14)
plt.legend(loc='best', fontsize=10)
plt.savefig("results/Customers_Segments_KMeans.png")
plt.show()

print("\nCluster Summary:")
summary = df.groupby('Cluster')[['Annual Income (k$)', 'Spending Score (1-100)']].mean()
print(summary)

df.to_csv("results/Clustered_Customers.csv", index=False)
print("\nClustered data saved as 'results/Clustered_Customers.csv'")
