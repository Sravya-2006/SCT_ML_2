 import pandas as pd
 import matplotlib.pyplot as plt
 from sklearn.cluster import KMeans
 df = pd.read_csv("Mall_Customers.csv")
 X = df[["Annual Income (k$)", "Spending Score (1-100)"]]
 wcss = []  
 for k in range(1, 11):
     kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
     kmeans.fit(X)
     wcss.append(kmeans.inertia_)
 plt.figure(figsize=(8, 5))
 plt.plot(range(1, 11), wcss, marker='o')
 plt.title('The Elbow Method')
 plt.xlabel('Number of Clusters')
 plt.ylabel('WCSS')
 plt.grid(True)
 plt.show()
 k_optimal = 5
 kmeans = KMeans(n_clusters=k_optimal, init='k-means++', random_state=42)
 y_kmeans = kmeans.fit_predict(X)
 plt.figure(figsize=(10, 6))
 colors = ['turquoise', 'orange', 'lightgreen', 'orchid', 'mediumpurple']
 labels = [
     "Type 1: High Income, Low Spending",
     "Type 2: Low Income, Low Spending",
     "Type 3: Low Income, High Spending",
     "Type 4: Average Income, Average Spending",
     "Type 5: High Income, High Spending"
 ]
 for i in range(k_optimal):
     plt.scatter(
         X[y_kmeans == i]["Annual Income (k$)"],
         X[y_kmeans == i]["Spending Score (1-100)"],
        s=60,
         c=colors[i],
         label=labels[i]
     )
 plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
     s=200,
     c='black',
     marker='X',
     label='Centroids'
 )

plt.title('Customer Segmentation Based on Income and Spending')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


