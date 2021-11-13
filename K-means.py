import matplotlib.pyplot as plt
#from kneed import KneeLocator
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
#from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from random import randint

points = []
clabels = []
for i in range(20):
    xc  = randint(3,7)
    yc  = randint(3,5)

    points.append([xc,yc])
    clabels.append(0)


for i in range(30):
    xc  = randint(0,2)
    yc  = randint(6,9)

    points.append([xc,yc])
    clabels.append(1)


print("123...")

features, true_labels = make_blobs(
    n_samples=200,
    centers=3,
    cluster_std=2.75,
    random_state=42
)


#print(features)

scaler = StandardScaler()
#scaled_features = scaler.fit_transform(features)
scaled_features = scaler.fit_transform(points)

plt.plot([k[0] for k in scaled_features],[k[1] for k in scaled_features],'bo')
plt.show()

#plt.plot([k[0] for k in points],[k[1] for k in points],'bo')
#plt.show()


kmeans = KMeans(
    init="random",
    n_clusters=2,#    n_clusters=3,

    n_init=10,
    max_iter=300,
    random_state=42
)

kmeans.fit(scaled_features)

# The lowest SSE value
print(kmeans.inertia_)


# Final locations of the centroid
print(kmeans.cluster_centers_)





# The number of iterations required to converge
print(kmeans.n_iter_)
print(kmeans.labels_)
