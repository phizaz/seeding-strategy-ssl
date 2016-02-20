import sklearn

# inspect over K for Kmeans
def kmeans(data):
    last_inertia = 1000000000
    inertias = []
    improvements = []
    for clusters in range(1,10):
        print('clusters:', clusters)

        kmeans = sklearn.cluster.KMeans(n_clusters=clusters)
        kmeans.fit(data)

        print('cluster_centers:', kmeans.cluster_centers_)
        print('labels:', kmeans.labels_)
        print('inertia:', kmeans.inertia_)

        improvement = kmeans.inertia_ / last_inertia
        improvements.append(improvement)
        inertias.append(kmeans.inertia_)
        print('improvement:', improvement, '%')

        last_inertia = kmeans.inertia_

    print('inertias:', inertias)
    print('improvements:', improvements)