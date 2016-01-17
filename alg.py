import sklearn
import util

def cluster_n_label(cluster_count, seed_count, data, target):
    kmeans = sklearn.cluster.KMeans(n_clusters=cluster_count)
    kmeans.fit(data)

    total_score = 0
    avg_acc = None
    seed_numbers = util.seed_numbers(len(data), seed_count)

    #print('seed_numbers:', seed_numbers)

    clusters_labels = [ [] for i in range(cluster_count)]
    seed_labels = []
    for seed_number in seed_numbers:
        seed_label = target[seed_number]
        seed_data = data[seed_number]

        cluster = kmeans.predict([seed_data])
        clusters_labels[cluster].append(seed_label)
        seed_labels.append(seed_label)

    #print(seed_labels)

    #using a consensus from inside the same cluster

    major_labels = map(lambda cluster: util.most_common(cluster)[0] if len(cluster) > 0 else util.most_common(seed_labels)[0],
                       clusters_labels)
    major_labels = list(major_labels)

    #print('major:', major_labels)

    score = 0
    ssl_label = []
    for i, each in enumerate(data):
        prediction = kmeans.predict([each])[0]
        predicted_label = major_labels[prediction]
        ssl_label.append(predicted_label)

        # print('cluster:', prediction, 'label:', predicted_label, 'actual:', test[i][0])

        if target[i][0] == predicted_label:
            score += 1

    #print('score:', score, 'total:', len(data), 'acc:', (score / len(data)) * 100)

    return data, ssl_label

class cluster_n_label_classifier(sklearn.neighbors.KNeighborsClassifier):

    def __init__(self, cluster_count, seed_count, goodK):
        self.cluster_count = cluster_count
        self.seed_count = seed_count
        self.goodK = goodK
        super().__init__(n_neighbors=goodK)

    def fit(self, data, target):
        ssl_data, ssl_label = cluster_n_label(cluster_count=self.cluster_count,
                                              seed_count=self.seed_count,
                                              data=data,
                                              target=target)


        return super().fit(ssl_data, ssl_label)


