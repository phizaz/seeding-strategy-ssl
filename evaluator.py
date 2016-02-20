import alg
import util
import sklearn.cross_validation
import numpy


def cluster_n_label_knn(cluster_count, seed_count, data, target, n_seeding = 5):
    #goodK, _ = util.goodKforKNN(data, target)
    goodK = 3
    print('good K:', goodK)

    accuracies = []
    # remove uncertainty of randomized seeds
    for nth_seeding in range(n_seeding):
        #print('seeding th:', nth_seeding)
        knn_ssl = alg.cluster_n_label_classifier(cluster_count=cluster_count,
                                                 seed_count=seed_count,
                                                 goodK=goodK)
        scores = sklearn.cross_validation.cross_val_score(knn_ssl, data, target,cv=5, n_jobs=2)

        # print('scores:', scores)
        acc = scores.mean()
        accuracies.append(acc)

    expected_acc = numpy.array(accuracies).mean()

    return expected_acc
