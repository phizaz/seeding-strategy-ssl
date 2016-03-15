from sklearn.neighbors import BallTree
from numpy import inner

def rmsd_nearest_from_centroids(seeding, centroids):
    # root mean squared distance from each centroids to its closest seeding
    ball_tree = BallTree(seeding)
    dist, idx = ball_tree.query(centroids)

    # root mean squared distance
    sum_sqdist = sum(d[0] ** 2 for d in dist)
    mean = sum_sqdist / len(centroids)
    return mean ** 0.5

def md_nearest_from_centroids(seeding, centroids):
    # mean distance
    ball_tree = BallTree(seeding)
    dist, idx = ball_tree.query(centroids)
    sum_dist = sum(d[0] for d in dist)
    mean = sum_dist / len(centroids)
    return mean

def md_weighted_nearest_from_centroids(seeding, centroids, weights):
    assert len(centroids) == len(weights)

    sum_weight = sum(weights)

    ball_tree = BallTree(seeding)
    dist, idx = ball_tree.query(centroids)
    sum_weighted_dist = sum(d[0] * weight for d, weight in zip(dist, weights))
    mean = sum_weighted_dist / sum_weight
    return mean

def voronoid_filling(seeding, centroids, weights):
    assert len(centroids) == len(weights)

    ball_tree = BallTree(centroids)
    _, indexes = ball_tree.query(seeding)
    filled_centroids = set()

    sum_weights = sum(weights)
    badness = sum_weights
    for idx, in indexes:
        if idx in filled_centroids:
            continue

        filled_centroids.add(idx)
        badness -= weights[idx]

    return badness / sum_weights