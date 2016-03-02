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
    return mean ** 0.5
