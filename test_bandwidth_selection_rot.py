from dataset import *
from bandwidth_selection import BandwidthSelection

datasets = [
    get_iris_with_test(bandwidth='cv_ml').rescale(),
    get_pendigits(bandwidth='cv_ml').rescale(),
    get_yeast_with_test(bandwidth='cv_ml').rescale(),
    get_satimage(bandwidth='cv_ml').rescale(),
    get_banknote_with_test(bandwidth='cv_ml').rescale(),
    get_spam_with_test(bandwidth='cv_ml').rescale(),
    get_drd_with_test(bandwidth='cv_ml').rescale(),
    get_imagesegment(bandwidth='cv_ml').rescale(),
    get_pageblock_with_test(bandwidth='cv_ml').rescale(),
    get_statlogsegment_with_test(bandwidth='cv_ml').rescale(),
    get_winequality_with_test('white', bandwidth='cv_ml').rescale(),
    get_winequality_with_test('red', bandwidth='cv_ml').rescale(),
]

datasets.sort(key=lambda dataset: len(dataset.X))

for dataset in datasets:
    print('generating for', dataset.name)
    bw = BandwidthSelection.gaussian_distribution(dataset.X)
    print('bw:', 'rot:', bw, 'cv_ml:', dataset.bandwidth)
