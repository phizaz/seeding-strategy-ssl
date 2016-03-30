from sklearn.grid_search import GridSearchCV
from sklearn.neighbors.kde import KernelDensity
from statsmodels.nonparametric.kernel_density import KDEMultivariate
import numpy as np

class BandwidthSelection:

    @staticmethod
    def gaussian_distribution(X):
        X = np.array(X)
        n = len(X)
        d = len(X[0])

        def sq_dist(x, y):
            diff = x - y
            return np.inner(diff, diff)

        def sd(X):
            mean = sum(X) / len(X)
            variance =  1 / n * sum(sq_dist(x, mean) for x in X)
            return variance ** 0.5

        bw = 1.06 * sd(X) * (n ** (-1 / 5))
        return bw

    @staticmethod
    def cv_maximum_likelihood(X,
                              folds=10,
                              search=np.linspace(1e-4, 1.0, 300)):
        X = np.array(X)
        # tends to give an undersmooth bandwidth
        # exhaustive search
        params = {
            'bandwidth': search
        }
        grid = GridSearchCV(
                KernelDensity(rtol=1e-6),
                params,
                cv=folds,
                n_jobs=-1)
        grid.fit(X)
        # print('best_params:', grid.best_params_)
        return grid.best_params_['bandwidth']