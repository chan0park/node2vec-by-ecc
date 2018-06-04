import argparse
import heapq

import numpy as np
from time import strftime
from six import iteritems
from surprise import Reader
from surprise import Dataset
from surprise import accuracy
from surprise.model_selection import cross_validate
from surprise.model_selection import train_test_split
from surprise import PredictionImpossible
from surprise import AlgoBase
from surprise import similarities as sims
from settings import *
from utils import calculate_ie_from_iu, calculate_ir_from_iu, calculate_ire_from_iu, calculate_ier_from_iu, import_ml, mark_timewindow, check_and_import, save_pickle

# parse arg
parser = argparse.ArgumentParser(description='Model hyperparameters.')
parser.add_argument('-desc', type=str, action='store', default='', help='description for this run')
parser.add_argument('-input', type=str, action='store', default='data/ml-1m-ratings.csv', help='input')
parser.add_argument('-k', type=int, action='store', default=20, help='Number of neighbors of KNN algorithm')
parser.add_argument('-mink', type=int, action='store', default=5, help='Minimum number of common items/users')
parser.add_argument('-algo', type=str, action='store', default='eccen', help='algorithm (knn/svd)')
parser.add_argument('-sim', type=str, action='store', default='cosine', help='algorithm (msd/cosine/pearson_baseline/pearson)')
parser.add_argument('-mode', type=str, action='store', default='ir', help='ie/ir')
parser.add_argument('-item-based', action='store_true', default=False, help='Item_based CF')
parser.add_argument('-cv', action='store_true', default=True, help='Bollean specifying Cross Validation')
args = parser.parse_args()
title_str = "{}: input {}/algo {}/sim {}/mode {}/cv {}/item_based {}/k {}/mink {}".format(args.desc, args.input, args.algo, args.sim, args.mode, args.cv, args.item_based, args.k, args.mink)
print(title_str)

class SymmetricAlgo(AlgoBase):
    """When the algo is user-based x denotes a user and y an item. Else, it's
    reversed.
    """

    def __init__(self, sim_options={}, verbose=True, **kwargs):

        AlgoBase.__init__(self, sim_options=sim_options, **kwargs)
        self.verbose = verbose

    def fit(self, trainset):

        AlgoBase.fit(self, trainset)

        ub = self.sim_options['user_based']
        self.n_x = self.trainset.n_users if ub else self.trainset.n_items
        self.n_y = self.trainset.n_items if ub else self.trainset.n_users
        self.xr = self.trainset.ur if ub else self.trainset.ir
        self.yr = self.trainset.ir if ub else self.trainset.ur

        return self

    def switch(self, u_stuff, i_stuff):
        """Return x_stuff and y_stuff depending on the user_based field."""

        if self.sim_options['user_based']:
            return u_stuff, i_stuff
        else:
            return i_stuff, u_stuff

class EccenKNN(SymmetricAlgo):

    def __init__(self, k=40, min_k=1, sim_options={}, mode='ie', verbose=True, **kwargs):

        SymmetricAlgo.__init__(self, sim_options=sim_options, verbose=verbose,
                               **kwargs)
        self.k = k
        self.min_k = min_k
        self.mode=mode

    def cosine_eccen(self, n_x, yr, min_support, i_dict):
        # # sum (r_xy * r_x'y) for common ys
        # cdef np.ndarray[np.double_t, ndim=2] prods
        # # number of common ys
        # cdef np.ndarray[np.int_t, ndim=2] freq
        # # sum (r_xy ^ 2) for common ys
        # cdef np.ndarray[np.double_t, ndim=2] sqi
        # # sum (r_x'y ^ 2) for common ys
        # cdef np.ndarray[np.double_t, ndim=2] sqj
        # # the similarity matrix
        # cdef np.ndarray[np.double_t, ndim=2] sim

        # cdef int xi, xj
        # cdef double ri, rj
        # cdef int min_sprt = min_support

        prods = np.zeros((n_x, n_x), np.double)
        freq = np.zeros((n_x, n_x), np.int)
        sqi = np.zeros((n_x, n_x), np.double)
        sqj = np.zeros((n_x, n_x), np.double)
        sim = np.zeros((n_x, n_x), np.double)

        for y, y_ratings in iteritems(yr):
            for xi, ri in y_ratings:
                for xj, rj in y_ratings:
                    freq[xi, xj] += 1
                    prods[xi, xj] += ri * rj * i_dict[y]
                    sqi[xi, xj] += ri**2
                    sqj[xi, xj] += rj**2

        for xi in range(n_x):
            sim[xi, xi] = 1
            for xj in range(xi + 1, n_x):
                if freq[xi, xj] < min_support:
                    sim[xi, xj] = 0
                else:
                    denum = np.sqrt(sqi[xi, xj] * sqj[xi, xj])
                    sim[xi, xj] = prods[xi, xj] / denum

                sim[xj, xi] = sim[xi, xj]

        return sim


    def msd_eccen(self, n_x, yr, min_support, i_dict):
        # # sum (r_xy - r_x'y)**2 for common ys
        # cdef np.ndarray[np.double_t, ndim=2] sq_diff
        # # number of common ys
        # cdef np.ndarray[np.int_t, ndim=2] freq
        # # the similarity matrix
        # cdef np.ndarray[np.double_t, ndim=2] sim

        # cdef int xi, xj
        # cdef double ri, rj
        # cdef int min_sprt = min_support

        sq_diff = np.zeros((n_x, n_x), np.double)
        freq = np.zeros((n_x, n_x), np.int)
        sim = np.zeros((n_x, n_x), np.double)

        for y, y_ratings in iteritems(yr):
            for xi, ri in y_ratings:
                for xj, rj in y_ratings:
                    # sq_diff[xi, xj] += ((ri - rj)/(i_dict[y]+1))**2
                    sq_diff[xi, xj] += ((ri - rj)*(i_dict[y]))**2
                    freq[xi, xj] += 1

        for xi in range(n_x):
            sim[xi, xi] = 1  # completely arbitrary and useless anyway
            for xj in range(xi + 1, n_x):
                if freq[xi, xj] < min_support:
                    sim[xi, xj] == 0
                else:
                    # return inverse of (msd + 1) (+ 1 to avoid dividing by zero)
                    sim[xi, xj] = 1 / (sq_diff[xi, xj] / freq[xi, xj] + 1)

                sim[xj, xi] = sim[xi, xj]

        return sim

    def compute_similarities_eccen(self, i_dict):
        """Build the similarity matrix.

        The way the similarity matrix is computed depends on the
        ``sim_options`` parameter passed at the creation of the algorithm (see
        :ref:`similarity_measures_configuration`).

        This method is only relevant for algorithms using a similarity measure,
        such as the :ref:`k-NN algorithms <pred_package_knn_inpired>`.

        Returns:
            The similarity matrix."""

        construction_func = {'cosine': self.cosine_eccen,
                             'msd': self.msd_eccen,
                             'pearson': sims.pearson,
                             'pearson_baseline': sims.pearson_baseline}

        if self.sim_options['user_based']:
            n_x, yr = self.trainset.n_users, self.trainset.ir
        else:
            n_x, yr = self.trainset.n_items, self.trainset.ur

        min_support = self.sim_options.get('min_support', 1)

        args = [n_x, yr, min_support]

        name = self.sim_options.get('name', 'msd').lower()
        if name == 'pearson_baseline':
            shrinkage = self.sim_options.get('shrinkage', 100)
            bu, bi = self.compute_baselines()
            if self.sim_options['user_based']:
                bx, by = bu, bi
            else:
                bx, by = bi, bu

            args += [self.trainset.global_mean, bx, by, shrinkage]

        if not name in construction_func:
            raise NameError('Wrong sim name ' + name + '. Allowed values ' +
                            'are ' + ', '.join(construction_func.keys()) + '.')            
        if self.verbose:
            timestr = strftime("%H:%M:%S")
            print('Computing the {} similarity matrix... ({})'.format(name, timestr))
        sim = construction_func[name](*args, i_dict=i_dict)
        if self.verbose:
            timestr = strftime("%H:%M:%S")
            print('Done computing similarity matrix. ({})'.format(timestr))
        return sim

    def fit(self, trainset):
        SymmetricAlgo.fit(self, trainset)
        if self.mode == "ie":
            if '30' in args.input:
                df_ie = check_and_import('df/df_30_ie_zeroone')
            elif 'ml' in args.input:
                file_name = 'df_'+args.input.replace('./data/','').replace('data/','').replace('ratings.csv', '').replace('-','_')+'ie_zeroone'
                df_ie = check_and_import('df/'+file_name)

            if df_ie is None:
                print("Calculating Item Eccentricity")
                if '30' in args.input:
                    df_iu = import_ml('data/30-ratings.csv', headercol=['uid','id','feedback','timewindow'])
                elif 'ml' in args.input:
                    df_iu = import_ml(args.input, headercol=['uid','id','feedback','timestamp'])
                    df_iu = mark_timewindow(df_iu)
                    df_iu.columns = ['uid','id','feedback','timestamp','timewindow']
                df_iu.feedback = df_iu.feedback.astype(float)
                df_ie = calculate_ie_from_iu(df_iu)
                if '30' in args.input:
                    save_pickle(df_ie, 'df_30_ie_zeroone')
                elif 'ml' in args.input:
                    save_pickle(df_ie, file_name)
                del df_iu
        elif self.mode == "ir":
            if '30' in args.input:
                df_ie = check_and_import('df/df_30_i_zeroone')
            elif 'ml' in args.input:
                file_name = 'df_'+args.input.replace('./data/','').replace('data/','').replace('ratings.csv', '').replace('-','_')+'i_zeroone'
                df_ie = check_and_import('df/'+file_name)

            if df_ie is None:
                print("Calculating Item Rarity")
                if '30' in args.input:
                    df_iu = import_ml('data/30-ratings.csv', headercol=['uid','id','feedback','timewindow'])
                else:
                    df_iu = import_ml(args.input, headercol=['uid','id','feedback','timestamp'])
                    df_iu = mark_timewindow(df_iu)
                    df_iu.columns = ['uid','id','feedback','timestamp','timewindow']
                df_iu.feedback = df_iu.feedback.astype(float)
                df_ie = calculate_ir_from_iu(df_iu)
                if '30' in args.input:
                    save_pickle(df_ie, 'df_30_i_zeroone')
                elif 'ml' in args.input:
                    save_pickle(df_ie, file_name)
                del df_iu
        elif self.mode == "ire":
            if '30' in args.input:
                df_ie = check_and_import('df/df_30_ire_zeroone')
            elif 'ml' in args.input:
                file_name = 'df_'+args.input.replace('./data/','').replace('data/','').replace('ratings.csv', '').replace('-','_')+'ire_zeroone'
                df_ie = check_and_import('df/'+file_name)

            if df_ie is None:
                print("Calculating Item Rarity * Item Eccentricity")
                if '30' in args.input:
                    df_iu = import_ml('data/30-ratings.csv', headercol=['uid','id','feedback','timewindow'])
                else:
                    df_iu = import_ml(args.input, headercol=['uid','id','feedback','timestamp'])
                    df_iu = mark_timewindow(df_iu)
                    df_iu.columns = ['uid','id','feedback','timestamp','timewindow']
                df_iu.feedback = df_iu.feedback.astype(float)
                df_ie = calculate_ire_from_iu(df_iu)
                if '30' in args.input:
                    save_pickle(df_ie, 'df_30_ire_zeroone')
                elif 'ml' in args.input:
                    save_pickle(df_ie, file_name)
                
                del df_iu 
        elif self.mode == "ier":
            if '30' in args.input:
                df_ie = check_and_import('df/df_30_ier_zeroone')
            elif 'ml' in args.input:
                file_name = 'df_'+args.input.replace('./data/','').replace('data/','').replace('ratings.csv', '').replace('-','_')+'ier_zeroone'
                df_ie = check_and_import('df/'+file_name)

            if df_ie is None:
                print("Calculating Item Eccentricity / Item Rarity")
                if '30' in args.input:
                    df_iu = import_ml('data/30-ratings.csv', headercol=['uid','id','feedback','timewindow'])
                else:
                    df_iu = import_ml(args.input, headercol=['uid','id','feedback','timestamp'])
                    df_iu = mark_timewindow(df_iu)
                    df_iu.columns = ['uid','id','feedback','timestamp','timewindow']
                df_iu.feedback = df_iu.feedback.astype(float)
                df_ie = calculate_ier_from_iu(df_iu)
                if '30' in args.input:
                    save_pickle(df_ie, 'df_30_ier_zeroone')
                if 'ml' in args.input:
                    save_pickle(df_ie, file_name)
                
                del df_iu                        
        i_dict = {}
        for index, row in df_ie.iterrows():
            try:
                i_dict[trainset.to_inner_iid(row['id'])] = row[self.mode]
            except:
                pass
        self.sim = self.compute_similarities_eccen(i_dict)

        return self

    def estimate(self, u, i):

        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible('User and/or item is unkown.')

        x, y = self.switch(u, i)

        neighbors = [(self.sim[x, x2], r) for (x2, r) in self.yr[y]]
        k_neighbors = heapq.nlargest(self.k, neighbors, key=lambda t: t[0])

        # compute weighted average
        sum_sim = sum_ratings = actual_k = 0
        for (sim, r) in k_neighbors:
            if sim > 0:
                sum_sim += sim
                sum_ratings += sim * r
                actual_k += 1

        if actual_k < self.min_k:
            raise PredictionImpossible('Not enough neighbors.')

        est = sum_ratings / sum_sim

        details = {'actual_k': actual_k}
        return est, details


def main(data_path, line_format, sep=',', algo="knn", item_based=False, verbose=True):
    reader = Reader(line_format=line_format, sep=sep)
    sim_options = {'name':args.sim, 'user_based':not item_based}
    try:
        data = Dataset.load_from_file(data_path, reader=reader)
    except:
        reader = Reader(line_format=line_format, sep=sep, skip_lines=1)
        data = Dataset.load_from_file(data_path, reader=reader)
    raise
    if algo == "knn":
        from surprise import KNNBasic
        algo = KNNBasic(sim_options=sim_options, k=args.k, min_k=args.mink)
    elif algo == "svd":
        from surprise import SVD
        algo = SVD()
    elif algo == "eccen":
        algo = EccenKNN(sim_options=sim_options, k=args.k, min_k=args.mink, mode=args.mode)
    if args.cv:
        cross_validate(algo, data, n_jobs=1, verbose=verbose)
    else:
        trainset, testset = train_test_split(data, test_size=.2)
        algo.fit(trainset)
        predictions = algo.test(testset)
        accuracy.rmse(predictions)



if __name__ == "__main__":
    line_format = 'user item rating timestamp'
    main(data_path=args.input, line_format=line_format, algo=args.algo, item_based=args.item_based)