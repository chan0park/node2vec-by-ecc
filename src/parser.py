import argparse
from settings import *

'''
Parses the node2vec arguments.
'''
parser = argparse.ArgumentParser(description="Run node2vec.")

parser.add_argument('-input', nargs='?', default='graph/karate.edgelist',
                    help='Input graph path')

parser.add_argument('-output', nargs='?', default='emb/test.emb',
                    help='Embeddings path')

parser.add_argument('-dimensions', type=int, default=EMB_DIMENSION,
                    help='Number of dimensions. Default is 128.')

parser.add_argument('-walk-length', type=int, default=WALK_LENGTH,
                    help='Length of walk per source. Default is 80.')

parser.add_argument('-num-walks', type=int, default=NUM_WALKS,
                    help='Number of walks per source. Default is 10.')

parser.add_argument('-window-size', type=int, default=10,
                    help='Context size for optimization. Default is 10.')

parser.add_argument('-iter', type=int, default=SGD_EPOCHS,
                    help='Number of epochs in SGD')

parser.add_argument('-workers', type=int, default=WORKERS,
                    help='Number of parallel workers. Default is 8.')
parser.add_argument('-multi-num', type=int, default=MULTI_NUM,
                    help='Number of multi-processor.')
parser.add_argument('-add-multi-num', type=int, default=ADD_MULTI_NUM,
                    help='Number of multi-processor.')

parser.add_argument('-p', type=float, default=1,
                    help='Return hyperparameter. Default is 1.')

parser.add_argument('-q', type=float, default=1,
                    help='Inout hyperparameter. Default is 1.')

parser.add_argument('-weighted', dest='weighted', action='store_true', default=WEIGHTED_BOOL, 
                    help='Boolean specifying (un)weighted. Default is unweighted.')

parser.add_argument('-prediction', dest='prediction', action='store_true', default=False, 
                    help='Boolean specifying prediction process. Default is false.')
parser.add_argument('-auc', dest='auc', action='store', default=True, 
                    help='Boolean specifying prediction process. Default is false.')

parser.add_argument('-segment', default=1, type=int,
                    help='number of segments for evaluation')
parser.add_argument('-score-iter', default=SCORE_ITER, type=int,
                    help='number of iterations for auc scoring process')
parser.add_argument('-test-ratio', default=TEST_RATIO, type=float,
                    help='ratio for train/test edges')

parser.add_argument('-add-user-edges', action='store_true', default=ADD_USER_EDGE_BOOL,
                    help='add additional similar user edges')
parser.add_argument('-user-edges-mode', action='store', default=USER_EDGES_MODE,
                    help='select mode for adding user edges (ratio/step/relu)')
parser.add_argument('-user-edges-ratio', default=USER_EDGES_RATIO, type=float,
                    help='ratio for train/test edges')
parser.add_argument('-user-edges-thre', default=USER_EDGES_THRE, type=float,
                    help='threshold for ReLu activation function')

parser.add_argument('-link-method', action='store', default=LINK_METHOD,
                    help='select method for an edge representation (cos/avg/hadamard/weight1/weight2)')
parser.add_argument('-add-by-matrix', action='store', default=ADD_BY_MATRIX_BOOL,
                    help='select method for an edge representation (cos/avg/hadamard/weight1/weight2)')
                    

parser.add_argument('-popwalk', dest='popwalk', action='store', default=POPWALK,
                    help='three modes are supported for random walk(none/pop/both)')

parser.add_argument('-unseparated', dest='unseparated', action='store_true', default=UNSEPARATED_BOOL,
                    help='Boolean specifying user and item nodes are seperated. Default is separated.')
parser.add_argument('-directed', dest='directed', action='store_true', default=False,
                    help='Graph is (un)directed. Default is undirected.')
parser.add_argument('-undirected', dest='undirected', action='store_false', default=True)

parser.add_argument('-on-the-fly', default=ON_THE_FLY_BOOL, action='store_true', help='process random walk on the fly')

parser.add_argument('-by-chunk', dest='by_chunk', action='store_true', default=BY_CHUNK_BOOL)
parser.add_argument('-chunk-size', type=int, default=1000,
                    help='Number of nodes to simulate random walk and learn embedding')

parser.add_argument('-verbose', dest='verbose', action='store_true', default=VERBOSE_BOOL)
parser.add_argument('-walk-path', action='store', default=None, type=str)

parser.add_argument('-sim-method', action='store', default=SIMILARITY_METHOD, type=str)
parser.add_argument('-desc', action='store', default="", type=str)


args = parser.parse_args()
settings_str = "{}\nsettings: data {}/score_iter {}/tratio {}/add {}/dim {}/wlk_len {}/wlk_num {}/wndw {}/workers {}/mlti {}/add_multi {}/p {}/q {}/weighted {}/add_mode {}/ratio {}/thre {}/sim {}".format(args.desc, args.input, args.score_iter, args.test_ratio, args.add_user_edges, args.dimensions, args.walk_length, args.num_walks, args.window_size, args.workers, args.multi_num, args.add_multi_num, args.p, args.q, args.weighted, args.user_edges_mode, args.user_edges_ratio, args.user_edges_thre, args.sim_method)
print(settings_str)