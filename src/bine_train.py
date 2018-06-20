#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'CLH'

from argparse import ArgumentParser, FileType, ArgumentDefaultsHelpFormatter
import sys
import numpy as np
from sklearn import preprocessing
from data_utils import DataUtils
from graph_utils import GraphUtils
from scipy.stats.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import entropy
import random
import math
import os
import time

def link_score(emb, a, b, args):
	if args.link_method == "cos":
		try:
			result = emb.similarity(str(a),str(b))
		except:
			print("something's wrong. a:{}, b:{}".format(a, b))
			result = 0
	elif args.link_method == "hadamard":
		result = np.multiply(emb[str(a)], emb[str(b)])
	elif args.link_method == "avg":
		result = (emb[str(a)]+emb[str(b)])/2.0
	elif args.link_method == "weight1":
		diff = emb[str(a)]-emb[str(b)]
		result = np.sqrt(diff.dot(diff))
	elif args.link_method == "weight2":
		diff = emb[str(a)]-emb[str(b)]
		result = diff.dot(diff)
	return result

def get_roc_score(emb, edges_pos, edges_neg, args):
	# Store positive edge predictions, actual values
	preds_pos = []
	for edge in edges_pos:
		preds_pos.append(link_score(emb, edge[0], edge[1], args)) # predicted score

	# Store negative edge predictions, actual values
	preds_neg = []
	for edge in edges_neg:
		preds_neg.append(link_score(emb, edge[0], edge[1], args)) # predicted score

	# Calculate scores
	preds_all = np.hstack([preds_pos, preds_neg])
	labels_all = np.hstack([np.ones(len(preds_pos)), np.zeros(len(preds_neg))])
	roc_score = roc_auc_score(labels_all, preds_all)
	ap_score = average_precision_score(labels_all, preds_all)
	return roc_score, ap_score

def js(p, q):
   # normalize
    p_norm = p/p.sum()
    q_norm = q/q.sum()
    m = (p_norm + q_norm) / 2
    return (entropy(p_norm, m) + entropy(q_norm, m)) / 2

def get_similarity(node_list_u, user1, user2, sim_method="cos"):
    user1 = np.array(node_list_u[user1]['embedding_vectors'])
    user2 = np.array(node_list_u[user2]['embedding_vectors'])
    if sim_method =="cos":
        sim = cosine_similarity(user1, user2)
    elif sim_method == "pearson":
        sim = pearsonr(user1, user2)[0]
    elif sim_method == "jsd":
        sim = js(user1, user2)
    return sim

def build_user_sim_matrx(node_list_u, user_nodes, sim_method):
    user_matrix = np.zeros((len(user_nodes), len(user_nodes)))
    for i, user in enumerate(user_nodes):
        for j in range(i+1, len(user_nodes)):
            similarity = get_similarity(node_list_u, str(user), str(user_nodes[j]), sim_method)
            user_matrix[i][j] = similarity
            user_matrix[j][i] = similarity
    return user_matrix

def get_add_edge_by_ratio(user_nodes, ratio, sim_method="cos", user_matrix=None, emb=None):
    add_edge_num = int(len(user_nodes)*ratio)
    add_edge = []
    for i, user in enumerate(user_nodes):
        if user_matrix is not None:
            user_user_sim_list = list(user_matrix[i])
        elif emb is not None:
            user_user_sim_list = [get_similarity(emb, str(user), str(user2), sim_method) for user2 in user_nodes]
            user_user_sim_list[i] = 0
        else:
            raise Exception("You should provide either the user_sim_matrix or the embedding")
        
        similarities = zip(user_nodes, user_user_sim_list)
        similarities = sorted(similarities, key=lambda tup:-tup[1])
        add_edge.extend([(user, x[0], 1) for x in similarities[:add_edge_num]])
    return add_edge

def get_add_edge_by_step(user_nodes, thre, sim_method="cos", user_matrix=None, emb=None):
    add_edge = []
    for i, user in enumerate(user_nodes):
        if user_matrix is not None:
            user_user_sim_list = list(user_matrix[i])
        elif emb is not None:
            user_user_sim_list = [get_similarity(emb, str(user), str(user2), sim_method) for user2 in user_nodes]
            user_user_sim_list[i] = 0
        else:
            raise Exception("You should provide either the user_sim_matrix or the embedding")
        similarities = zip(user_nodes, user_user_sim_list)
        similarities = [x for x in similarities if x[1]>thre]
        add_edge.extend([(user, x[0], 1) for x in similarities])
    return add_edge

def get_add_edge_by_relu(user_nodes, thre, sim_method="cos", user_matrix=None, emb=None):
    add_edge = []
    for i, user in enumerate(user_nodes):
        if user_matrix is not None:
            user_user_sim_list = list(user_matrix[i])
        elif emb is not None:
            user_user_sim_list = [get_similarity(emb, str(user), str(user2), sim_method) for user2 in user_nodes]
            user_user_sim_list[i] = 0
        else:
            raise Exception("You should provide either the user_sim_matrix or the embedding")
        similarities = zip(user_nodes, user_user_sim_list)
        similarities = [x for x in similarities if x[1]>thre]
        add_edge.extend([(user, x[0], x[1]) for x in similarities])
    return add_edge

def get_add_edge_by_relu_ratio(user_nodes, ratio, sim_method="cos", user_matrix=None, emb=None):
    add_edge_num = int(len(user_nodes)*ratio)
    add_edge = []
    for i, user in enumerate(user_nodes):
        if user_matrix is not None:
            user_user_sim_list = list(user_matrix[i])
        elif emb is not None:
            user_user_sim_list = [get_similarity(emb, str(user), str(user2), sim_method) for user2 in user_nodes]
            user_user_sim_list[i] = 0
        else:
            raise Exception("You should provide either the user_sim_matrix or the embedding")
        similarities = zip(user_nodes, user_user_sim_list)
        similarities = sorted(similarities, key=lambda tup:-tup[1])
        add_edge.extend([(user, x[0], x[1]) for x in similarities[:add_edge_num]])
    return add_edge

def get_add_edge_linear(user_nodes, sim_method="cos", user_matrix=None, emb=None):
    add_edge = []
    for i, user in enumerate(user_nodes):
        if user_matrix is not None:
            user_user_sim_list = list(user_matrix[i])
        elif emb is not None:
            user_user_sim_list = [get_similarity(emb, str(user), str(user2), sim_method) for user2 in user_nodes]
            user_user_sim_list[i] = 0
        else:
            raise Exception("You should provide either the user_sim_matrix or the embedding")
        similarities = zip(user_nodes, user_user_sim_list)
        add_edge.extend([(user, x[0], x[1]) for x in similarities])
    return add_edge

def add_user_edge(args, node_list_u, sim_method="cos", by_matrix=True):
    # if args.unseparated:
    #     user_nodes = [x for x in g.nodes()]
    # else:
    #     user_nodes = [x for x in g.nodes() if not str(x).startswith('9999999')]
    user_nodes = [user for user in node_list_u]
    user_matrix = build_user_sim_matrx(node_list_u, user_nodes, sim_method)
    emb = None
    if args.user_edges_mode == "ratio":
        add_edge = get_add_edge_by_ratio(user_nodes, args.user_edges_ratio, sim_method, user_matrix, emb)
    elif args.user_edges_mode == "step":
        add_edge = get_add_edge_by_step(user_nodes, args.user_edges_thre, sim_method, user_matrix, emb)
    elif args.user_edges_mode == "relu":
        add_edge = get_add_edge_by_relu(user_nodes, args.user_edges_thre, sim_method, user_matrix, emb)
    elif args.user_edges_mode == "relu-ratio":
        add_edge = get_add_edge_by_relu(user_nodes, args.user_edges_ratio, sim_method, user_matrix, emb)
    elif args.user_edges_mode == "linear":
        add_edge = get_add_edge_linear(user_nodes, sim_method, user_matrix, emb)
    else:
        raise Exception("user-edges-mode value fault: "+str(args.user_edges_mode))
    
    return add_edge

def init_embedding_vectors(node_u, node_v, node_list_u, node_list_v, args):
    """
    initialize embedding vectors
    :param node_u:
    :param node_v:
    :param node_list_u:
    :param node_list_v:
    :param args:
    :return:
    """
    # user
    for i in node_u:
        vectors = np.random.random([1, args.d])
        help_vectors = np.random.random([1, args.d])
        node_list_u[i] = {}
        node_list_u[i]['embedding_vectors'] = preprocessing.normalize(vectors, norm='l2')
        node_list_u[i]['context_vectors'] = preprocessing.normalize(help_vectors, norm='l2')
    # item
    for i in node_v:
        vectors = np.random.random([1, args.d])
        help_vectors = np.random.random([1, args.d])
        node_list_v[i] = {}
        node_list_v[i]['embedding_vectors'] = preprocessing.normalize(vectors, norm='l2')
        node_list_v[i]['context_vectors'] = preprocessing.normalize(help_vectors, norm='l2')



def walk_generator(gul,args):
    """
    walk generator
    :param gul:
    :param args:
    :return:
    """
    gul.calculate_centrality()
    if args.large == 0:
        gul.homogeneous_graph_random_walks(percentage=args.p, maxT=args.maxT, minT=args.minT)
    elif args.large == 1:
        gul.homogeneous_graph_random_walks_for_large_bipartite_graph(datafile=args.train_data, percentage=args.p, maxT=args.maxT, minT=args.minT)
    return gul


def get_context_and_negative_samples(gul, args):
    """
    get context and negative samples offline
    :param gul:
    :param args:
    :return: context_dict_u, neg_dict_u, context_dict_v, neg_dict_v,gul.node_u,gul.node_v
    """
    neg_dict_u, neg_dict_v = gul.get_negs()
    print("negative samples is ok.....")
    if args.large == 0:
        context_dict_u, neg_dict_u = gul.get_context_and_negatives(gul.G_u, gul.walks_u, args.ws, args.ns, neg_dict_u)
        context_dict_v, neg_dict_v = gul.get_context_and_negatives(gul.G_v, gul.walks_v, args.ws, args.ns, neg_dict_v)
    else:
        context_dict_u, neg_dict_u = gul.get_context_and_negatives(gul.node_u, gul.walks_u, args.ws, args.ns, neg_dict_u)
        context_dict_v, neg_dict_v = gul.get_context_and_negatives(gul.node_v, gul.walks_v, args.ws, args.ns, neg_dict_v)
    return context_dict_u, neg_dict_u, context_dict_v, neg_dict_v,gul.node_u,gul.node_v


def skip_gram(center, contexts, negs, node_list, lam, pa):
    """
    skip-gram
    :param center:
    :param contexts:
    :param negs:
    :param node_list:
    :param lam:
    :param pa:
    :return:
    """
    loss = 0
    I_z = {center: 1}  # indication function
    for node in negs:
        I_z[node] = 0
    V = np.array(node_list[contexts]['embedding_vectors'])
    update = [[0] * V.size]
    for u in I_z.keys():
        if node_list.get(u) is  None:
            pass
        Theta = np.array(node_list[u]['context_vectors'])
        X = float(max(V.dot(Theta.T), 0))
        sigmod = 1.0 / (1 + (math.exp(-X * 1.0)))
        update += pa * lam * (I_z[u] - sigmod) * Theta
        node_list[u]['context_vectors'] += pa * lam * (I_z[u] - sigmod) * V
        try:
            loss += pa * (I_z[u] * math.log(sigmod) + (1 - I_z[u]) * math.log(1 - sigmod))
        except:
            pass
            # print "skip_gram:",
            # print(V,Theta,sigmod,X,math.exp(-X * 1.0),round(math.exp(-X * 1.0),10))
    return update, loss


def KL_divergence(edge_dict_u, u, v, node_list_u, node_list_v, lam, gamma):
    """
    KL-divergenceO1
    :param edge_dict_u:
    :param u:
    :param v:
    :param node_list_u:
    :param node_list_v:
    :param lam:
    :param gamma:
    :return:
    """
    loss = 0
    e_ij = edge_dict_u[u][v]

    update_u = 0
    update_v = 0
    U = np.array(node_list_u[u]['embedding_vectors'])
    V = np.array(node_list_v[v]['embedding_vectors'])
    X = float(max(U.dot(V.T), 0))

    sigmod = 1.0 / (1 + (math.exp(-X * 1.0)))

    update_u += gamma * lam * ((e_ij * (1 - sigmod)) * 1.0 / math.log(10, math.e)) * V
    update_v += gamma * lam * ((e_ij * (1 - sigmod)) * 1.0 / math.log(10, math.e)) * U

    try:
        loss += gamma * e_ij * math.log(sigmod)
    except:
        pass
        # print "KL:",
        # print(U,V,sigmod,X,math.exp(-X * 1.0),round(math.exp(-X * 1.0),10))
    return update_u, update_v, loss

def top_N(test_u, test_v, test_rate, node_list_u, node_list_v, top_n):
    recommend_dict = {}
    for u in test_u:
        recommend_dict[u] = {}
        for v in test_v:
            if node_list_u.get(u) is None:
                pre = 0
            else:
                U = np.array(node_list_u[u]['embedding_vectors'])
                if node_list_v.get(v) is None:
                    pre = 0
                else:
                    V = np.array(node_list_v[v]['embedding_vectors'])
                    pre = U.dot(V.T)[0][0]
            recommend_dict[u][v] = float(pre)

    precision_list = []
    recall_list = []
    ap_list = []
    ndcg_list = []
    rr_list = []

    for u in test_u:
        tmp_r = sorted(recommend_dict[u].items(), lambda x, y: cmp(x[1], y[1]), reverse=True)[0:min(len(recommend_dict[u]),top_n)]
        tmp_t = sorted(test_rate[u].items(), lambda x, y: cmp(x[1], y[1]), reverse=True)[0:min(len(test_rate[u]),len(test_rate[u]))]
        tmp_r_list = []
        tmp_t_list = []
        for (item, rate) in tmp_r:
            tmp_r_list.append(item)

        for (item, rate) in tmp_t:
            tmp_t_list.append(item)
        pre, rec = precision_and_racall(tmp_r_list,tmp_t_list)
        ap = AP(tmp_r_list,tmp_t_list)
        rr = RR(tmp_r_list,tmp_t_list)
        ndcg = nDCG(tmp_r_list,tmp_t_list)
        precision_list.append(pre)
        recall_list.append(rec)
        ap_list.append(ap)
        rr_list.append(rr)
        ndcg_list.append(ndcg)
    precison = sum(precision_list) / len(precision_list)
    recall = sum(recall_list) / len(recall_list)
    #print(precison, recall)
    f1 = 2 * precison * recall / (precison + recall)
    map = sum(ap_list) / len(ap_list)
    mrr = sum(rr_list) / len(rr_list)
    mndcg = sum(ndcg_list) / len(ndcg_list)
    return f1,map,mrr,mndcg

def nDCG(ranked_list, ground_truth):
    dcg = 0
    idcg = IDCG(len(ground_truth))
    for i in range(len(ranked_list)):
        id = ranked_list[i]
        if id not in ground_truth:
            continue
        rank = i+1
        dcg += 1/ math.log(rank+1, 2)
    return dcg / idcg

def IDCG(n):
    idcg = 0
    for i in range(n):
        idcg += 1 / math.log(i+2, 2)
    return idcg

def AP(ranked_list, ground_truth):
    hits, sum_precs = 0, 0.0
    for i in range(len(ranked_list)):
        id = ranked_list[i]
        if id in ground_truth:
            hits += 1
            sum_precs += hits / (i+1.0)
    if hits > 0:
        return sum_precs / len(ground_truth)
    else:
        return 0.0

def RR(ranked_list, ground_list):

    for i in range(len(ranked_list)):
        id = ranked_list[i]
        if id in ground_list:
            return 1 / (i + 1.0)
    return 0

def precision_and_racall(ranked_list,ground_list):
    hits = 0
    for i in range(len(ranked_list)):
        id = ranked_list[i]
        if id in ground_list:
            hits += 1
    pre = hits/(1.0 * len(ranked_list))
    rec = hits/(1.0 * len(ground_list))
    return pre, rec

def pre_train(node_list_u, node_list_v,edge_list,edge_dict_u, args):
    alpha, beta, gamma, lam = args.alpha, args.beta, args.gamma, args.lam
    last_loss = 0
    epsilon = 1e-3
    for iter in range(0, 50):
        s1 = "\r[%s%s]%0.2f%%"%("*"* iter," "*(args.max_iter-iter),iter*50.0/(args.max_iter-1))
        loss = 0
        num = 0
        for (u, v, w) in edge_list:
            update_u, update_v, tmp_loss = KL_divergence(edge_dict_u, u, v, node_list_u, node_list_v, lam, gamma)
            loss += tmp_loss
            node_list_u[u]['embedding_vectors'] += update_u
            node_list_v[v]['embedding_vectors'] += update_v
            num += 1
        delt_loss = abs(loss - last_loss)
        if last_loss > loss:
            lam *= 1.05
        else:
            lam *= 0.95
        last_loss = loss
        if delt_loss < epsilon:
            break
        sys.stdout.write(s1)
        sys.stdout.flush()

def train(args, gul):
    alpha, beta, gamma, lam = args.alpha, args.beta, args.gamma, args.lam
    model_path = os.path.join('../', args.model_name)
    if os.path.exists(model_path) is False:
        os.makedirs(model_path)
    dul = DataUtils(model_path)
    if args.rec:
        test_user, test_item, test_rate = dul.read_data(args.test_data)
    edge_dict_u = gul.edge_dict_u
    edge_list = gul.edge_list
    walk_generator(gul,args)

    print("getting context and negative samples....")
    context_dict_u, neg_dict_u, context_dict_v, neg_dict_v, node_u, node_v = get_context_and_negative_samples(gul, args)
    node_list_u, node_list_v = {}, {}
    init_embedding_vectors(node_u, node_v, node_list_u, node_list_v, args)
    # print("pretraining....")
    # pre_train(node_list_u,node_list_v,edge_list,edge_dict_u,args)

    last_loss, count, epsilon = 0, 0, 1e-3
    print("============== training ==============")
    for iter in range(0, args.max_iter):
        s1 = "\r[%s%s]%0.2f%%"%("*"* iter," "*(args.max_iter-iter),iter*100.0/(args.max_iter-1))
        loss = 0
        num = 0
        visited_u = dict(zip(node_list_u.keys(), [0] * len(node_list_u.keys())))
        visited_v = dict(zip(node_list_v.keys(), [0] * len(node_list_v.keys())))

        for (u, v, w) in edge_list:
            if visited_u.get(u) == 0:
                # print(u)
                length = len(context_dict_u[u])
                index_list = random.sample(list(range(length)), min(length, 10))
                for index in index_list:
                    context_u = context_dict_u[u][index]
                    neg_u = neg_dict_u[u][index]
                    # center,context,neg,node_list,eta
                    for z in context_u:
                        tmp_z, tmp_loss = skip_gram(u, z, neg_u, node_list_u, lam, alpha)
                        node_list_u[z]['embedding_vectors'] += tmp_z
                        loss += tmp_loss
                visited_u[u] = 1
            if visited_v.get(v) == 0:
                # print(v)
                length = len(context_dict_v[v])
                index_list = random.sample(list(range(length)), min(length, 10))
                for index in index_list:
                    context_v = context_dict_v[v][index]
                    neg_v = neg_dict_v[v][index]
                    # center,context,neg,node_list,eta
                    for z in context_v:
                        tmp_z, tmp_loss = skip_gram(v, z, neg_v, node_list_v, lam, beta)
                        node_list_v[z]['embedding_vectors'] += tmp_z
                        loss += tmp_loss
                visited_v[v] = 1
            # print(len(edge_dict_u))
            update_u, update_v, tmp_loss = KL_divergence(edge_dict_u, u, v, node_list_u, node_list_v, lam, gamma)
            loss += tmp_loss
            node_list_u[u]['embedding_vectors'] += update_u
            node_list_v[v]['embedding_vectors'] += update_v
            count = iter
            num += 1
        delta_loss = abs(loss - last_loss)
        if last_loss > loss:
            lam *= 1.05
        else:
            lam *= 0.95
        last_loss = loss
        if delta_loss < epsilon:
            break
        sys.stdout.write(s1)
        sys.stdout.flush()
    save_to_file(node_list_u,node_list_v,model_path,args)
    print("")
    roc = 0
    ap = 0
    # roc, ap = get_roc_score(node_list_u, node_list_v, edges_pos, edges_neg, args)
    if args.rec:
        print("============== testing ===============")
        f1, map, mrr, mndcg = top_N(test_user,test_item,test_rate,node_list_u,node_list_v,args.top_n)
        print('recommendation metrics: F1 : %0.4f, MAP : %0.4f, MRR : %0.4f, NDCG : %0.4f' % (f1, map, mrr, mndcg))
    
    return node_list_u, roc, ap
    


def ndarray_tostring(array):
    string = ""
    for item in array[0]:
        string += str(item).strip()+" "
    return string+"\n"

def save_to_file(node_list_u,node_list_v,model_path,args):
    with open(os.path.join(model_path,"vectors_u.dat"),"w") as fw_u:
        for u in node_list_u.keys():
            fw_u.write(u+" "+ ndarray_tostring(node_list_u[u]['embedding_vectors']))
    with open(os.path.join(model_path,"vectors_v.dat"),"w") as fw_v:
        for v in node_list_v.keys():
            fw_v.write(v+" "+ndarray_tostring(node_list_v[v]['embedding_vectors']))

def parser():
    parser = ArgumentParser("BiNE",
                            formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')

    parser.add_argument('--train-data', default=r'BiNE/data/test_rating_train.dat',
                        help='Input graph file.')

    parser.add_argument('--test-data', default=r'BiNE/data/test_rating_test.dat')
    parser.add_argument('--model-name', default='dblp',
                        help='name of model.')

    parser.add_argument('--ws', default=5, type=int,
                        help='window size.')

    parser.add_argument('--ns', default=4, type=int,
                        help='number of negative samples.')

    parser.add_argument('--d', default=128, type=int,
                        help='embedding size.')

    parser.add_argument('--maxT', default=32, type=int,
                        help='maximal walks per vertex.')

    parser.add_argument('--minT', default=1, type=int,
                        help='minimal walks per vertex.')

    parser.add_argument('--p', default=0.15, type=float,
                        help='walk stopping probability.')

    parser.add_argument('--alpha', default=0.01, type=float,
                        help='trade-off parameter alpha.')

    parser.add_argument('--beta', default=0.01, type=float,
                        help='trade-off parameter beta.')

    parser.add_argument('--gamma', default=0.1, type=float,
                        help='trade-off parameter gamma.')

    parser.add_argument('--lam', default=0.01, type=float,
                        help='learning rate lambda.')
    parser.add_argument('--max-iter', default=50, type=int,
                        help='maximal number of iterations.')

    parser.add_argument('--top-n', default=10, type=int,
                        help='recommend top-n items for each user.')

    parser.add_argument('--rec', default=1, type=int,
                        help='calculate the recommendation metrics.')

    parser.add_argument('--large', default=1, type=int,
                        help='for large bipartite, do not generate homogeneous graph file.')

    parser.add_argument('--add-user-edges', action='store_true', default=True,
                        help='add additional similar user edges')
    parser.add_argument('--user-edges-mode', action='store', default='ratio',
                        help='select mode for adding user edges (ratio/step/relu)')
    parser.add_argument('--user-edges-ratio', default=0.1, type=float,
                        help='ratio for train/test edges')
    parser.add_argument('--user-edges-thre', default=0.5, type=float,
                        help='threshold for ReLu activation function')
    parser.add_argument('--sim-method', default='cos', action='store')
    parser.add_argument('--verbose', action='store_true', default=False)
    args = parser.parse_args()
    return args


def main():
    args = parser()
    model_path = os.path.join('../', args.model_name)
    if os.path.exists(model_path) is False:
        os.makedirs(model_path)
    alpha, beta, gamma, lam = args.alpha, args.beta, args.gamma, args.lam
    print('======== experiment settings =========')
    print('alpha : %0.4f, beta : %0.4f, gamma : %0.4f, lam : %0.4f, p : %0.4f, ws : %d, ns : %d, maxT : % d, minT : %d, max_iter : %d' % (alpha, beta, gamma, lam, args.p, args.ws, args.ns,args.maxT,args.minT,args.max_iter))
    print('========== processing data ===========')
    print("constructing graph....")
    gul = GraphUtils(model_path)
    gul.construct_training_graph(args.train_data)
    node_list_u, roc, ap = train(args, gul)

    if args.add_user_edges:
        #do something
        print("Adding user edges")
        # add_edges = add_user_edge(args, nx_G, emb, by_matrix=args.add_by_matrix)
        add_edges = add_user_edge(args, node_list_u, sim_method=args.sim_method, by_matrix=False)
        if args.verbose: print("\ngot adding edges ({})".format(time.strftime("%H:%M")))
        gul.G.add_weighted_edges_from(add_edges)
        if args.verbose: print("added weight to graph ({})".format(time.strftime("%H:%M")))
        node_list_u, roc_add, ap_add = train(args, gul)
        # if verbose: print('# nodes: {}, #train edges: {}, #test edges: {}'.format(len(nx_G.nodes()),len(nx_G.edges()), len(test_edges)))
        # G = node2vec.Graph(nx_G, args.directed, args.p, args.q, args.popwalk)
        # del nx_G
        # if verbose: print("constructed node2vec instance ({})".format(time.strftime("%H:%M")))
        # if args.by_chunk:
        #     emb = learn_embeddings_by_chunk(args, G, args.chunk_size, args.multi_num, verbose=verbose, added=True)
        #     if verbose: print("learn embeddings by chunk ({})".format(time.strftime("%H:%M")))
        # else:
        #     walks = simulate_walk_popularity_multi(args, G, num_pool=args.multi_num)
        #     # walks = simulate_walk_popularity(args, G)
        #     total_walk_file_name = get_walk_file_name(args, True)
        #     with open(total_walk_file_name, "w") as file:
        #         file.write("\n".join([" ".join(map(str, walk)) for walk in walks])+"\n")
        #     if verbose: print("saved walks to file {}".format(total_walk_file_name))
        #     if verbose: print("simulated walk ({})".format(time.strftime("%H:%M")))
        #     emb = learn_embeddings(walks)
        #     if verbose: print("learnt embeddings ({})".format(time.strftime("%H:%M")))
        #     del walks
        # nx_G = G.G
        # del G
        # results_user, final_results_user = link_prediction(args, nx_G, emb, train_edges, test_edges, ks=ks) if args.prediction else (False, False) if args.prediction else None, None
        # roc_score_user, ap_score_user = get_roc_score(emb, test_edges, neg_edges, args) if args.auc else (0,0)
        # if verbose: print("got roc score: {} ({})".format(roc_score_user, time.strftime("%H:%M")))
        # if verbose: print("got ap score: {} ({})".format(ap_score_user, time.strftime("%H:%M")))
        
        


if __name__ == "__main__":
    main()
    # sys.exit(main())