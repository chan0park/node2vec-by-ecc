'''
Reference implementation of node2vec. 

Author: Aditya Grover

For more details, refer to the paper:
node2vec: Scalable Feature Learning for Networks
Aditya Grover and Jure Leskovec 
Knowledge Discovery and Data Mining (KDD), 2016
'''

import numpy as np
import networkx as nx
import node2vec
import random
import time
import math
import os
import tempfile
import pathos.pools as pp
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from settings import RANDOM_SEED
from parser import args

def read_graph(args):
	'''
	Reads the input network in networkx.
	'''
	if args.weighted:
		G = nx.read_edgelist(args.input, nodetype=int, data=(('weight',float),), create_using=nx.DiGraph())
	else:
		G = nx.read_edgelist(args.input, nodetype=int, create_using=nx.DiGraph())
		for edge in G.edges():
			G[edge[0]][edge[1]]['weight'] = 1

	if not args.directed:
		G = G.to_undirected()

	return G

def learn_embeddings(walks):
    '''Learn embeddings by optimizing the Skipgram objective using SGD.'''
    walks = [map(str, walk) for walk in walks]
    model = Word2Vec(walks, size=args.dimensions, window=args.window_size, min_count=0, sg=1, workers=args.workers, iter=args.iter)

    return model.wv

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

def precision_at_k(pred_k, test_edges):
    count = 0.0
    for pred in pred_k:
        if pred in test_edges or (pred[1],pred[0]) in test_edges:
            count += 1 
    return count/len(pred_k)

def make_links_and_score(emb, nodes, start_point, end_point, train_edges, ks, args, verbose=False):
    if len(nodes) == 1:
        nodes = nodes[0]
        edges_to_eval = set([(nodes[i],nodes[j]) for i in range(start_point, end_point) for j in range((i+1),len(nodes))])
        edges_to_eval -= set(train_edges)
        edges_to_eval = list(edges_to_eval)

    elif len(nodes) == 2:
        user_nodes = nodes[0]
        item_nodes = nodes[1]
        if verbose:
            print("{} {} {}".format(start_point, end_point, len(nodes[0])))
        edges_to_eval = set([(user_nodes[i],item) for i in range(start_point, end_point) for item in item_nodes])
        edges_to_eval -= set(train_edges)
        edges_to_eval = list(edges_to_eval)
    else:
        print("Error: length of nodes larger than 2 ({}). Something's wrong.".format(len(nodes)))
        edges_to_eval = []

    partial_result = links_score(emb, edges_to_eval, ks, args)

    return partial_result

def links_score(emb, edges_to_eval, ks, args, verbose=False):
    pred_score = []
    edges_to_eval_len = len(edges_to_eval)
    mark = int(edges_to_eval_len/2)

    print("Number of edges to score: {}".format(edges_to_eval_len))
    for i, edge in enumerate(edges_to_eval):
        if verbose:
            if i % mark == 0:
                print("Scoring process: {0:.1f}%".format(float(i*100.0)/edges_to_eval_len))
        pred_score.append(link_score(emb,edge[0],edge[1], args))
    score_index = np.argsort(pred_score)[::-1]
    partial_result = {}
    for k in ks:
        pred_k = [(edges_to_eval[x], pred_score[x]) for x in score_index[:k]]
        partial_result[k] = pred_k
    return partial_result

def calculate_pop(args, g, chosen_k_links):
    pop_list = []
    for link in chosen_k_links:
        node_a, node_b = int(link[0]), int(link[1])
        if args.unseparated:
            pop_list.append(int((len(g[node_a])+len(g[node_b]))/2))
        else:
            if str(node_a).startswith('9999999'):
                pop_list.append(len(g[node_a]))
            elif str(node_b).startswith('9999999'):
                pop_list.append(len(g[node_b]))
            
    assert len(chosen_k_links) == len(pop_list)
    return pop_list


def link_prediction(args, g, emb, train_edges, test_edges, ks=[1,10,50,100,500,1000]):
    # nodes = emb.vocab.keys()
    segment = args.segment
    nodes = [str(x) for x in sorted([int(x) for x in emb.vocab.keys()])]
    item_nodes, user_nodes = [], []
    
    if not args.unseparated:
        for x in nodes:
            if x.startswith('9999999'):
                item_nodes.append(x)
            else:
                user_nodes.append(x)
        nodes = (user_nodes, item_nodes)
        print("#user nodes: {}, #item nodes: {}".format(len(user_nodes), len(item_nodes)))
    else:
        nodes = (nodes,)

    batch_nodes = int(len(nodes[0])/segment)
    test_edges = [(str(x[0]), str(x[1])) for x in test_edges.tolist()]
    train_edges = [(str(x[0]), str(x[1])) for x in train_edges.tolist()]
    results = {}
    for k in ks:
        results[k] = []

    for i in range(segment):
        print("working on {}/{}th batch".format(i+1, segment))
        if not i == (segment-1):
            partial_results = make_links_and_score(emb, nodes, batch_nodes*i, batch_nodes*(i+1), train_edges, ks, args)
        else:
            partial_results = make_links_and_score(emb, nodes, batch_nodes*i, len(nodes[0]), train_edges, ks, args)

        for k in ks:
            results[k].extend(partial_results[k])
    
    final_results = {}
    for k in ks:
        results[k] = list(set(results[k]))
        temp = sorted(results[k], key=lambda tup:-tup[1])
        chosen_k_links = [x[0] for x in temp[:k]]
        chosen_k_links_pop = calculate_pop(args, g, chosen_k_links)
        avg_pop = sum(chosen_k_links_pop)/len(chosen_k_links_pop)
        results[k] = [x+(chosen_k_links_pop[i],) for i, x in enumerate(temp[:k])]
        prec = precision_at_k(chosen_k_links, test_edges)
        final_results[k] = (prec, avg_pop)
    return results, final_results

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

def build_neg_samples(nodes, true_edges):
    true_edges = set(true_edges)
    false_samples = set()
    while len(false_samples) < len(true_edges):
        sample_nodes = random.sample(nodes, 2)
        false_edge = (min(sample_nodes), max(sample_nodes))
        if false_edge in true_edges:
            continue
        if false_edge in false_samples:
            continue
        false_samples.add(false_edge)
    assert len(false_samples) == len(true_edges)
    false_samples = list(false_samples)
    return false_samples

def simulate_walk_popularity(args, G):
    if args.popwalk == "none":
        G.preprocess_transition_probs()
        walks = G.simulate_walks(args.num_walks, args.walk_length)
    elif args.popwalk == "pop":
        G.preprocess_transition_probs_popularity()
        walks = G.simulate_walks(args.num_walks, args.walk_length)
    elif args.popwalk == "both":
        G.preprocess_transition_probs()
        num_walks = int(args.num_walks/2)
        walks = G.simulate_walks(num_walks, args.walk_length)        
        G.preprocess_transition_probs_popularity()
        walks.extend(G.simulate_walks(num_walks, args.walk_length))
    return walks

def node2vec_walk_multi(num_walks, walk_length,  G, nodes):
    walks = []
    for _ in range(num_walks):
        for node in nodes:
            walks.append(G.node2vec_walk(walk_length, node))
    return walks

def node2vec_walk_multi_by_chunk(num_walks, walk_length,  G, nodes, chunk_size):
    import tempfile
    tmp_walks_txt = tempfile.NamedTemporaryFile(delete=False)
    node_chunks = list(chunks(nodes, chunk_size))
    for i, chunk in enumerate(node_chunks):
        walks = []
        for _ in range(num_walks):
            for node in chunk:
                walks.append(G.node2vec_walk(walk_length, node))
        walks = [" ".join(map(str, walk)) for walk in walks]
        with open(tmp_walks_txt.name, "a") as file:
            file.write("\n".join(walks)+"\n")
    return tmp_walks_txt.name

def node2vec_walk_multi_on_the_fly(num_walks, walk_length,  G, nodes):
    walks = G.simulate_walks_on_the_fly(num_walks, args.walk_length, nodes)
    return walks

def node2vec_walk_multi_on_the_fly_by_chunk(num_walks, walk_length,  G, nodes, chunk_size):
    import tempfile
    tmp_walks_txt = tempfile.NamedTemporaryFile(delete=False)
    tmp_walks_txt.close()
    # print(tmp_walks_txt.name)
    nodes = list(chunks(nodes, chunk_size))
    for i, chunk in enumerate(nodes):
        walks = G.simulate_walks_on_the_fly(num_walks, args.walk_length, chunk)
        walks = [" ".join(map(str, walk)) for walk in walks]
        with open(tmp_walks_txt.name, "a") as file:
            file.write("\n".join(walks)+"\n")
    return tmp_walks_txt.name

def simulate_walk_popularity_multi(args, G, num_pool=4):

    nodes = list(G.G.nodes())
    len_nodes = len(nodes)
    split_point = range(0,len_nodes,int(math.ceil(float(len_nodes)/num_pool)))+[len_nodes]
    splited_nodes = [nodes[split_point[i]:split_point[i+1]]for i in xrange(len(split_point)-1)]

    num_walks = int(args.num_walks/2) if args.popwalk == "both" else args.num_walks

    walks = []
    p = pp.ProcessPool(num_pool)
    p.restart()
    if args.on_the_fly:
        if args.popwalk == "both":
            G.popwalk == "none"
            walks = p.map(node2vec_walk_multi_on_the_fly, [num_walks]*num_pool, [args.walk_length]*num_pool, [G]*num_pool, splited_nodes)
            G.popwalk == "pop"
            walks.extend(p.map(node2vec_walk_multi_on_the_fly, [num_walks]*num_pool, [args.walk_length]*num_pool, [G]*num_pool, splited_nodes))
        else:
            walks = p.map(node2vec_walk_multi_on_the_fly, [num_walks]*num_pool, [args.walk_length]*num_pool, [G]*num_pool, splited_nodes)
    else:
        if args.popwalk == "none" or args.popwalk == "both":
            G.preprocess_transition_probs()
            walks = p.map(node2vec_walk_multi, [num_walks]*num_pool, [args.walk_length]*num_pool, [G]*num_pool, splited_nodes)
        if args.popwalk == "pop" or args.popwalk == "both":
            G.preprocess_transition_probs_popularity()
            walks.extend(p.map(node2vec_walk_multi, [num_walks]*num_pool, [args.walk_length]*num_pool, [G]*num_pool, splited_nodes))

    walks = [w for walk in walks for w in walk]
    return walks

def learn_embeddings_by_chunk(args, G, chunk_size, num_pool, verbose=True, added=False):
    nodes = list(G.G.nodes())
    len_nodes = len(nodes)
    node_per_num_pool = int(math.ceil(len_nodes*1.0/num_pool))
    splited_nodes = [nodes[i*node_per_num_pool:(i+1)*node_per_num_pool] for i in range(num_pool)]
    num_walks = int(args.num_walks/2) if args.popwalk == "both" else args.num_walks
    walks_files = []

    p = pp.ProcessPool(num_pool)
    if added:
        p.restart()
    if args.on_the_fly:
        if args.popwalk == "both":
            G.popwalk = "none"
            walks_files = p.map(node2vec_walk_multi_on_the_fly_by_chunk, [num_walks]*num_pool, [args.walk_length]*num_pool, [G]*num_pool, splited_nodes, [chunk_size]*num_pool)
            G.popwalk = "pop"
            walks_files.extend(p.map(node2vec_walk_multi_on_the_fly_by_chunk, [num_walks]*num_pool, [args.walk_length]*num_pool, [G]*num_pool, splited_nodes, [chunk_size]*num_pool))
        else:
            walks_files = p.map(node2vec_walk_multi_on_the_fly_by_chunk, [num_walks]*num_pool, [args.walk_length]*num_pool, [G]*num_pool, splited_nodes, [chunk_size]*num_pool)
    else:
        if args.popwalk == "none" or args.popwalk == "both":
            G.preprocess_transition_probs()
            walks_files = p.map(node2vec_walk_multi_by_chunk, [num_walks]*num_pool, [args.walk_length]*num_pool, [G]*num_pool, splited_nodes, [chunk_size]*num_pool)
        if args.popwalk == "pop" or args.popwalk == "both":
            G.preprocess_transition_probs_popularity()
            walks_files.extend(p.map(node2vec_walk_multi_by_chunk, [num_walks]*num_pool, [args.walk_length]*num_pool, [G]*num_pool, splited_nodes, [chunk_size]*num_pool))
    p.terminate()

    timestr = time.strftime("%y%m%d_%H%M%S")
    total_walk_file_name = args.input.replace("graph","walks").replace("edgelist","")+"num{}_length{}_pop{}_".format(args.num_walks, args.walk_length, args.popwalk)+timestr
    if added: total_walk_file_name = total_walk_file_name + "_user_edges_added"
    if verbose: print("total walks file: "+total_walk_file_name)
    for wfile in walks_files:
        os.system("cat "+wfile+" >> "+total_walk_file_name)
    
    sentences = LineSentence(total_walk_file_name)
    if verbose: print("start learning embedidngs")
    model = Word2Vec(sentences, size=args.dimensions, window=args.window_size, min_count=0, sg=1, workers=args.workers, iter=args.iter) 
    return model.wv
    
def build_user_sim_matrx(emb, user_nodes):
    user_matrix = np.zeros((len(user_nodes), len(user_nodes)))
    for i, user in enumerate(user_nodes):
        for j in range(i+1, len(user_nodes)):
            similarity = emb.similarity(str(user), str(user_nodes[j]))
            user_matrix[i][j] = similarity
            user_matrix[j][i] = similarity
    return user_matrix

def get_add_edge_by_ratio(user_nodes, ratio, user_matrix=None, emb=None):
    add_edge_num = int(len(user_nodes)*ratio)
    add_edge = []
    for i, user in enumerate(user_nodes):
        if user_matrix is not None:
            user_user_sim_list = list(user_matrix[i])
        elif emb is not None:
            user_user_sim_list = [emb.similarity(str(user), str(user2)) for user2 in user_nodes]
            user_user_sim_list[i] = 0
        else:
            raise Exception("You should provide either the user_sim_matrix or the embedding")
        
        similarities = zip(user_nodes, user_user_sim_list)
        similarities = sorted(similarities, key=lambda tup:-tup[1])
        add_edge.extend([(user, x[0], 1) for x in similarities[:add_edge_num]])
    return add_edge

def get_add_edge_by_step(user_nodes, thre, user_matrix=None, emb=None):
    add_edge = []
    for i, user in enumerate(user_nodes):
        if user_matrix is not None:
            user_user_sim_list = list(user_matrix[i])
        elif emb is not None:
            user_user_sim_list = [emb.similarity(str(user), str(user2)) for user2 in user_nodes]
            user_user_sim_list[i] = 0
        else:
            raise Exception("You should provide either the user_sim_matrix or the embedding")
        similarities = zip(user_nodes, user_user_sim_list)
        similarities = [x for x in similarities if x[1]>thre]
        add_edge.extend([(user, x[0], 1) for x in similarities])
    return add_edge

def get_add_edge_by_relu(user_nodes, thre, user_matrix=None, emb=None):
    add_edge = []
    for i, user in enumerate(user_nodes):
        if user_matrix is not None:
            user_user_sim_list = list(user_matrix[i])
        elif emb is not None:
            user_user_sim_list = [emb.similarity(str(user), str(user2)) for user2 in user_nodes]
            user_user_sim_list[i] = 0
        else:
            raise Exception("You should provide either the user_sim_matrix or the embedding")
        similarities = zip(user_nodes, user_user_sim_list)
        similarities = [x for x in similarities if x[1]>thre]
        add_edge.extend([(user, x[0], x[1]) for x in similarities])
    return add_edge

def get_add_edge_by_relu_ratio(user_nodes, ratio, user_matrix=None, emb=None):
    add_edge_num = int(len(user_nodes)*ratio)
    add_edge = []
    for i, user in enumerate(user_nodes):
        if user_matrix is not None:
            user_user_sim_list = list(user_matrix[i])
        elif emb is not None:
            user_user_sim_list = [emb.similarity(str(user), str(user2)) for user2 in user_nodes]
            user_user_sim_list[i] = 0
        else:
            raise Exception("You should provide either the user_sim_matrix or the embedding")
        similarities = zip(user_nodes, user_user_sim_list)
        similarities = sorted(similarities, key=lambda tup:-tup[1])
        add_edge.extend([(user, x[0], x[1]) for x in similarities[:add_edge_num]])
    return add_edge


def add_user_edge(args, g, emb, by_matrix=False):
    if args.unseparated:
        user_nodes = [x for x in g.nodes()]
    else:
        user_nodes = [x for x in g.nodes() if not str(x).startswith('9999999')]
    
    user_matrix = build_user_sim_matrx(emb, user_nodes) if by_matrix else None
    emb = None if by_matrix else emb
    if args.user_edges_mode == "ratio":
        add_edge = get_add_edge_by_ratio(user_nodes, args.user_edges_ratio, user_matrix, emb)
    elif args.user_edges_mode == "step":
        add_edge = get_add_edge_by_step(user_nodes, args.user_edges_thre, user_matrix, emb)
    elif args.user_edges_mode == "relu":
        add_edge = get_add_edge_by_relu(user_nodes, args.user_edges_thre, user_matrix, emb)
    elif args.user_edges_mode == "relu-ratio":
        add_edge = get_add_edge_by_relu(user_nodes, args.user_edges_ratio, user_matrix, emb)
    else:
        raise Exception("user-edges-mode value fault: "+str(args.user_edges_mode))
    
    return add_edge

def add_user_edge_by_chunk(args, g, emb, num_pool, by_matrix=False, added=False):
    if args.unseparated:
        user_nodes = [x for x in g.nodes()]
    else:
        user_nodes = [x for x in g.nodes() if not str(x).startswith('9999999')]
    node_per_num_pool = int(math.ceil(len(user_nodes)*1.0/num_pool))
    splited_nodes = [user_nodes[i*node_per_num_pool:(i+1)*node_per_num_pool] for i in range(num_pool)]

    user_matrix = build_user_sim_matrx(emb, user_nodes) if by_matrix else None
    emb = None if by_matrix else emb

    p = pp.ProcessPool(num_pool)
    if added:
        p.restart()
    if args.user_edges_mode == "ratio":
        add_edge = p.map(get_add_edge_by_ratio, splited_nodes, [args.user_edges_ratio]*num_pool, [user_matrix]*num_pool, [emb]*num_pool)
        add_edge = [x for chunk in add_edge for x in chunk]
    elif args.user_edges_mode == "step":
        add_edge = p.map(get_add_edge_by_step, splited_nodes, [args.user_edges_thre]*num_pool, [user_matrix]*num_pool, [emb]*num_pool)
        add_edge = [x for chunk in add_edge for x in chunk]
    elif args.user_edges_mode == "relu":
        add_edge = p.map(get_add_edge_by_relu, splited_nodes, [args.user_edges_thre]*num_pool, [user_matrix]*num_pool, [emb]*num_pool)
        add_edge = [x for chunk in add_edge for x in chunk]
    elif args.user_edges_mode == "relu-ratio":
        add_edge = p.map(get_add_edge_by_relu, splited_nodes, [args.user_edges_ratio]*num_pool, [user_matrix]*num_pool, [emb]*num_pool)
        add_edge = [x for chunk in add_edge for x in chunk]
    else:
        raise Exception("user-edges-mode value fault: "+str(args.user_edges_mode))
    p.terminate()
    return add_edge

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def main(args, ep, verbose=True):
    '''
    Pipeline for representational learning for all nodes in a graph.
    '''
    nx_G = read_graph(args)
    if verbose: print("readed graph ({})".format(time.strftime("%H:%M")))
    all_edges = nx_G.edges()
    train_edges, test_edges = train_test_split(np.asarray(all_edges), test_size=args.test_ratio, random_state=RANDOM_SEED)
    if verbose: print("splitted edges ({})".format(time.strftime("%H:%M")))

    if ep == 0:
        print('# nodes: {}, #train edges: {}, #test edges: {}'.format(len(nx_G.nodes()),len(train_edges), len(test_edges)))

    nx_G.remove_edges_from(test_edges)
    G = node2vec.Graph(nx_G, args.directed, args.p, args.q, args.popwalk)
    if verbose: print("Construced instance of node2vec class ({})".format(time.strftime("%H:%M")))

    if args.by_chunk:
        
        emb = learn_embeddings_by_chunk(args, G, args.chunk_size, args.multi_num)
        if verbose: print("learnt embedding by chunk ({})".format(time.strftime("%H:%M")))

    else:
        walks = simulate_walk_popularity_multi(args, G, num_pool=args.multi_num)
        if verbose: print("simulated walk ({})".format(time.strftime("%H:%M")))

        emb = learn_embeddings(walks)
        if verbose: print("learnt embedding ({})".format(time.strftime("%H:%M")))

        del walks
    del G
    
    ks = [1,5,10,15,50,100,500,1000]
    results, final_results = link_prediction(args, nx_G, emb, train_edges, test_edges, ks=ks) if args.prediction else (False, False) if args.prediction else None, None
    nodes = nx_G.nodes()
    neg_edges = build_neg_samples(nodes, all_edges)
    if verbose: print("built negative edges ({})".format(time.strftime("%H:%M")))

    roc_score, ap_score = get_roc_score(emb, test_edges, neg_edges, args) if args.auc else (0,0)
    if verbose: print("got roc score: {} ({})".format(roc_score, time.strftime("%H:%M")))
    del all_edges, nodes
    

    if args.add_user_edges and not args.unseparated:
        if args.by_chunk:
            add_edges = add_user_edge_by_chunk(args, nx_G, emb, args.add-multi_num, by_matrix=args.add_by_matrix)
        else:
            add_edges = add_user_edge(args, nx_G, emb, by_matrix=args.add_by_matrix)
        if verbose: print("got adding edges ({})".format(time.strftime("%H:%M")))
        nx_G.add_weighted_edges_from(add_edges)
        if verbose: print("added weight to graph ({})".format(time.strftime("%H:%M")))
        print('# nodes: {}, #train edges: {}, #test edges: {}'.format(len(nx_G.nodes()),len(nx_G.edges()), len(test_edges)))
        G = node2vec.Graph(nx_G, args.directed, args.p, args.q, args.popwalk)
        del nx_G
        if verbose: print("constructed node2vec instance ({})".format(time.strftime("%H:%M")))
        if args.by_chunk:
            emb = learn_embeddings_by_chunk(args, G, args.chunk_size, args.multi_num, added=True)
            if verbose: print("learn embeddings by chunk ({})".format(time.strftime("%H:%M")))
        else:
            walks = simulate_walk_popularity_multi(args, G, num_pool=args.multi_num)
            if verbose: print("simulated walk ({})".format(time.strftime("%H:%M")))
            emb = learn_embeddings(walks)
            if verbose: print("learnt embeddings ({})".format(time.strftime("%H:%M")))
            del walks
        nx_G = G.G
        del G
        results_user, final_results_user = link_prediction(args, nx_G, emb, train_edges, test_edges, ks=ks) if args.prediction else (False, False) if args.prediction else None, None
        roc_score_user, ap_score_user = get_roc_score(emb, test_edges, neg_edges, args) if args.auc else (0,0)
        if verbose: print("got roc score ({})".format(time.strftime("%H:%M")))
    else:
        roc_score_user, ap_score_user = None, None
    return results, final_results, roc_score, ap_score, roc_score_user, ap_score_user

if __name__ == "__main__":
    total_roc_score = []
    total_ap_score = []
    total_roc_score_user = []
    total_ap_score_user = []

    start_time = time.time()
    for ep in range(args.score_iter):
        results, final_results, roc_score, ap_score, roc_score_user, ap_score_user = main(args, ep)
        total_roc_score.append(roc_score)
        total_ap_score.append(ap_score)
        total_roc_score_user.append(roc_score_user)
        total_ap_score_user.append(ap_score_user)
        
    print("Taken time for {}epoch: {}\n".format(args.score_iter, round(time.time()-start_time,3)))

    total_roc_score = sum(total_roc_score)/len(total_roc_score)
    total_ap_score = sum(total_ap_score)/len(total_ap_score)

    if not final_results is None:
        final_results = sorted(final_results.items(), key=lambda tup: tup[0])
        with open('./link_pred_results/'+args.input.split('/')[-1]+'.results', 'w') as file:
            file.write(str(final_results))
        print("final_results"+str(final_results))

    print("roc_score: {} (iter: {})".format(round(total_roc_score,4), args.score_iter))
    print("ap_score: {} (iter: {})".format(round(total_ap_score,4), args.score_iter))

    if args.add_user_edges:
        total_roc_score_user = sum(total_roc_score_user)/len(total_roc_score_user)
        total_ap_score_user = sum(total_ap_score_user)/len(total_ap_score_user)
        print("add_user - roc_score: {} (iter: {})".format(round(total_roc_score_user,4), args.score_iter))
        print("add_user - ap_score: {} (iter: {})\n".format(round(total_ap_score_user,4), args.score_iter))

    os.system("rm /tmp/tmp*")
