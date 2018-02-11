class PrecisionAtKEval(Callback):
    def __init__(self, g, true_edges, excluded_edges, decoder, ks):
        """
        ks = [2, 10, 100, 200, 300, 500, 800, 1000, 10000]
        """
        self.ks = ks
        self.maps = []
        self.decoder = decoder

        N = g.number_of_nodes()
        self.nodes = np.arange(N)[:, None]
        self.edges_to_eval = set([(i, j) for i in range(N) for j in range(i+1, N)])

        # we don't consider train edges and excluded_edges
        # only consider dev edges and other ones
        excluded_edges = [tuple(x) for x in excluded_edges.tolist()]
        true_edges = [tuple(x) for x in true_edges.tolist()]
        self.edges_to_eval -= set(g.edges()) | set(excluded_edges)
        true_edges = set(true_edges)

        # edges in true_edges are labeled 1
        # othe edges labeled 0
        # edge2label = {e: (e in true_edges)
        #               for e in self.edges_to_eval}
        self.true_y = np.array([(e in true_edges)
                                for e in self.edges_to_eval])

    def eval_map(self):
        # to enable calling this function outside of `keras.Model`
        reconstructed_adj = self.decoder.predict(self.nodes)

        pred_y = np.array([reconstructed_adj[u, v]
                           for u, v in self.edges_to_eval])

        sort_idx = np.argsort(pred_y)[::-1]  # index ranked by score from high to low

        return [precision_at_k(pred_y, self.true_y, k=k, sort_idx=sort_idx)
                for k in self.ks]

    def on_epoch_end(self, epoch, logs={}):
        row = self.eval_map()
        print('EPOCH {}'.format(epoch))
        print('[DEV] precision at k' + ' '.join(["{}:{}".format(k, r) for k, r in zip(self.ks, row)]))
        self.maps.append(row)