from __future__ import division
import numpy as np
import sys, os


class Predict:
    def __init__(self, vocabfile="../model/model_vocab.txt", modelfile="../model/model_count_table.txt"):
        self.vocabfile = vocabfile
        self.modelfile = modelfile

        self.K = 0
        self.wordprobs = None
        self.w2num = None
        self.N_wk = np.loadtxt(self.modelfile)
        self.N_w = self.N_wk.sum(1)
        self.N_k = self.N_wk.sum(0)
        self.K = len(self.N_k)

    def load_model(self):
        """Idempotent"""
        if self.wordprobs is not None:
            # assume already loaded
            return

        self.wordprobs = (self.N_wk + 1) / self.N_k

        self.vocab = [L.split("\t")[-1].strip() for L in open(self.vocabfile, encoding='utf8')]
        self.w2num = {w: i for i, w in enumerate(self.vocab)}
        assert len(self.vocab) == self.N_wk.shape[0]


    def infer_cvb0(self, invocab_tokens, alpha, numpasses):
        # global K, wordprobs, w2num
        doclen = len(invocab_tokens)

        # initialize with likelihoods
        Qs = np.zeros((doclen, self.K))
        for i in range(doclen):
            w = invocab_tokens[i]
            Qs[i, :] = self.wordprobs[self.w2num[w], :]
            Qs[i, :] /= Qs[i, :].sum()
        lik = Qs.copy()  # pertoken normalized but proportionally the same for inference

        Q_k = Qs.sum(0)
        for itr in range(1, numpasses):
            # print "cvb0 iter", itr
            for i in range(doclen):
                Q_k -= Qs[i, :]
                Qs[i, :] = lik[i, :] * (Q_k + alpha)
                Qs[i, :] /= Qs[i, :].sum()
                Q_k += Qs[i, :]

        Q_k /= Q_k.sum()
        return Q_k

    def predict(self, tokens, alpha=1, numpasses=5, thresh1=1, thresh2=0.2):
        if len(tokens) > 0:
            assert isinstance(tokens[0], str)
        invocab_tokens = [w.lower() for w in tokens if w.lower() in self.w2num]
        # check that at least xx tokens are in vocabulary
        if len(invocab_tokens) < thresh1:
            return None
            # check that at least yy% of tokens are in vocabulary
        elif len(invocab_tokens) / len(tokens) < thresh2:
            return None
        else:
            posterior = self.infer_cvb0(invocab_tokens, alpha=alpha, numpasses=numpasses)
            return posterior
