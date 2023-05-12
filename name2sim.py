__author__ = 'Wei Xie'
__email__ = 'linegroup3@gmail.com'
__affiliation__ = 'Living Analytics Research Centre, Singapore Management University'
__website__ = 'http://mysmu.edu/phdis2012/wei.xie.2012'


import os
import numpy as np
import scipy.sparse
import codecs
import ngram_sim
import cosine_embedding_for_sparse_input
import exp_config


def name2sim(input_names, n_dim, prefix='name', with_embedding=True):
    PATH = exp_config.get('data', 'path')
    method = exp_config.get('predicate_name', 'method')
    embedding_iters = eval(exp_config.get('cosine_embedding', 'n_iter'))

    print 'In name2sim() method ', method, ' embedding_iters ', embedding_iters

    assert method in ['jaro_winkler', 'tfidf']

    if os.path.isfile(PATH + prefix + '_list_' + method + '.txt'):
        names = list()
        fin = codecs.open(PATH + prefix + '_list_' + method + '.txt', 'r', 'utf-8')
        for line in fin:
            names.append(line[:-1])
        fin.close()
    else:
        names = set()
        for name in input_names:
            if name is not None:
                names.add(name)
        names = list(names)
        fout = codecs.open(PATH + prefix + '_list_' + method + '.txt', 'w', 'utf-8')
        for name in names:
            fout.write(name)
            fout.write('\n')
        fout.close()

    sim = None
    if method == 'jaro_winkler':
        if os.path.isfile(PATH + prefix + '_sim_' + method + '.npz'):
            sim = scipy.sparse.load_npz(PATH + prefix + '_sim_' + method + '.npz')
        else:
            sim = ngram_sim.name2sim_jw(names)
            scipy.sparse.save_npz(PATH + prefix + '_sim_' + method + '.npz', sim)

    if method == 'tfidf':
        if os.path.isfile(PATH + prefix + '_sim_' + method + '.npz'):
            sim = scipy.sparse.load_npz(PATH + prefix + '_sim_' + method + '.npz')
        else:
            sim = ngram_sim.name2sim_tfidf(names)
            scipy.sparse.save_npz(PATH + prefix + '_sim_' + method + '.npz', sim)

    if with_embedding:
        if os.path.isfile(PATH + prefix + '_embeddings_' + method + '_' + str(n_dim) + '.npy'):
            embeddings = np.load(PATH + prefix + '_embeddings_' + method + '_' + str(n_dim) + '.npy')
        else:
            embeddings = cosine_embedding_for_sparse_input.embed(sim, n_dim, embedding_iters)
            np.save(PATH + prefix + '_embeddings_' + method + '_' + str(n_dim) + '.npy', embeddings)

        embeddings = np.append(embeddings, np.zeros((1, n_dim), dtype=np.float32), axis=0)
    else:
        embeddings = None

    name2eid = dict(zip(names, range(len(names))))

    name2eid[None] = len(names)

    return name2eid, sim, embeddings
