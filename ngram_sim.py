__author__ = 'Wei Xie'
__email__ = 'linegroup3@gmail.com'
__affiliation__ = 'Living Analytics Research Centre, Singapore Management University'
__website__ = 'http://mysmu.edu/phdis2012/wei.xie.2012'


import sys, time
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import itertools
import jellyfish


max_ngram_len = 5
indexed_ngram_len = 3

inverted_index = dict()
_X = None  # tf-idf vectors


def get_ngram(name, length):
    ret = list()

    for i in xrange(len(name) - length + 1):
        token = name[i: i+length]
        ret.append(token)

    return ret


def get_all_ngrams(name, _id):
    ret = list()

    for length in xrange(1, max_ngram_len + 1):
        ngrams = get_ngram(name, length)
        ret.append(ngrams)

        if length == indexed_ngram_len:
            for ngram in ngrams:
                if ngram in inverted_index:
                    inverted_index[ngram].add(_id)
                else:
                    inverted_index[ngram] = set([_id])

    ret = list(itertools.chain.from_iterable(ret))
    return ' '.join(ret)


def process_unit_tfidf(i, name):
    global _X

    if i % 100 == 0:
        sys.stdout.write("\r%d%%" % (100*i/_X.shape[0]))
        sys.stdout.flush()

    ngrams = get_ngram(name, indexed_ngram_len)

    candidate_set = set()
    for ngram in ngrams:
        if ngram in inverted_index:
            candidate_set |= inverted_index[ngram]

    candidates = list(candidate_set)

    row = [i] * len(candidates)
    col = candidates
    if len(candidates) > 0:
        data = np.power(linear_kernel(_X[i:i+1], _X[candidates]).flatten(), 0.25)
    else:
        data = []

    return row, col, data


def process_unit_jw(i, names):
    name = names[i]

    if i % 100 == 0:
        sys.stdout.write("\r%d%%" % (100*i/len(names)))
        sys.stdout.flush()

    ngrams = get_ngram(name, indexed_ngram_len)

    candidate_set = set()
    for ngram in ngrams:
        if ngram in inverted_index:
            candidate_set |= inverted_index[ngram]

    candidates = list(candidate_set)

    col = list()
    data = list()
    for x in candidates:
        sim = 2*jellyfish.jaro_winkler(name, names[x]) - 1
        if sim != 0.:
            col.append(x)
            data.append(sim)
    row = [i] * len(col)

    return row, col, data


def name2sim_tfidf(names):
    '''
    transform a list of names to a similarity matrix, based on their n-gram tf-idf vectors
    :param names:
    :return: a sparse matrix (csr_matrix)
    '''
    global _X

    print 'Maximum N-gram length', max_ngram_len

    vectorizer = TfidfVectorizer(min_df=1, stop_words=None, analyzer='word', tokenizer=lambda x: x.split())

    corpus = list()

    for _id, name in enumerate(names):
        corpus.append(get_all_ngrams(name, _id))

    _X = vectorizer.fit_transform(corpus)

    print type(_X), _X.shape

    print vectorizer.get_feature_names()[:50]

    ct = time.time()

    ret = [process_unit_tfidf(i, name) for i, name in enumerate(names)]

    data_list = list()
    row_list = list()
    col_list = list()
    for element in ret:
        row, col, data = element
        row_list.append(row)
        col_list.append(col)
        data_list.append(data)

    row_list = list(itertools.chain(*row_list))
    col_list = list(itertools.chain(*col_list))
    data_list = list(itertools.chain(*data_list))

    print 'Sparse matrix X constructed, use ', (time.time() - ct), 's'

    return csr_matrix((data_list, (row_list, col_list)), shape=(len(names), len(names)))


def name2sim_jw(names):
    '''
    transform a list of names to a similarity matrix, based on their jaro_winkler similarities
    :param names:
    :return: a sparse matrix (csr_matrix)
    '''

    corpus = list()

    for _id, name in enumerate(names):
        corpus.append(get_all_ngrams(name, _id))

    ct = time.time()

    ret = [process_unit_jw(i, names) for i, name in enumerate(names)]

    data_list = list()
    row_list = list()
    col_list = list()
    for element in ret:
        row, col, data = element
        row_list.append(row)
        col_list.append(col)
        data_list.append(data)

    row_list = list(itertools.chain(*row_list))
    col_list = list(itertools.chain(*col_list))
    data_list = list(itertools.chain(*data_list))

    print 'Sparse matrix X constructed, use ', (time.time() - ct), 's'

    return csr_matrix((data_list, (row_list, col_list)), shape=(len(names), len(names)))