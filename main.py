__author__ = 'Wei Xie'
__email__ = 'linegroup3@gmail.com'
__affiliation__ = 'Living Analytics Research Centre, Singapore Management University'
__website__ = 'http://mysmu.edu/phdis2012/wei.xie.2012'


'''
In this version, only one attribute embedding.
'''


import codecs

import os

import sys

import time

import jellyfish

import numpy as np

import exp_config

settings_f = './settings.cnf'

print 'settings:', settings_f

exp_config.read(settings_f)

import name2sim
import image2sim
import factoid_embedding

import argparse

parser = argparse.ArgumentParser(description='Embedding users from two different social platforms into one hidden space.')

parser.add_argument('--path', type=str, nargs=1, help='data path')

parser.add_argument('--skip_network', type=str, nargs=1, help='do not use network information.')

parser.add_argument('--source_prefix', type=str, nargs=1, help='the prefix of source platform.')
parser.add_argument('--target_prefix', type=str, nargs=1, help='the prefix of target platform.')
parser.add_argument('--source_col', type=str, nargs=1, help='the source column in the ground truth.')
parser.add_argument('--target_col', type=str, nargs=1, help='the target column in the ground truth.')

parser.add_argument('--name_dim', type=str, nargs=1, help='the dimension of user name embedding.')
parser.add_argument('--name_concatenate', type=str, nargs=1, help='indicate whether concatenate user name and screen name.')
parser.add_argument('--name_preprocess', type=str, nargs=1, help='indicate whether perform pre-process on names.')
parser.add_argument('--name_method', type=str, nargs=1, help='method of calculating name similarities, e.g. jaro_winkler, tfidf.')
parser.add_argument('--screen_name_exist', type=str, nargs=1, help='indicate whether screen name exists.')

parser.add_argument('--image_exist', type=str, nargs=1, help='indicate whether user profile images exist.')
parser.add_argument('--image_method', type=str, nargs=1, help='method of calculating image similarities, e.g. identical.')
parser.add_argument('--image_identical_threshold', type=str, nargs=1, help='threshold of detecting identical images.')
parser.add_argument('--image_dim', type=str, nargs=1, help='the dimension of image embedding.')

parser.add_argument('--cosine_embedding_batch_size', type=str, nargs=1, help='batch size for cosine embedding.')
parser.add_argument('--cosine_embedding_learning_rate', type=str, nargs=1, help='learning rate for cosine embedding.')

parser.add_argument('--supervised', type=str, nargs=1, help='indicate whether it is supervised learning.')
parser.add_argument('--snapshot', type=str, nargs=1, help='save snapshot or not.')
parser.add_argument('--snapshot_gap', type=str, nargs=1, help='save snapshot for each #snapshot_gap iterations.')
parser.add_argument('--n_iter', type=str, nargs=1, help='learning iterations.')
parser.add_argument('--warm_up_iter', type=str, nargs=1, help='warming up iterations.')
parser.add_argument('--user_dim', type=str, nargs=1, help='the dimension of user embedding.')
parser.add_argument('--nce_sampling', type=str, nargs=1, help='number of negative sampling.')
parser.add_argument('--triplet_embedding_batch_size', type=str, nargs=1, help='batch size for triplet embedding.')
parser.add_argument('--triplet_embedding_learning_rate_f', type=str, nargs=1, help='learning rate for follow triplet.')
parser.add_argument('--triplet_embedding_learning_rate_a', type=str, nargs=1, help='learning rate for attribute triplet.')

parser.add_argument('--stratified_attribute', type=str, nargs=1, help='stratified attribute.')

args = parser.parse_args()

if args.path is not None:
    exp_config.set('data', 'path', args.path[0])

flag_skip_network = False
if args.skip_network is not None:
    flag_skip_network = eval(args.skip_network[0])

if args.source_prefix is not None:
    exp_config.set('data', 'source_prefix', args.source_prefix[0])
if args.target_prefix is not None:
    exp_config.set('data', 'target_prefix', args.target_prefix[0])
if args.source_col is not None:
    exp_config.set('data', 'source_col', args.source_col[0])
if args.target_col is not None:
    exp_config.set('data', 'target_col', args.target_col[0])

if args.name_dim is not None:
    exp_config.set('predicate_name', 'name_dim', args.name_dim[0])
if args.name_preprocess is not None:
    exp_config.set('predicate_name', 'preprocess', args.name_preprocess[0])
if args.name_method is not None:
    exp_config.set('predicate_name', 'method', args.name_method[0])
if args.name_concatenate is not None:
    exp_config.set('predicate_name', 'concatenate', args.name_concatenate[0])
if args.screen_name_exist is not None:
    exp_config.set('predicate_name', 'screen_name_exist', args.screen_name_exist[0])


if args.image_exist is not None:
    exp_config.set('predicate_image', 'exist', args.image_exist[0])
if args.image_method is not None:
    exp_config.set('predicate_image', 'method', args.image_method[0])
if args.image_identical_threshold is not None:
    exp_config.set('predicate_image', 'identical_threshold', args.image_identical_threshold[0])
if args.image_dim is not None:
    exp_config.set('predicate_image', 'dim', args.image_dim[0])


if args.cosine_embedding_batch_size is not None:
    exp_config.set('cosine_embedding', 'batch_size', args.cosine_embedding_batch_size[0])
if args.cosine_embedding_learning_rate is not None:
    exp_config.set('cosine_embedding', 'learning_rate', args.cosine_embedding_learning_rate[0])

if args.supervised is not None:
    exp_config.set('triplet_embedding', 'supervised', args.supervised[0])
if args.snapshot is not None:
    exp_config.set('triplet_embedding', 'snapshot', args.snapshot[0])
if args.snapshot_gap is not None:
    exp_config.set('triplet_embedding', 'snapshot_gap', args.snapshot_gap[0])
if args.n_iter is not None:
    exp_config.set('triplet_embedding', 'n_iter', args.n_iter[0])
if args.warm_up_iter is not None:
    exp_config.set('triplet_embedding', 'warm_up_iter', args.warm_up_iter[0])
if args.user_dim is not None:
    exp_config.set('triplet_embedding', 'user_dim', args.user_dim[0])
if args.nce_sampling is not None:
    exp_config.set('triplet_embedding', 'nce_sampling', args.nce_sampling[0])
if args.triplet_embedding_batch_size is not None:
    exp_config.set('triplet_embedding', 'batch_size', args.triplet_embedding_batch_size[0])
if args.triplet_embedding_learning_rate_f is not None:
    exp_config.set('triplet_embedding', 'learning_rate_f', args.triplet_embedding_learning_rate_f[0])
if args.triplet_embedding_learning_rate_a is not None:
    exp_config.set('triplet_embedding', 'learning_rate_a', args.triplet_embedding_learning_rate_a[0])

if args.stratified_attribute is not None:
    exp_config.set('evaluation', 'stratified_attribute', args.stratified_attribute[0])

PATH = exp_config.get('data', 'path')

SOURCE_PREFIX = exp_config.get('data', 'source_prefix')
TARGET_PREFIX = exp_config.get('data', 'target_prefix')
SOURCE_COL = int(exp_config.get('data', 'source_col'))
TARGET_COL = int(exp_config.get('data', 'target_col'))

USER_DIM = exp_config.get('triplet_embedding', 'user_dim')
ITER_NUM = exp_config.get('triplet_embedding', 'n_iter')
WARM_UP_NUM = exp_config.get('triplet_embedding', 'warm_up_iter')
NCE_NUM = exp_config.get('triplet_embedding', 'nce_sampling')

SUPERVISED_FLAG = eval(exp_config.get('triplet_embedding', 'supervised'))
BIAS_FLAG = eval(exp_config.get('triplet_embedding', 'bias'))


screen_name_exist = eval(exp_config.get('predicate_name', 'screen_name_exist'))
flag_preprocess = eval(exp_config.get('predicate_name', 'preprocess'))
flag_concatenate = eval(exp_config.get('predicate_name', 'concatenate'))
name_dim = eval(exp_config.get('predicate_name', 'name_dim'))

image_exist = eval(exp_config.get('predicate_image', 'exist'))
image_dim = eval(exp_config.get('predicate_image', 'dim'))

OUTPUT = exp_config.get('data', 'output')
if OUTPUT is None:
    OUTPUT = 'results'
    if SUPERVISED_FLAG:
        OUTPUT += '_supervised'
    else:
        OUTPUT += '_unsupervised'
    if BIAS_FLAG:
        OUTPUT += '_biased'
    else:
        OUTPUT += '_unbiased'
    OUTPUT += '_d' + USER_DIM
    OUTPUT += '_i' + ITER_NUM
    OUTPUT += '_w' + WARM_UP_NUM
    OUTPUT += '_ns' + NCE_NUM

    if image_exist:
        OUTPUT += '_m' + str(image_dim)

    OUTPUT += '_' + exp_config.get('predicate_name', 'method')

    if screen_name_exist:
        OUTPUT += '_s'

    if image_exist:
        OUTPUT += '_m'

OUTPUT = SOURCE_PREFIX + '2' + TARGET_PREFIX + '_' + OUTPUT
OUTPUT += '/'


def preprocess_fun(s):
    return s.encode('ascii', 'ignore').decode('utf-8').replace('.', '').replace('_', '').replace(' ', '')


source_users = list()
target_users = list()

source_user_names = dict()

f = codecs.open(PATH + SOURCE_PREFIX + '_user_names.txt', 'r', 'utf-8')
for line in f:
    terms = line[:-1].split('\t')
    if terms[1] == 'None':
        un = None
    else:
        un = terms[1].lower().strip()
        if flag_preprocess:
            un = preprocess_fun(un)
            if len(un) == 0:
                un = None
    source_user_names[terms[0]] = un
    source_users.append(terms[0])
f.close()

print 'source_user_names', len(source_user_names)

source_screen_names = None
if screen_name_exist:
    source_screen_names = dict()

    f = codecs.open(PATH + SOURCE_PREFIX + '_screen_names.txt', 'r', 'utf-8')
    for line in f:
        terms = line[:-1].split('\t')
        if terms[1] == 'None':
            sn = None
        else:
            sn = terms[1].lower().strip()
            if flag_preprocess:
                sn = preprocess_fun(sn)
                if len(sn) == 0:
                    sn = None
        source_screen_names[terms[0]] = sn
    f.close()

    print 'source_screen_names', len(source_screen_names)

source_images = None
if image_exist:
    source_images = dict()

    for source_user in source_users:
        if os.path.isfile(PATH + 'images/' + SOURCE_PREFIX + '/' + source_user + '.jpg'):
            source_images[source_user] = 'images/' + SOURCE_PREFIX + '/' + source_user + '.jpg'

    print 'source_images', len(source_images)

target_user_names = dict()

f = codecs.open(PATH + TARGET_PREFIX + '_user_names.txt', 'r', 'utf-8')
for line in f:
    terms = line[:-1].split('\t')
    if terms[1] == 'None':
        un = None
    else:
        un = terms[1].lower().strip()
        if flag_preprocess:
            un = preprocess_fun(un)
            if len(un) == 0:
                un = None
    target_user_names[terms[0]] = un
    target_users.append(terms[0])
f.close()

print 'target_user_names', len(target_user_names)

target_screen_names = None
if screen_name_exist:
    target_screen_names = dict()

    f = codecs.open(PATH + TARGET_PREFIX + '_screen_names.txt', 'r', 'utf-8')
    for line in f:
        terms = line[:-1].split('\t')
        if terms[1] == 'None':
            sn = None
        else:
            sn = terms[1].lower().strip()
            if flag_preprocess:
                sn = preprocess_fun(sn)
                if len(sn) == 0:
                    sn = None
        target_screen_names[terms[0]] = sn
    f.close()

    print 'target_screen_names', len(target_screen_names)

target_images = None
if image_exist:
    target_images = dict()

    for target_user in target_users:
        if os.path.isfile(PATH + 'images/' + TARGET_PREFIX + '/' + target_user + '.jpg'):
            target_images[target_user] = 'images/' + TARGET_PREFIX + '/' + target_user + '.jpg'

    print 'target_images', len(target_images)

print 'source_users', len(source_users), 'target_users', len(target_users)


if not flag_concatenate:
    # for user names
    user_name2eid, _, user_name_embeddings = name2sim.name2sim(
                                        source_user_names.values() + target_user_names.values(), name_dim, 'user_name')

    # for screen names
    if screen_name_exist:
        screen_name2eid, _, screen_name_embeddings = name2sim.name2sim(
                                source_screen_names.values() + target_screen_names.values(), name_dim, 'screen_name')
else:
    source_concat_names = dict()
    for uid in source_users:
        cn = ''
        if source_user_names[uid]:
            cn += source_user_names[uid]
        if screen_name_exist:
            if source_screen_names[uid]:
                cn += source_screen_names[uid]
        if len(cn) == 0:
            cn = None
        source_concat_names[uid] = cn

    target_concat_names = dict()
    for uid in target_users:
        cn = ''
        if target_user_names[uid]:
            cn += target_user_names[uid]
        if screen_name_exist:
            if target_screen_names[uid]:
                cn += target_screen_names[uid]
        if len(cn) == 0:
            cn = None
        target_concat_names[uid] = cn

    concat_name2eid, _, concat_name_embeddings = name2sim.name2sim(
                                source_concat_names.values() + target_concat_names.values(), 2*name_dim, 'concat_name')

if image_exist:
    image2eid, _, image_embeddings = image2sim.image2sim(
                                        source_images.values() + target_images.values())


def unit_vector(vector):
    s = np.linalg.norm(vector)
    if s > 1e-8:
        return vector / s
    else:
        return vector


def sim(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.dot(v1_u, v2_u)


if not os.path.isdir(PATH + OUTPUT):
    os.makedirs(PATH + OUTPUT)

# training and testing
stratified_attribute = exp_config.get('evaluation', 'stratified_attribute')
TRAIN_TEST_PATH = eval(exp_config.get('data', 'train_test_paths'))

# users in ground truth
if not SUPERVISED_FLAG:
    source_users_in_gt = list()
    f = open(PATH + '/ground_truth.txt', 'r')
    for line in f:
        terms = line.split()
        sc_id = terms[SOURCE_COL]
        source_users_in_gt.append(sc_id)
    f.close()

all_res = list()
name_dis_list = list()
for tt_path in TRAIN_TEST_PATH:
    print tt_path

    source_users_in_training = list()
    target_users_in_training = list()

    training_map = dict()
    f = open(PATH + tt_path + '/training.txt', 'r')
    for line in f:
        terms = line.split()
        sc_id = terms[SOURCE_COL]
        tg_id = terms[TARGET_COL]
        training_map[sc_id] = tg_id
        source_users_in_training.append(sc_id)
        target_users_in_training.append(tg_id)
    f.close()

    source_users_in_testing = list()
    target_users_in_testing = list()
    testing_map = dict()
    f = open(PATH + tt_path + '/testing.txt', 'r')
    for line in f:
        terms = line.split()
        sc_id = terms[SOURCE_COL]
        tg_id = terms[TARGET_COL]
        testing_map[sc_id] = tg_id
        source_users_in_testing.append(sc_id)
        target_users_in_testing.append(tg_id)
    f.close()

    # constructing triples
    triplets = list()
    attribute_embeddings = list()
    attribute_id = 0

    sc2uid = dict()
    tg2uid = dict()

    uid = 0
    # for matched users
    if SUPERVISED_FLAG:
        for source_id, target_id in training_map.iteritems():
            sc2uid[source_id] = uid
            tg2uid[target_id] = uid
            uid += 1

    unmatched_uid = uid
    print 'unmatched_uid', unmatched_uid

    for source_id in source_users:
        if SUPERVISED_FLAG:
            if source_id in source_users_in_training:
                continue

        if flag_concatenate:
            attribute_embedding = concat_name_embeddings[concat_name2eid[source_concat_names[source_id]]]
        else:
            attribute_embedding = user_name_embeddings[user_name2eid[source_user_names[source_id]]]
            if screen_name_exist:
                attribute_embedding = np.concatenate([attribute_embedding,
                                            screen_name_embeddings[screen_name2eid[source_screen_names[source_id]]]])

        if image_exist:
            image = source_images[source_id] if source_id in source_images else None
            eid = image2eid[image] if image in image2eid else -1
            attribute_embedding = np.concatenate([attribute_embedding, 0.25*image_embeddings[eid]])

        attribute_embeddings.append(attribute_embedding)

        sc2uid[source_id] = uid
        triplets.append((uid, 'a', attribute_id))

        attribute_id += 1
        uid += 1

    for target_id in target_users:
        if SUPERVISED_FLAG:
            if target_id in target_users_in_training:
                continue

        if flag_concatenate:
            attribute_embedding = concat_name_embeddings[concat_name2eid[target_concat_names[target_id]]]
        else:
            attribute_embedding = user_name_embeddings[user_name2eid[target_user_names[target_id]]]
            if screen_name_exist:
                attribute_embedding = np.concatenate([attribute_embedding,
                                            screen_name_embeddings[screen_name2eid[target_screen_names[target_id]]]])

        if image_exist:
            image = target_images[target_id] if target_id in target_images else None
            eid = image2eid[image] if image in image2eid else -1
            attribute_embedding = np.concatenate([attribute_embedding, 0.25*image_embeddings[eid]])

        attribute_embeddings.append(attribute_embedding)

        tg2uid[target_id] = uid
        triplets.append((uid, 'a', attribute_id))

        attribute_id += 1
        uid += 1

    attribute_embeddings = np.array(attribute_embeddings)
    print 'uid', uid, 'tg2uid', len(tg2uid), 'sc2uid', len(sc2uid), 'attribute_embeddings', len(attribute_embeddings), attribute_embeddings.shape

    # for source_links
    f = open(PATH + SOURCE_PREFIX + '_sub_network.txt', 'r')
    for line in f:
        source_id1, source_id2 = line.split()
        if source_id1 in sc2uid and source_id2 in sc2uid:
            uid1 = sc2uid[source_id1]
            uid2 = sc2uid[source_id2]
            triplets.append((uid1, 'f', uid2))
    f.close()

    # for target_links
    f = open(PATH + TARGET_PREFIX + '_sub_network.txt', 'r')
    for line in f:
        target_id1, target_id2 = line[:-1].split('\t')
        if target_id1 in tg2uid and target_id2 in tg2uid:
            uid1 = tg2uid[target_id1]
            uid2 = tg2uid[target_id2]
            triplets.append((uid1, 'f', uid2))
    f.close()

    print 'triplets', len(triplets)

    # triplet embedding
    if not SUPERVISED_FLAG:
        tt_path = 'without_training_data_'

    testing_ids = None
    if SUPERVISED_FLAG:
        testing_ids = list()
        for source_user in source_users_in_testing:
            testing_ids.append(sc2uid[source_user])
        testing_ids = np.array(testing_ids, dtype=np.int32)
    else:
        testing_ids = list()
        for source_user in source_users_in_gt:
            testing_ids.append(sc2uid[source_user])
        testing_ids = np.array(testing_ids, dtype=np.int32)

    if flag_skip_network:
        ue_s, dist_s = factoid_embedding.attribute_embed(triplets, attribute_embeddings, testing_ids)#for debugging
    else:
        if os.path.isfile(PATH + OUTPUT + tt_path + 'user_embedding_result.npy'):
            ue_s, dist_s, (f_losses, usn_losses, P_norms) = np.load(PATH + OUTPUT + tt_path + 'user_embedding_result.npy')
            print PATH + OUTPUT + tt_path + 'user_embedding_result.npy', 'loaded'
        else:
            ue_s, dist_s, (f_losses, usn_losses, P_norms) = factoid_embedding.embed(triplets, attribute_embeddings,
                                                                                                 testing_ids, BIAS_FLAG)
            np.save(PATH + OUTPUT + tt_path + 'user_embedding_result.npy',
                    [ue_s, dist_s, (f_losses, usn_losses, P_norms)])

    target_users_in_training = set(target_users_in_training)
    start_time = time.time()
    _id = 0
    for source_user in source_users_in_testing:
        snapshot_res = list()
        if not SUPERVISED_FLAG:
            source_user_id_in_gt = source_users_in_gt.index(source_user)
        for snapshot_i in xrange(len(ue_s)):
            testing_data = list()
            target_candidates = list()
            ue = ue_s[snapshot_i]
            dist = dist_s[snapshot_i]
            for target_user in target_users:
                if target_user in target_users_in_training:
                    continue

                target_candidates.append(target_user)
                dis = dist[_id, tg2uid[target_user]] if SUPERVISED_FLAG else dist[source_user_id_in_gt, tg2uid[target_user]]#sim(ue[sc2uid[source_user]], ue[tg2uid[target_user]])
                testing_data.append(dis)

            testing_data = np.array(testing_data)

            predict = testing_data

            scores = zip(predict.tolist(), target_candidates)

            scores.sort(reverse=True)
            _, target_candidates = zip(*scores)

            # for mrr
            mrr_index = target_candidates.index(testing_map[source_user])
            mrr = 1./(mrr_index + 1)

            # for top_list

            top_k = 30
            res = np.zeros(top_k+1)
            res[mrr_index:] = 1
            res[-1] = mrr

            snapshot_res.append(res)

        _id += 1

        sys.stdout.write("\r%d%%" % (100*_id/len(source_users_in_testing)))
        sys.stdout.flush()

        all_res.append(snapshot_res)
        if stratified_attribute == 'screen_name':
            source_name = source_screen_names[source_user]
            target_name = target_screen_names[testing_map[source_user]]
        else:
            source_name = source_user_names[source_user]
            target_name = target_user_names[testing_map[source_user]]

        if source_name is None:
            source_name = u''
        if target_name is None:
            target_name = u''

        name_dis_list.append(jellyfish.jaro_winkler(source_name, target_name))

    print time.time() - start_time, 'seconds used.'

print 'all_res', len(all_res)
all_res = np.array(all_res)
name_dis_list = np.array(name_dis_list)

np.save(PATH + OUTPUT + 'all_res.npy', all_res)
np.save(PATH + OUTPUT + 'name_dis_list.npy', name_dis_list)

precision = np.mean(all_res, axis=0) if len(all_res) > 0 else 0.
print precision

f = open(PATH + OUTPUT + 'report.txt', 'w')
f.write(str(precision))
f.write('\n')
f.write('########################################')
f.write('\n')

f.write(exp_config.get_info())

f.close()













