__author__ = 'Wei Xie'
__email__ = 'linegroup3@gmail.com'
__affiliation__ = 'Living Analytics Research Centre, Singapore Management University'
__website__ = 'http://mysmu.edu/phdis2012/wei.xie.2012'



import numpy as np
import tensorflow as tf
import time
import exp_config

'''
A simple implementation of factoid embedding which requires less parameter updates.
'''


def embed(triplets, attribute_embeddings, testing_ids=None, bias=False):

    # parameters
    BATCH_SIZE = eval(exp_config.get('triplet_embedding', 'batch_size'))
    USER_DIM = eval(exp_config.get('triplet_embedding', 'user_dim'))
    NCE_SAM_NUM = eval(exp_config.get('triplet_embedding', 'nce_sampling'))
    SNAPSHOT_FLAG = eval(exp_config.get('triplet_embedding', 'snapshot'))
    SNAPSHOT_GAP = eval(exp_config.get('triplet_embedding', 'snapshot_gap'))
    LEARNING_RATE_FOLLOW = eval(exp_config.get('triplet_embedding', 'learning_rate_f'))
    LEARNING_RATE_ATTRIBUTE = eval(exp_config.get('triplet_embedding', 'learning_rate_a'))
    DEBUG_FLAG = eval(exp_config.get('debug', 'flag'))

    # process triplets

    net_degrees = dict()
    max_user_id = 0
    triplets_attribute = list()
    triplets_follow = list()
    predicates = set()
    for trip in triplets:
        s_, p_, o_ = trip
        predicates.add(p_)
        max_user_id = max(max_user_id, s_)

        if p_ == 'a':
            triplets_attribute.append((s_, o_))
        if p_ == 'f':
            triplets_follow.append((s_, o_))
            max_user_id = max(max_user_id, o_)
            if s_ in net_degrees:
                net_degrees[s_] += 1
            else:
                net_degrees[s_] = 1

    num_users = max_user_id + 1
    print 'num_users', num_users
    print 'predicates', predicates

    triplets_attribute = np.array(triplets_attribute, dtype=np.int32)
    triplets_follow = np.array(triplets_follow, dtype=np.int32)

    net_probs = map(lambda x: net_degrees[x] ** 0.75 if x in net_degrees else 0., xrange(num_users))
    net_probs /= np.sum(net_probs)

    # build model

    graph = tf.Graph()

    with graph.as_default():

        with tf.device('/gpu:0'):

            trip_attribute_subject = tf.placeholder(tf.int32, shape=[BATCH_SIZE, 2])
            trip_attribute_object = tf.placeholder(tf.int32, shape=[BATCH_SIZE])

            trip_follow_source = tf.placeholder(tf.int32, shape=[BATCH_SIZE, 1])
            trip_follow_target = tf.placeholder(tf.int32, shape=[BATCH_SIZE])

            user_embeddings = tf.Variable(tf.truncated_normal([num_users, USER_DIM], stddev=1e-2))

            if testing_ids is not None:
                normalized_user_embeddings = tf.nn.l2_normalize(user_embeddings, dim=1)
                testing_ids_ph = tf.placeholder(tf.int32, shape=[len(testing_ids)])
                dist = tf.tensordot(tf.nn.embedding_lookup(normalized_user_embeddings, testing_ids_ph), normalized_user_embeddings, axes=[[1], [1]])

            P_follow = tf.Variable(tf.truncated_normal([USER_DIM, USER_DIM], stddev=1.0 / USER_DIM))
            P_follow_norm = tf.norm(P_follow)

            bias_var = tf.Variable(tf.zeros(USER_DIM))
            bias_norm = tf.norm(bias_var)

            f_loss = 0
            a_loss = 0

            for p_ in predicates:
                if p_ == 'f':
                    target = tf.tensordot(tf.nn.embedding_lookup(user_embeddings, trip_follow_target), P_follow, [[1], [1]])
                    if bias:
                        target += bias_var

                    f_loss += tf.reduce_mean(tf.nn.nce_loss(user_embeddings, tf.zeros(num_users), trip_follow_source,
                                                            target, NCE_SAM_NUM, num_users, num_true=1,
                                                            sampled_values=(np.random.choice(num_users, NCE_SAM_NUM, False, net_probs),
                                                                            tf.ones(BATCH_SIZE, dtype=tf.float32),
                                                                            tf.ones(NCE_SAM_NUM, dtype=tf.float32))
                                                            ))

                if p_ == 'a':
                    attribute_embeddings = tf.nn.embedding_lookup(attribute_embeddings, trip_attribute_object)

                    dot = tf.tensordot(user_embeddings, attribute_embeddings, [[1], [1]])
                    softm = tf.nn.softmax(dot, dim=0)
                    softm = tf.gather_nd(softm, trip_attribute_subject)
                    a_loss -= tf.reduce_mean(tf.log(softm))

            f_global_step = tf.Variable(0, trainable=False)
            a_global_step = tf.Variable(0, trainable=False)
            f_learning_rate = tf.train.exponential_decay(LEARNING_RATE_FOLLOW, f_global_step, 10000, 0.96, staircase=True)
            a_learning_rate = tf.train.exponential_decay(LEARNING_RATE_ATTRIBUTE, a_global_step, 10000, 0.96, staircase=True)
            if f_loss != 0:
                f_optimizer = tf.train.GradientDescentOptimizer(f_learning_rate).minimize(f_loss, var_list=[user_embeddings])
                f_optimizer_P = tf.train.GradientDescentOptimizer(1e-4).minimize(f_loss, var_list=[P_follow, bias_var])
            if a_loss != 0:
                a_optimizer = tf.train.GradientDescentOptimizer(a_learning_rate).minimize(a_loss)

    with tf.Session(graph=graph, config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)) as session:

        f_losses = list()
        a_losses = list()
        P_norms = list()
        bias_norms = list()

        tf.global_variables_initializer().run()

        warm_up_iter = eval(exp_config.get('triplet_embedding', 'warm_up_iter')) * 1000
        n_iter = eval(exp_config.get('triplet_embedding', 'n_iter')) * 1000

        n_triplets_follow = len(triplets_follow)
        n_triplets_attribute = len(triplets_attribute)
        follow_batch_start_id = 0
        follow_permutation = np.random.permutation(n_triplets_follow)
        attribute_batch_start_id = 0
        attribute_permutation = np.random.permutation(n_triplets_attribute)

        ue_s = list()
        dist_s = list()

        for i in xrange(n_iter):
            if SNAPSHOT_FLAG and i % (SNAPSHOT_GAP*1000) == 0:
                ue = user_embeddings.eval(session)
                dist_values = None if testing_ids is None else session.run(dist, feed_dict={testing_ids_ph: testing_ids})
                ue_s.append(ue)
                dist_s.append(dist_values)

            if a_loss != 0:
                start_time = time.time()
                if attribute_batch_start_id + BATCH_SIZE > n_triplets_attribute:
                    attribute_batch_start_id = 0
                    attribute_permutation = np.random.permutation(n_triplets_attribute)
                choice = attribute_permutation[attribute_batch_start_id: attribute_batch_start_id + BATCH_SIZE]
                selected_triplets_attribute = triplets_attribute[choice, :]

                attribute_s = selected_triplets_attribute[:, 0:1]
                attribute_s = np.concatenate((attribute_s, np.arange(BATCH_SIZE).reshape(BATCH_SIZE, 1)), axis=1)
                attribute_o = selected_triplets_attribute[:, 1]

                attribute_batch_start_id += BATCH_SIZE

                _, a_loss_val = session.run([a_optimizer, a_loss],
                                             feed_dict={trip_attribute_subject: attribute_s, trip_attribute_object: attribute_o})
                a_losses.append(a_loss_val)
                if DEBUG_FLAG and np.random.rand() < 0.1:
                    print i, 'a_loss_val', a_loss_val, time.time() - start_time

            if i >= warm_up_iter and f_loss != 0:
                # follow
                start_time = time.time()
                if follow_batch_start_id + BATCH_SIZE > n_triplets_follow:
                    follow_batch_start_id = 0
                    follow_permutation = np.random.permutation(n_triplets_follow)
                choice = follow_permutation[follow_batch_start_id: follow_batch_start_id + BATCH_SIZE]
                selected_triplets_follow = triplets_follow[choice, :]
                _source, _target = np.expand_dims(selected_triplets_follow[:, 0], axis=1), selected_triplets_follow[:, 1]
                follow_batch_start_id += BATCH_SIZE
                if i % 5 == 0:
                    _, f_loss_val, P_norm_, bias_norm_ = session.run([f_optimizer_P, f_loss, P_follow_norm, bias_norm],
                                        feed_dict={trip_follow_source: _source, trip_follow_target: _target})
                    P_norms.append(P_norm_)
                    bias_norms.append(bias_norm_)
                else:
                    _, f_loss_val = session.run([f_optimizer, f_loss],
                                        feed_dict={trip_follow_source: _source, trip_follow_target: _target})

                f_losses.append(f_loss_val)
                if DEBUG_FLAG and np.random.rand() < 0.1:
                    print i, 'f_loss_val', f_loss_val, time.time() - start_time

        ue = user_embeddings.eval(session)
        dist_values = None if testing_ids is None else session.run(dist, feed_dict={testing_ids_ph: testing_ids})
        ue_s.append(ue)
        dist_s.append(dist_values)

    return ue_s, dist_s, (f_losses, a_losses, (P_norms, bias_norms))
