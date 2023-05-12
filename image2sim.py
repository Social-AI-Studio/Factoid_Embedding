__author__ = 'Wei Xie'
__email__ = 'linegroup3@gmail.com'
__affiliation__ = 'Living Analytics Research Centre, Singapore Management University'
__website__ = 'http://mysmu.edu/phdis2012/wei.xie.2012'


import sys
import os
import time
import numpy as np
import scipy.sparse
from PIL import Image
import imagehash



from scipy.sparse import dok_matrix

import cosine_embedding_for_sparse_input
import exp_config
import lsh


def unit_vector(vector):
    s = np.linalg.norm(vector)
    if s > 1e-8:
        return vector / s
    else:
        return vector


def similarity(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.dot(v1_u, v2_u)


def image2sim(input_images, prefix='image', with_embedding=True):

    PATH = exp_config.get('data', 'path')
    IDENTICAL_T = eval(exp_config.get('predicate_image', 'identical_threshold'))
    method = exp_config.get('predicate_image', 'method')
    n_dim = eval(exp_config.get('predicate_image', 'dim'))
    embedding_iters = eval(exp_config.get('cosine_embedding', 'n_iter'))

    assert method in ['identical', 'vgg16', 'vgg19', 'xception', 'inception_resnet_v2']
    print 'input_images', len(input_images)

    if os.path.isfile(PATH + prefix + '_list' + '.txt'):
        images = list()
        fin = open(PATH + prefix + '_list' + '.txt', 'r')
        for line in fin:
            images.append(line[:-1])
        fin.close()
    else:
        images = list()
        for image in input_images:
            if image is not None:
                images.append(image)
        fout = open(PATH + prefix + '_list' + '.txt', 'w')
        for image in images:
            fout.write(image)
            fout.write('\n')
        fout.close()

    if method == 'identical':
        if os.path.isfile(PATH + prefix + '_sim_' + method + '.npz'):
            sim = scipy.sparse.load_npz(PATH + prefix + '_sim_' + method + '.npz')
        else:
            funs = [imagehash.average_hash, imagehash.phash, imagehash.dhash, imagehash.whash]

            im_objs = list()
            for image in images:
                im_objs.append(Image.open(PATH + image))

            print 'images', len(images), 'im_objs', len(im_objs)

            vs = list()
            for i in xrange(len(im_objs)):
                obj_i = im_objs[i]
                v_i = np.array([fun(obj_i) for fun in funs])
                vs.append(v_i)

            sim = dok_matrix((len(images), len(images)), dtype=np.float32)
            for i in xrange(len(images)):

                current_t = time.time()

                v_i = vs[i]
                for j in xrange(len(images)):
                    v_j = vs[j]
                    s = np.median(v_i - v_j)
                    if s < IDENTICAL_T:
                        sim[i, j] = (IDENTICAL_T - s) / IDENTICAL_T

                print 'processing images ', i, 100*i/len(images), time.time() - current_t, 's'

            sim = sim.asformat('csr')
            scipy.sparse.save_npz(PATH + prefix + '_sim_' + method + '.npz', sim)

        if os.path.isfile(PATH + prefix + '_embeddings_' + method + '_' + str(n_dim) + '.npy'):
            embeddings = np.load(PATH + prefix + '_embeddings_' + method + '_' + str(n_dim) + '.npy')
        else:
            embeddings = cosine_embedding_for_sparse_input.embed(sim, n_dim, embedding_iters)
            np.save(PATH + prefix + '_embeddings_' + method + '_' + str(n_dim) + '.npy', embeddings)

        '''
        if os.path.isfile(PATH + prefix + '_embeddings2_' + method + '_' + str(n_dim) + '.npy'):
            embeddings = np.load(PATH + prefix + '_embeddings2_' + method + '_' + str(n_dim) + '.npy')
        else:
            embeddings = cosine_embedding_for_sparse_input.embed2(sim, n_dim, embedding_iters)
            np.save(PATH + prefix + '_embeddings2_' + method + '_' + str(n_dim) + '.npy', embeddings)
        '''

    if method in ['vgg16', 'vgg19', 'xception', 'inception_resnet_v2']:
        if method == 'vgg16':
            from keras.applications.vgg16 import VGG16
            from keras.preprocessing import image as keras_image
            from keras.applications.vgg16 import preprocess_input
            model = VGG16(weights='imagenet', include_top=False)
        if method == 'vgg19':
            from keras.applications.vgg19 import VGG19
            from keras.preprocessing import image as keras_image
            from keras.applications.vgg19 import preprocess_input
            model = VGG19(weights='imagenet', include_top=False)
        if method == 'xception':
            from keras.applications.xception import Xception
            from keras.preprocessing import image as keras_image
            from keras.applications.xception import preprocess_input
            model = Xception(weights='imagenet', include_top=False)
        if method == 'inception_resnet_v2':
            from keras.applications.inception_resnet_v2 import InceptionResNetV2
            from keras.preprocessing import image as keras_image
            from keras.applications.inception_resnet_v2 import preprocess_input
            model = InceptionResNetV2(weights='imagenet', include_top=False)

        def get_feature(img_path):
            img = keras_image.load_img(img_path, target_size=(224, 224))
            x = keras_image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            feature = model.predict(x)
            return feature

        if os.path.isfile(PATH + prefix + '_embeddings_' + method + '.npy'):
            embeddings = np.load(PATH + prefix + '_embeddings_' + method + '.npy')
        else:
            embeddings = list()
            for image in images:
                embeddings.append(get_feature(PATH + image).flatten())
            embeddings = np.array(embeddings, dtype=np.float32)

            np.save(PATH + prefix + '_embeddings_' + method + '.npy', embeddings)

        #'''
        if os.path.isfile(PATH + prefix + '_sim_' + method + '.npz'):
            sim = scipy.sparse.load_npz(PATH + prefix + '_sim_' + method + '.npz')
        else:
            lsh_instance = lsh.LSH(8, 5)
            indices = lsh_instance.load(embeddings)
            sim = dok_matrix((len(images), len(images)), dtype=np.float32)
            for i in xrange(len(images)):

                v_i = embeddings[i]
                for j in lsh_instance.query(indices[i]):
                    v_j = embeddings[j]
                    sim[i, j] = similarity(v_i, v_j)

                sys.stdout.write("\r%d%%" % (100*i/len(images)))
                sys.stdout.flush()

            sim = sim.asformat('csr')
            scipy.sparse.save_npz(PATH + prefix + '_sim_' + method + '.npz', sim)

        if with_embedding:
            if os.path.isfile(PATH + prefix + '_embeddings_' + method + '_' + str(n_dim) + '.npy'):
                embeddings = np.load(PATH + prefix + '_embeddings_' + method + '_' + str(n_dim) + '.npy')
            else:
                embeddings = cosine_embedding_for_sparse_input.embed(sim, n_dim, embedding_iters)
                np.save(PATH + prefix + '_embeddings_' + method + '_' + str(n_dim) + '.npy', embeddings)

            print 'embeddings', embeddings.shape #debugging
            embeddings = np.append(embeddings, np.zeros((1, embeddings.shape[1]), dtype=np.float32), axis=0)
        else:
            embeddings = None
        #'''

    image2eid = dict(zip(images, range(len(images))))
    image2eid[None] = len(images)

    return image2eid, sim, embeddings



