#!/usr/bin/env bash



python main.py --path ../data_for_experiment/facebook_twitter/ \
                --source_prefix fb --target_prefix tw --source_col 0 --target_col 1  \
                --name_method tfidf --image_exist True --user_dim 1024+256 \
                --name_concatenate True \
                --image_dim 256 --n_iter 52 --supervised False\
                 --image_method vgg16 --skip_network False;



