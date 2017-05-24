#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import os
import argparse
import tfloader
import tfembedding
import tfloader2

data_dir = '../../nmn2/data/shapes/'
train_prefix='../../nmn2/data/shapes/train.large'
val_prefix='../../nmn2/data/shapes/val'
test_prefix='../../nmn2/data/shapes/test'

max_doclength=7

shapes_data=tfloader2.get_dataset(train_prefix,val_prefix,test_prefix,max_document_length=max_doclength)

X_train,y_train,q_train,ques_train= shapes_data.train.images, shapes_data.train.labels, shapes_data.train.queries, shapes_data.train.ques
X_validation,y_validation,q_validation,ques_validation= shapes_data.val.images, shapes_data.val.labels, shapes_data.val.queries, shapes_data.val.ques
X_test,y_test,q_test,ques_test= shapes_data.test.images, shapes_data.test.labels, shapes_data.test.queries, shapes_data.test.ques

