"""
This script is what created the dataset pickled.

1) You need to download this file and put it in the same directory as this file.
https://github.com/moses-smt/mosesdecoder/raw/master/scripts/tokenizer/tokenizer.perl . Give it execution permission.

2) Get the dataset from http://ai.stanford.edu/~amaas/data/sentiment/ and extract it in the current directory.

3) Then run this script.
"""

# dataset_path='/Tmp/bastienf/aclImdb/'
dataset_path='/home/huangkun/works/repertory/ProjectDesign-local/shapes_tensorflow/aclImdb/'

import numpy
import cPickle as pkl

from collections import OrderedDict

import glob
import os

from subprocess import Popen, PIPE
import tfembedding
import tfargs
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.contrib import learn

# tokenizer.perl is from Moses: https://github.com/moses-smt/mosesdecoder/tree/master/scripts/tokenizer
# tokenizer_cmd = ['./tokenizer.perl', '-l', 'en', '-q', '-']
tokenizer_cmd = ['./tokenizer.perl', '-l', 'en','-q', '-','-e']



def tokenize(sentences):

    print 'Tokenizing..',
    text = "\n".join(sentences)
    tokenizer = Popen(tokenizer_cmd, stdin=PIPE, stdout=PIPE)
    tok_text, _ = tokenizer.communicate(text)
    toks = tok_text.split('\n')[:-1]
    print 'Done'

    return toks


def build_dict(path):
    print 'path',path
    sentences = []
    currdir = os.getcwd()
    print currdir
    os.chdir('%s/pos/' % path)
    for ff in glob.glob("*.txt"):
        with open(ff, 'r') as f:
            sentences.append(f.readline().strip())
    os.chdir('%s/neg/' % path)
    for ff in glob.glob("*.txt"):
        with open(ff, 'r') as f:
            sentences.append(f.readline().strip())
    os.chdir(currdir)

    sentences = tokenize(sentences)

    print 'Building dictionary..',
    wordcount = dict()
    for ss in sentences:
        words = ss.strip().lower().split()
        for w in words:
            if w not in wordcount:
                wordcount[w] = 1
            else:
                wordcount[w] += 1

    counts = wordcount.values()
    keys = wordcount.keys()

    sorted_idx = numpy.argsort(counts)[::-1]

    worddict = dict()

    for idx, ss in enumerate(sorted_idx):
        worddict[keys[ss]] = idx+2  # leave 0 and 1 (UNK)

    print numpy.sum(counts), ' total words ', len(keys), ' unique words'

    return worddict


def grab_data(path, dictionary):
    sentences = []
    currdir = os.getcwd()
    os.chdir(path)
    for ff in glob.glob("*.txt"):
        with open(ff, 'r') as f:
            sentences.append(f.readline().strip())
    os.chdir(currdir)
    sentences = tokenize(sentences)

    seqs = [None] * len(sentences)
    for idx, ss in enumerate(sentences):
        words = ss.strip().lower().split()
        seqs[idx] = [dictionary[w] if w in dictionary else 1 for w in words]

    return seqs

def grab_data2(path, dictionary):
    sentences = []
    currdir = os.getcwd()
    os.chdir(path)
    for ff in glob.glob("*.txt"):
        with open(ff, 'r') as f:
            sentences.append(f.readline().strip())
    os.chdir(currdir)
    sentences = tokenize(sentences)

    # seqs = [None] * len(sentences)

    seqs=list(tfargs.Pretrain.transform_vocab(sentences))

    return seqs

def embedding_prepare(max_document_length=100,use_glove=True,trainable=False):
    print 'use_glove:',use_glove,'max_document_length:',max_document_length
    if use_glove:
        vocab,embd=tfembedding.loadGloVe()
        vocab_size = len(vocab)
        embedding_dim = len(embd[0])

        tfargs.embedded_dim=embedding_dim
        tfargs.vocab_size=vocab_size

        embedding_matrix = numpy.asarray(embd)
        tfargs.Embedding_tensor = tf.Variable(tf.constant(0.1, shape=[vocab_size, embedding_dim]),
                                          trainable=trainable, name="W")
        tfargs.Embedding_tensor=tfargs.Embedding_tensor.assign(embedding_matrix)
        vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
        tfargs.Pretrain= vocab_processor.fit_vocab(vocab)
    else:
        tfargs.Embedding_tensor = tf.Variable(tf.random_normal(shape=[tfargs.vocab_size, tfargs.embedded_dim]),trainable=True, name="W")
        print 'Not use glove,embdtensor is :',tfargs.Embedding_tensor

    return


def main():
    # Get the dataset from http://ai.stanford.edu/~amaas/data/sentiment/
    path = dataset_path
    # dictionary = build_dict(os.path.join(path, 'train'))
    dictionary=None
    max_document_length=100
    use_glove=True
    trainable=True
    tfembedding.embedding_prepare(max_document_length=max_document_length,use_glove=use_glove,trainable=trainable)

    train_x_pos = grab_data2(path+'train/pos', dictionary)
    train_x_neg = grab_data2(path+'train/neg', dictionary)
    train_x = train_x_pos + train_x_neg
    train_y = [1] * len(train_x_pos) + [0] * len(train_x_neg)

    test_x_pos = grab_data2(path+'test/pos', dictionary)
    test_x_neg = grab_data2(path+'test/neg', dictionary)
    test_x = test_x_pos + test_x_neg
    test_y = [1] * len(test_x_pos) + [0] * len(test_x_neg)

    train_x,train_y,test_x,test_y=shuffle(train_x,train_y,test_x,test_y)

    valid_portion=0.1
    n_samples = len(train_x)
    n_train = int(numpy.round(n_samples * (1. - valid_portion)))
    train_x_new=train_x[0:n_train]
    train_y_new=train_y[0:n_train]
    valid_x=train_x[n_train:]
    valid_y=train_y[n_train:]

    print 'len:train',len(train_x_new),'len:valid',len(valid_x),'len:test',len(test_x)

    f_train = open('imdb.train.pkl', 'wb')
    f_valid = open('imdb.valid.pkl', 'wb')
    f_test = open('imdb.test.pkl', 'wb')

    pkl.dump((train_x_new, train_y_new), f_train, -1)
    pkl.dump((valid_x, valid_y), f_valid, -1)
    pkl.dump((test_x,test_y),f_test,-1)

    f_train.close()
    f_valid.close()
    f_test.close()

    # f = open('imdb.dict.pkl', 'wb')
    # pkl.dump(dictionary, f, -1)
    # f.close()

if __name__ == '__main__':
    main()