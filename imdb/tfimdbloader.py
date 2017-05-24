from __future__ import print_function
from six.moves import xrange
import six.moves.cPickle as pickle

import gzip
import os

import numpy
import theano
import tfloader
import tfargs
import tensorflow as tf

def prepare_data(seqs, labels, maxlen=None):
    """Create the matrices from the datasets.

    This pad each sequence to the same lenght: the lenght of the
    longuest sequence or maxlen.

    if maxlen is set, we will cut all sequence to this maximum
    lenght.

    This swap the axis!
    """
    # x: a list of sentences
    lengths = [len(s) for s in seqs]

    if maxlen is not None:
        new_seqs = []
        new_labels = []
        new_lengths = []
        for l, s, y in zip(lengths, seqs, labels):
            if l < maxlen:
                new_seqs.append(s)
                new_labels.append(y)
                new_lengths.append(l)
        lengths = new_lengths
        labels = new_labels
        seqs = new_seqs

        if len(lengths) < 1:
            return None, None, None

    n_samples = len(seqs)
    maxlen = numpy.max(lengths)

    x = numpy.zeros((maxlen, n_samples)).astype('int64')
    x_mask = numpy.zeros((maxlen, n_samples)).astype(theano.config.floatX)
    for idx, s in enumerate(seqs):
        x[:lengths[idx], idx] = s
        x_mask[:lengths[idx], idx] = 1.

    return x, x_mask, labels


def get_dataset_file(dataset, default_dataset, origin):
    '''Look for it as if it was a full path, if not, try local file,
    if not try in the data directory.

    Download dataset if it is not present

    '''
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            "..",
            "data",
            dataset
        )
        if os.path.isfile(new_path) or data_file == default_dataset:
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == default_dataset:
        from six.moves import urllib
        print('Downloading data from %s' % origin)
        urllib.request.urlretrieve(origin, dataset)

    return dataset


def load_data(path="imdb.pkl", n_words=100000, valid_portion=0.1, maxlen=None,
              sort_by_len=True):
    '''Loads the dataset

    :type path: String
    :param path: The path to the dataset (here IMDB)
    :type n_words: int
    :param n_words: The number of word to keep in the vocabulary.
        All extra words are set to unknow (1).
    :type valid_portion: float
    :param valid_portion: The proportion of the full train set used for
        the validation set.
    :type maxlen: None or positive int
    :param maxlen: the max sequence length we use in the train/valid set.
    :type sort_by_len: bool
    :name sort_by_len: Sort by the sequence lenght for the train,
        valid and test set. This allow faster execution as it cause
        less padding per minibatch. Another mechanism must be used to
        shuffle the train set at each epoch.

    '''

    #############
    # LOAD DATA #
    #############

    # Load the dataset
    path = get_dataset_file(
        path, "imdb.pkl",
        "http://www.iro.umontreal.ca/~lisa/deep/data/imdb.pkl")

    if path.endswith(".gz"):
        f = gzip.open(path, 'rb')
    else:
        f = open(path, 'rb')

    train_set = pickle.load(f)
    test_set = pickle.load(f)

    f.close()
    if maxlen:
        new_train_set_x = []
        new_train_set_y = []
        for x, y in zip(train_set[0], train_set[1]):
            if len(x) < maxlen:
                new_train_set_x.append(x)
                new_train_set_y.append(y)
        train_set = (new_train_set_x, new_train_set_y)
        del new_train_set_x, new_train_set_y

    # split training set into validation set
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = numpy.random.permutation(n_samples)
    n_train = int(numpy.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    train_set = (train_set_x, train_set_y)
    valid_set = (valid_set_x, valid_set_y)

    def remove_unk(x):
        return [[1 if w >= n_words else w for w in sen] for sen in x]

    test_set_x, test_set_y = test_set
    valid_set_x, valid_set_y = valid_set
    train_set_x, train_set_y = train_set

    train_set_x = remove_unk(train_set_x)
    valid_set_x = remove_unk(valid_set_x)
    test_set_x = remove_unk(test_set_x)

    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    if sort_by_len:
        sorted_index = len_argsort(test_set_x)
        test_set_x = [test_set_x[i] for i in sorted_index]
        test_set_y = [test_set_y[i] for i in sorted_index]

        sorted_index = len_argsort(valid_set_x)
        valid_set_x = [valid_set_x[i] for i in sorted_index]
        valid_set_y = [valid_set_y[i] for i in sorted_index]

        sorted_index = len_argsort(train_set_x)
        train_set_x = [train_set_x[i] for i in sorted_index]
        train_set_y = [train_set_y[i] for i in sorted_index]

    train = (train_set_x, train_set_y)
    valid = (valid_set_x, valid_set_y)
    test = (test_set_x, test_set_y)

    return train, valid, test


def load_dictionary(path):
    path = get_dataset_file(path, 'imdb.dict.pkl', "http://www.iro.umontreal.ca/~lisa/deep/data/imdb.pkl")
    if path.endswith('.gz'):
        f = gzip.open(path, 'rb')
    else:
        f = open(path, 'rb')
    dic_set = pickle.load(f)
    print(len(dic_set))
    print(type(dic_set))
    return dic_set


def load_raw_data(path):
    path = get_dataset_file(path, 'imdb.dict.pkl', "http://www.iro.umontreal.ca/~lisa/deep/data/imdb.pkl")
    if path.endswith('.gz'):
        f = gzip.open(path, 'rb')
    else:
        f = open(path, 'rb')
    data = pickle.load(f)
    print(len(data))
    print(type(data))
    return data



def load_imdb(path='./',max_length=100):

    path_train=os.path.join(path,'imdb.train.pkl')
    path_valid=os.path.join(path,'imdb.valid.pkl')
    path_test=os.path.join(path,'imdb.test.pkl')

    f_train = open(path_train, 'rb')
    f_valid=open(path_valid,'rb')
    f_test=open(path_test,'rb')

    train = pickle.load(f_train)
    valid=pickle.load(f_valid)
    test=pickle.load(f_test)

    train_ques_arr=numpy.array(train[0])
    train_labels_arr=numpy.array(train[1])
    valid_ques_arr=numpy.array(valid[0])
    valid_labels_arr=numpy.array(valid[1])
    test_ques_arr=numpy.array(test[0])
    test_labels_arr=numpy.array(test[1])
    train_data=tfloader.Data(None,train_labels_arr,None,train_ques_arr)
    valid_data=tfloader.Data(None,valid_labels_arr,None,valid_ques_arr)
    test_data=tfloader.Data(None,test_labels_arr,None,test_ques_arr)
    print('imdb dataset loaded successfully!')
    print('train:',train_ques_arr.shape,train_labels_arr.shape,'validation:',valid_ques_arr.shape,'test',test_ques_arr.shape)

    return tfloader.Dataset(train_data,valid_data,test_data)


if __name__ == '__main__':
    # path = 'aclImdb/imdb.pkl'
    # path_dic = 'aclImdb/imdb.dict.pkl'
    # train,valid,test=load_data(path)
    dataset=load_imdb()





#history
    # path='../data/'
    # max_document_length=100
    # imdb_data=load_imdb(path,max_document_length)
    # ques_train=imdb_data.train.ques
    # batch_size=2
    # test=numpy.array([batch_size,10])
    # print(type(test[0]),test[0])
    # a=[2,3,4]
    # b=[4,5,6,7]
    # c=[]
    # c.append(a)
    # c.append(b)
    # print (c,type(c))
    # d=numpy.array(c)
    # print (d,d.shape,type(d),type(d[0]))
    # y=tf.placeholder(tf.int64,None)
    # x=tf.placeholder(tf.int32)
    # dic_set=load_dictionary('../data/imdb.dict.pkl')
    # for i in range(5):
    #     print(dic_set['is'])
    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())






