import numpy as np
import tensorflow as tf
#
# vocab_size=10
# arr0=[-1]*vocab_size
# max_document_length=7
# word_ids = np.zeros(max_document_length, np.int64)
# word_ids=word_ids-1
# print type(arr0)
# print word_ids

#0519
# iternum=10
# str='test'+str(iternum)
# print str
# dataroot='matsmall-past/'
#
# from scipy.io import loadmat
# x = loadmat(dataroot+'iternum5.mat')['feature']
#
# print x
# print type(x)
#
# dataroot2='matsmall2/'
#
# x2 = loadmat(dataroot+'iternum5.mat')['feature']
# print type(x2)
# print x2

import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

fh = logging.FileHandler('log_filename.txt')
fh.setLevel(logging.INFO)
fh.setFormatter(formatter)
logger.addHandler(fh)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
logger.addHandler(ch)
logger.info('This is a test log message.')