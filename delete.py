import numpy as np
import tensorflow as tf
import os
import argparse
import tfloader
import tfembedding

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

#0521

parser = argparse.ArgumentParser(description='tune VQA MLB baseline')


parser.add_argument('--gdp', type=float, default=0.5,
                    help='general dropout')
parser.add_argument('--dembd', type=int, default=50,
                    help='dimension for word embedding')
parser.add_argument('--dcommon', type=int, default=256,
                    help='q and img projected to the same dimension')
parser.add_argument('--dq', type=int, default=4800,
                    help='dimension for question feature')
parser.add_argument('--dimg', type=int, default=192,
                    help='dimension for images CNN feature')
parser.add_argument('--use-mlb', type=int, default=0,
                    help='0 do not use mlb,1 use mlb')
parser.add_argument('--max-doclength', type=int, default=7,
                    help='max length for each question')
parser.add_argument('--epochs', type=int, default=200,
                    help='training epochs')
parser.add_argument('--batch-size', type=int, default=128,
                    help='batch size for training epoch')
parser.add_argument('--use-glove', type=bool, default=True,
                    help='whether use glove')
parser.add_argument('--is-emtrainable', type=bool, default=False,
                    help='whether embedding matrix trainable')
parser.add_argument('--vocabs', type=int, default=400001,
                    help='if use-glove is true,vocabs=400001')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='learning rate')
parser.add_argument('--n-class', type=int, default=2,
                    help='# output class')
parser.add_argument('--img-size', type=int, default=30,
                    help='size of input images')
parser.add_argument('--channel', type=int, default=3,
                    help='channel for input images')
parser.add_argument('--log-dir', type=str, default='../data/tensorboard',
                    help='directory for tensorboard')
parser.add_argument('--pool-method', type=int, default=0,
                    help='0 concatenate,1 element-wise product')
parser.add_argument('--use-lenet', type=int, default=0,
                    help='0 cifar network,1 lenet')
parser.add_argument('--expnum', type=str, default='delete',
                    help='exp number')
parser.add_argument('--res-root', type=str, default='../data/expresult/0521/',
                    help='path for restoring result')
parser.add_argument('--data-root', type=str, default='../data/shapes',
                    help='path for restoring result')
parser.add_argument('--use-senenc', type=bool, default=False,
                    help='whether use sentence encoding')

args=parser.parse_args()

train_prefix=os.path.join(args.data_root,'train.large')
val_prefix=os.path.join(args.data_root,'val')
test_prefix=os.path.join(args.data_root,'test')

tfembedding.embedding_prepare(args.max_doclength,args.vocabs,args.use_glove,args.is_emtrainable,args.dembd)

shapes_data=tfloader.get_dataset(train_prefix,val_prefix,test_prefix,max_document_length=args.max_doclength,use_glove=args.use_glove)

X_train,y_train,q_train,ques_train= shapes_data.train.images, shapes_data.train.labels, shapes_data.train.queries, shapes_data.train.ques
X_validation,y_validation,q_validation,ques_validation= shapes_data.val.images, shapes_data.val.labels, shapes_data.val.queries, shapes_data.val.ques
X_test,y_test,q_test,ques_test= shapes_data.test.images, shapes_data.test.labels, shapes_data.test.queries, shapes_data.test.ques

def cal(data,description):
    length=len(data)
    one_num=np.sum(data)
    one_percent=one_num*1.0/length
    zero_percent=1-one_percent
    print description,'......'
    print 'length',length
    print 'one_percent',one_percent
    print 'zeor_percent',zero_percent

cal(y_train,'y_train')
cal(y_validation,'y_validation')
cal(y_test,'y_test')