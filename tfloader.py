import numpy as np
import sexpdata
import tensorflow as tf
import tfargs
import tfembedding
import pickle


def extract_query(sexp):
  if isinstance(sexp, sexpdata.Symbol):
    return sexp.value()
  return tuple(extract_query(q) for q in sexp)

def tuple_to_str(exprs):
  if isinstance(exprs, tuple):
    return "{:s}".format(" ".join([tuple_to_str(expr) for expr in exprs]))
  return str(exprs)

def parse_query(query):
  sexp = sexpdata.loads(query)
  data = extract_query(sexp)
  data = tuple_to_str(data)
  data = data.split()
  return data

class Data(object):
  def __init__(self,images,labels,queries,ques):
    self.images=images
    self.labels=labels
    self.queries=queries
    self.ques=ques
    self.batch_id=0

  def next(self, batch_size):
      """ Return a batch of data. When dataset end is reached, start over.
      """
      if self.batch_id == len(self.images):
          self.batch_id = 0
      batch_data = (self.images[self.batch_id:min(self.batch_id +
                                                batch_size, len(self.images))])
      batch_labels = (self.labels[self.batch_id:min(self.batch_id +
                                                    batch_size, len(self.labels))])
      batch_queries = (self.queries[self.batch_id:min(self.batch_id +
                                                    batch_size, len(self.queries))])
      batch_ques = (self.ques[self.batch_id:min(self.batch_id +
                                                    batch_size, len(self.ques))])
      self.batch_id = min(self.batch_id + batch_size, len(self.images))
      return batch_data, batch_labels, batch_queries,batch_ques

class Dataset():
    def __init__(self,train_data,val_data,test_data):
        self.train=train_data
        self.val=val_data
        self.test=test_data
        print 'Dataset with train,val and test data loaded successfully!'

    # def __init__(self,train_data,val_data=None,test_data=None):
    #     self.train=train_data
    #     self.val=val_data
    #     self.test=test_data
    #     print 'Dataset with train_data only loaded successfully!'


def get_dataset(data_prefix_train='../data/shapes/train.tiny',data_prefix_val='../data/shapes/val',data_prefix_test='../data/shapes/test',max_document_length=7,use_glove=True):
    train_data=[]
    val_data=[]
    test_data=[]
    # prepare_embedding(max_document_length)
    if data_prefix_train != '':
        train_data=input_pipeline(data_prefix_train,max_doc_length=max_document_length,use_glove=use_glove)
    if data_prefix_val!='':
        val_data=input_pipeline(data_prefix_val,max_doc_length=max_document_length,use_glove=use_glove)
    if data_prefix_test!='':
        test_data=input_pipeline(data_prefix_test,max_doc_length=max_document_length,use_glove=use_glove)
    return Dataset(train_data,val_data,test_data)

shapes_words = ['NULL', 'red', 'green', 'blue', 'circle', 'square', 'triangle', \
            'above', 'below', 'right', 'left','of', 'is', 'and']
shapes_vocab = {qw:n for n, qw in enumerate(shapes_words)}

def generate_dataset(dataroot='./newly',filename='shapes.tiny.pkl',data_prefix_train='shapes/train.tiny',data_prefix_val='shapes/val',\
                     data_prefix_test='shapes/test',max_doc_length=7,use_glove=False):

    # prepare_embedding(max_doc_length)
    if data_prefix_train != '':
        train_data=input_pipeline(data_prefix_train,max_doc_length,use_glove)
    if data_prefix_val!='':
        val_data=input_pipeline(data_prefix_val,max_doc_length,use_glove)
    if data_prefix_test!='':
        test_data=input_pipeline(data_prefix_test,max_doc_length,use_glove)
    dataset=Dataset(train_data,val_data,test_data)
    pickle.dump(dataset, open(dataroot+'/'+filename, 'w'))
    print 'data dumped successfully!'
    return

def prepare_embedding(max_document_length=10):
    tfargs.definition()
    tfembedding.embedding_prepare(max_document_length)


def input_pipeline(data_prefix='../data/shapes/train.tiny',max_doc_length=6,use_glove=False):

    queries_list=[]
    labels_list=[]
    imgs_list=[]
    ques_list=[]

    fimages = data_prefix + '.input.npy'
    fqueries = data_prefix + '.query'
    foutputs = data_prefix + '.output'
    index = {'true': 1, 'false': 0}

    with open(fqueries) as fq, open(foutputs) as fo:
        inputs = np.load(fimages)
        print 'inputs shape',inputs.shape
        start_idx = 0
        for query, iimg, output in zip(fq, range(inputs.shape[0]), fo):
            output = index[output.strip()]
            query_sexp = sexpdata.loads(query.strip())
            query_data = extract_query(query_sexp)
            #('is', 'green', ('left_of', 'red'))
            query_data = tuple_to_str(query_data)
            #['is green left','of red']
            query_data = query_data.split('_')
            #is green left of red
            query_str=' '.join(query_data)
            query=query_str.split()
            ques = np.zeros((max_doc_length,), dtype=np.int32)
            for i in range(len(query)):
                if query[i] in shapes_vocab:
                    ques[i] = shapes_vocab[query[i]]
                else:
                    ques[i]=shapes_vocab['NULL']
            image = inputs[iimg, :, :, :].copy()
            # make it directly processable by CNN
            tmp = image[:, :, 0].copy()
            image[:, :, 0] = image[:, :, 2]
            image[:, :, 2] = tmp
            # image = np.transpose(image, (2, 0, 1)) / 255.
            image = np.float32(image)

            labels_list.append(output)
            imgs_list.append(image)
            queries_list.append(query_str)
            ques_list.append(ques)

    imgs_arr=np.array(imgs_list)

    labels_arr=np.array(labels_list)
    queries_arr=np.array(queries_list)
    if use_glove:
        #using self defined transform_vocab not transform
        ques_arr=np.array(list(tfargs.Pretrain.transform_vocab(queries_arr)))
    else:
        ques_arr=np.array(ques_list)

    print data_prefix,imgs_arr.shape,' imgs ',labels_arr.shape,' labels ',queries_arr.shape,'ques',ques_arr.shape,' queries loaded successfully!'
    print ques_arr[0],labels_arr[0]
    return Data(imgs_arr,labels_arr,queries_arr,ques_arr)

def generate_all_dataset(dataroot='./noglovedata'):

    rawdata_prefix_list=['shapes/train.tiny','shapes/train.small','shapes/train.med','shapes/train.large']
    filename_list=['shapes.tiny.pkl','shapes.small.pkl','shapes.med.pkl','shapes.large.pkl']
    for r,f in zip(rawdata_prefix_list,filename_list):
        generate_dataset(dataroot=dataroot,filename=f,data_prefix_train=r,max_doc_length=7,use_glove=False)

def load_shapes(dataroot):
    dataset=pickle.load(open(dataroot))
    train=dataset.train
    test=dataset.test
    valid=dataset.valid
    print dataroot,'loaded successfully!'
    return train,valid,test

if __name__=="__main__":
    train_prefix = '../data/shapes/train.tiny'
    val_prefix = '../data/shapes/val'
    test_prefix = '../data/shapes/test'
    # q0='is green left of red'
    # q1='is square left of above square'
    # q2='is triangle above green'
    max_document_length=7
    embedded_dim=50
    # batch_size=2
    # hidden_size=128

    batch_size=2
    start_id=0
    tfembedding.embedding_prepare(max_document_length)
    shapes_data = get_dataset(train_prefix, val_prefix, test_prefix,max_document_length)

    X_train, y_train, q_train,ques_train = shapes_data.train.images, shapes_data.train.labels, shapes_data.train.queries,shapes_data.train.ques
    #
    # ques=tf.placeholder(tf.int32)
    embedded_chars = tfembedding.get_embedded_from_wordid(ques_train[start_id:start_id+batch_size],batch_size,max_document_length,embedded_dim)
    print embedded_chars
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print 'embedded_chars',sess.run(embedded_chars)

    print ''

#pickle problem
    # datapath='./noglovedata/shapes.small.pkl'
    # dataset=pickle.load(open(datapath))
    # print dataset.train

#zero vector works
    # train_prefix = '../data/shapes/train.tiny'
    # val_prefix = '../data/shapes/val'
    # test_prefix = '../data/shapes/test'
    # # q0='is green left of red'
    # # q1='is square left of above square'
    # # q2='is triangle above green'
    # max_document_length=7
    # embedded_dim=50
    # # batch_size=2
    # # hidden_size=128
    #
    # batch_size=2
    # start_id=0
    # tfembedding.embedding_prepare(max_document_length)
    # shapes_data = get_dataset(train_prefix, val_prefix, test_prefix,max_document_length)
    #
    # X_train, y_train, q_train,ques_train = shapes_data.train.images, shapes_data.train.labels, shapes_data.train.queries,shapes_data.train.ques
    # #
    # # ques=tf.placeholder(tf.int32)
    # embedded_chars = tfembedding.get_embedded_from_wordid(ques_train[start_id:start_id+batch_size],batch_size,max_document_length,embedded_dim)
    # print embedded_chars
    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #     print 'embedded_chars',sess.run(embedded_chars)
