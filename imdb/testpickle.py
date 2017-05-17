import pickle
import tfloader
#
# class Dataset():
#     def __init__(self,train_data,val_data,test_data):
#         self.train=train_data
#         self.val=val_data
#         self.test=test_data
#         print 'Dataset with train,val and test data loaded successfully!'
#
# if __name__=='__main__':
#
#     dataroot='./noglovedata/shapes.small.pkl'
#     shapes_data =pickle.load(open(dataroot))
#     print 'd'
#

import tfloader
tfloader.generate_all_dataset('./newly')

dataroot='./noglovedata/shapes.small.pkl'
shapes_data =pickle.load(open(dataroot))

print shapes_data.train.ques