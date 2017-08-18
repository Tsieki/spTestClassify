import yaml
import pandas
import os
import numpy
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, normalization
from keras import activations
from keras import initializers 
from keras import metrics
from keras import optimizers
from keras.utils.np_utils import to_categorical
#k_init = initializers.RandomNormal(mean=0.0, stddev=0.1)
#b_init = initializers.RandomNormal(mean=0.0, stddev=0.1)


print("Loading dataset...")
#load test data
#classes = ['water', 'non_water']
#classes = pandas.DataFrame(classes, columns=['classes'])

#test_data = pandas.read_csv('test_videos.txt', header=None, skipinitialspace=1, sep='\\', engine='python')
#fix column indexing
#test_data.columns = (['bin_class','multi_class','vid_name'])

#skip_lines = 2
#data=[]
#for file in test_data.itertuples():
#    infile = open('descriptors\\'+file[1]+'\\'+file[2]+'\\'+file[3]+'.bow')
#    next(infile)
#    next(infile)
#    data.append([yaml.load(infile)['data']])
with open('testDescr.txt') as f:
	data = []
	for line in f:
		data.append([float(x) for x in line.split()])
#print(data)
		
print('Shape of data is: ', len(data), len(data[0]))
test = numpy.array(data)
#test.shape = (test.shape[0], test.shape[2])

#test = numpy.array([(pandas.read_csv('descriptors\\'+file[1]+'\\'+file[2]+'\\'+file[3]+'.bow', header=None, skipinitialspace=1, sep='data: \[|,|\]', engine='python', skiprows=5)).values for file in test_data.itertuples()])
#test[:,0,0:3] = test[:,0,1:4]
#test = test[:,:,0:3]
#test = test.reshape(test.shape[0], test.shape[1]*3)
#test = test[:, 0:test.shape[1]-1]
#t_gt = numpy.array([classes.query('classes == @what_class').index[0] for what_class in test_data.bin_class])
print('Shape of test data is: (total_examples, features)', test.shape)

print("Loading existing model...")

from keras.models import load_model
model = load_model('model.h5')
#model.evaluate(test, t_gt)
print("Writing prediction results of test set to file...")
predictions = model.predict(test)
nw_predictions = pandas.DataFrame(predictions)
w_predictions = pandas.DataFrame(1-predictions)
#nw_predictions.to_csv('non_water_scores.txt', header=None, index=None)
w_predictions.to_csv('water_scores.txt', header=None, index=None)
print("ALL DONE!")

