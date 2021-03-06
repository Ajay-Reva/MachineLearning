#Following a tutorial from Machine Learning Mastry - Deep Learning with python textbook
from keras.models import Sequential
from keras.layers import Dense
import numpy

#fixed seed for same result.
seed = 7
numpy.random.seed(seed)

#load dataset.
dataset = numpy.loadtxt("C:/Users/ajay.balasubramaniam/Desktop/pima-indians-diabetes.csv", delimiter=",")


#split input(X) and output(Y) from the dataset.
  #Manual splitting
X_train = dataset[0:600,0:8] #all rows (0-599) of the dataset, columns from 0-7 (8th is not included when written as 0:8).
Y_train = dataset[0:600,8]   #all rows (0-599) of the dataset, only the last column that says if it is an onset of diabetes or not.
X_test = dataset[600:768,0:8] #test data is split - the rest of the data apart from training set                                         split: ~ 78%(training) - 22%(test)
Y_test = dataset[600:768,8] #test data is split - the rest of the data apart from training set

  #Automatic splitting, for now the whole data is in X and Y, later in model.fit, validation_split takes care of automatic split. 
#X = dataset[:,0:8] #all rows of the dataset, columns from 0-7 (8th is not included when written as 0:8).
#Y = dataset[:,8]   #all rows of the dataset, only the last column that says if it is an onset of diabetes or not.

#define the model.
model = Sequential()
model.add(Dense(12, input_dim=8, init='uniform', activation='relu')) #network architecture ===> Input_layer(8 inputs - columns 0-7 per row) ------> DenseLayer1(12 neurons) ------> DenseLayer2(8 neurons) -----> DenseLayer3(1 neuron)
model.add(Dense(8, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))

#compile model.
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#fit the model.
model.fit(X, Y, nb_epoch=150, batch_size=10) #Epochs - number of times the network sees the entire training dataset, batch_size - number of samples after which the weights are updated. 

#Validation_split parameter for automatic splitting is used
#model.fit(X, Y, validation_split=0.33, nb_epoch=150, batch_size=10) #Epochs - number of times the network sees the entire training dataset, batch_size - number of samples after which the weights are updated. 

#evaluate the model.
scores = model.evaluate(X_test,Y_test)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
