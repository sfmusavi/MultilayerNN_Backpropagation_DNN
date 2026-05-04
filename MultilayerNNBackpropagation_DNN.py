#Implementing a 784-100-10 MLP(A NN with 784 features, 100 hidden layers & 10 output units)

import numpy as np
import pandas
import sklearn
import seaborn
import scipy
import os
import struct
import torch
import sys
import time

from matplotlib import pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else'cpu')
#print(torch.cuda.is_available())

start = time.time()

def load_mnist(path, #=r'C:\Users\...\ProgrammingProjects\MyPersonalImplementation\...\data',
               kind): #='train'):
    #loading MNIST data from `path
    labels_path = os.path.join(path, 
                               '%s-labels-idx1-ubyte' % kind)
    images_path = os.path.join(path, 
                               '%s-images-idx3-ubyte' % kind)
        
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', 
                                 lbpath.read(8))
        labels = np.fromfile(lbpath, 
                             dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII", 
                                               imgpath.read(16))
        images = np.fromfile(imgpath, 
                             dtype=np.uint8).reshape(len(labels), 784)
        images = ((images / 255.) - .5) * 2
 
    return images, labels

X_train, y_train = load_mnist(path=r'C:\Users\S.Musavishavazi\ProgrammingProjects\MyPersonalImplementation\Chapter_12\data\MNIST\raw',
                              kind='train')
#print('X_train.shape: {0}'.format(X_train.shape))

X_test, y_test = load_mnist(path=r'C:\Users\S.Musavishavazi\ProgrammingProjects\MyPersonalImplementation\Chapter_12\data\MNIST\raw',
                              kind='t10k')
#print('X_test.shape: {0}'.format(X_test.shape))
print('\n********************\n')

np.savez('minst_scaled.npz',
         X_train=X_train,
         y_train=y_train,
         X_test=X_test,
         y_test=y_test)

mnist = np.load('minst_scaled.npz')
print('mnist.files: \n {0}'.format(mnist.files))

print('\n********************\n')

#print('n_mnist.files : {0} \n'.format(len(mnist.files)))
#print('mnist.files : {0}'.format(mnist.files))
#print('\n********************\n')

X_train = mnist['X_train']
#print('X_train.shape {0} \n'.format(X_train.shape))
#print('y_train.shape: {0}'.format(y_train.shape))
#print('\n********************\n')

#X_train, y_train, X_test, y_test = [mnist[f] for f in ['X_train', 'y_train', 'X_test', 'y_test']]
X_train, y_train, X_test, y_test = [mnist[f] for f in mnist.files]
#print('X_test.shape {0} \n'.format(X_test.shape))
#print('y_test.shape {0}'.format(y_test.shape))
#print('\n********************\n')

class NeuralNetMLP(object):
    #feedforward NN / a  MLP classifier (here: with one hidden layer)
    def __init__(self,
                 n_hidden=100,
                 l2=0.01, # 'l2 != 0' is used for regularization.
                 epochs=2,
                 eta=0.001,
                 shuffle=True, # "shuffle=True" means that the training data will be shuffled in each epoch.
                 minibatch_size=500,
                 seed=1):

        self.random = np.random.RandomState(seed)
        self.n_hidden = n_hidden
        self.l2 = l2
        self.epochs = epochs
        self.eta = eta
        self.shuffle = shuffle
        self.minibatch_size = minibatch_size

    def _onehot(self, y, n_classes):
        #encoding labels into a one-hot representation
        onehot = np.zeros((n_classes, y.shape[0]))
        for idx, val in enumerate(y.astype(int)):
            onehot[val, idx] = 1.
        #print('onehot.shape {0}'.format(onehot.shape))
        #print('\n********************\n')
        return onehot.T

    def _sigmoid(self, z):
        #actication (function logistic function(sigmoid)) calculation
        #print('sigmoid({0}) :  {1}'.format(z, 1. / (1. + np.exp(-np.clip(a=z, a_min=-250, a_max=250)))))
        #print('\n********************\n')
        return 1. / (1. + np.exp(-np.clip(a=z, a_min=-250, a_max=250)))

    def _forward(self, X):
        #forward propagation step calculation

        # step 1: net input of hidden layer
        # [n_examples, n_features] dot [n_features, n_hidden]
        # -> [n_examples, n_hidden]
        #print('X.shape: {0} \n'.format(X.shape))
        #print('w_h.shape: {0} \n'.format(self.w_h.shape))
        z_h = np.dot(X, self.w_h) + self.b_h
        #print('z_h.shape : {0} \n'.format(z_h.shape))

        # step 2: activation of hidden layer
        a_h = self._sigmoid(z_h)
        #print('a_h.shape : {0} \n'.format(a_h.shape))

        # step 3: net input of output layer
        # [n_examples, n_hidden] dot [n_hidden, n_classlabels]
        # -> [n_examples, n_classlabels]

        z_out = np.dot(a_h, self.w_out) + self.b_out
        #print('z_out.shape : {0} \n'.format(z_out.shape))

        # step 4: activation output layer
        a_out = self._sigmoid(z_out)
        #print('a_out.shape : {0} \n'.format(a_out.shape))
        #pass
        return z_h, a_h, z_out, a_out
    
    def _compute_cost(self, y_enc, output):
        #loss function calculation
        L2_term = (self.l2 *
                   (np.sum(self.w_h ** 2.) +
                    np.sum(self.w_out ** 2.)))

        #entropy as a loss function
        term1 = -y_enc * (np.log(output))
        term2 = (1. - y_enc) * np.log(1. - output)
        cost = np.sum(term1 - term2) + L2_term
        return cost

    def predict(self, X):
        #predicting class labels
        
        z_h, a_h, z_out, a_out = self._forward(X)
        y_pred = np.argmax(z_out, axis=1)
        return y_pred

    def fit(self, X_train, y_train, X_valid, y_valid):
        #learning weights from training data.

        #Parameters
        
        #X_train : array, shape = [n_examples, n_features]
            #Input layer with original features.
        #y_train : array, shape = [n_examples]
            #Target class labels.
        #X_valid : array, shape = [n_examples, n_features]
            #Sample features for validation during training
        #y_valid : array, shape = [n_examples]
            #Sample labels for validation during training

        n_output = np.unique(y_train).shape[0]  # number of class labels
        n_features = X_train.shape[1]
        #print('n_output : {0} \n'.format(n_output))
        #print('n_features : {0}'.format(n_features))
        #print('\n********************\n')
        
        #weight initialization
        #weights for input -> hidden
        
        self.b_h = np.zeros(self.n_hidden)
        #print('b_h.shape {0} \n'.format(self.b_h.shape))

        self.w_h = self.random.normal(loc=0.0, scale=0.1,
                                      size=(n_features, self.n_hidden))

        #print('w_h.shape {0}'.format(self.w_h.shape))
        #print('\n********************\n')

        #weights for hidden -> output
        self.b_out = np.zeros(n_output)
        #print('b_out.shape {0} \n'.format(self.b_out.shape))

        self.w_out = self.random.normal(loc=0.0, scale=0.1,
                                        size=(self.n_hidden, n_output))
        #print('w_out.shape {0}'.format(self.w_out.shape))
        #print('\n********************\n')

        epoch_strlen = len(str(self.epochs))  # for progress formatting
        #print('epoch_strlen : {0}'.format(epoch_strlen))
        #print('\n********************\n')

        self.eval_ = {'cost': [], 'train_acc': [], 'valid_acc': []}
        y_train_enc = self._onehot(y_train, n_output)
        #print('y_train_enc.shape : {0}'.format(y_train_enc.shape))
        #print('\n********************\n')

        # iterate over training epochs
        for i in range(self.epochs):
            #print('\n********************\n')
            #print('epoch number {0} out of {1}: \n'.format(i+1, self.epochs))

            # iterate over minibatches
            indices = np.arange(X_train.shape[0])
            #print('indices {0}: \n'.format(np.arange(X_train.shape[0])))

            if self.shuffle:
                self.random.shuffle(indices)

            for start_idx in range(0, indices.shape[0] - self.minibatch_size +1, self.minibatch_size):
                #print('start_idx : {0} \n'.format(start_idx))
                batch_idx = indices[start_idx:start_idx + self.minibatch_size]

                # forward propagation
                z_h, a_h, z_out, a_out = self._forward(X_train[batch_idx])

                # Backpropagation
                
                # [n_examples, n_classlabels]
                delta_out = a_out - y_train_enc[batch_idx]
                #print('delta_out.shape : {0} \n'.format(delta_out.shape))
                
                # [n_examples, n_hidden]
                sigmoid_derivative_h = a_h * (1. - a_h)

                # [n_examples, n_classlabels] dot [n_classlabels, n_hidden]
                # -> [n_examples, n_hidden]
                delta_h = (np.dot(delta_out, self.w_out.T) *
                           sigmoid_derivative_h)
                #print('delta_h.shape : {0} \n'.format(delta_h.shape))

                # [n_features, n_examples] dot [n_examples, n_hidden]
                # -> [n_features, n_hidden]
                grad_w_h = np.dot(X_train[batch_idx].T, delta_h)
                #print('X_train[batch_idx].T.shape : {0}'.format(X_train[batch_idx].shape))
                #print('delta_h.shape : {0}'.format(delta_h.shape))
                #print('grad_w_h.shape : {0} \n'.format(np.dot(X_train[batch_idx].T, delta_h).shape))
                grad_b_h = np.sum(delta_h, axis=0)
                #print('grad_b_h.shape : {0} \n'.format(grad_b_h.shape))

                # [n_hidden, n_examples] dot [n_examples, n_classlabels]
                # -> [n_hidden, n_classlabels]
                grad_w_out = np.dot(a_h.T, delta_out)
                #print('grad_w_out.shape : {0} \n'.format(grad_w_out.shape))
                grad_b_out = np.sum(delta_out, axis=0)
                #print('grad_b_out.shape : {0} \n'.format(grad_b_out.shape))

                #regularization and weight updates
                delta_w_h = (grad_w_h + self.l2*self.w_h)
                #print('delta_w_h.shape : {0} \n'.format(delta_w_h.shape))
                delta_b_h = grad_b_h # bias is not regularized
                #print('delta_b_h.shape : {0} \n'.format(delta_b_h.shape))
                self.w_h -= self.eta * delta_w_h
                #print('w_h.shape : {0} \n'.format(self.w_h.shape))
                self.b_h -= self.eta * delta_b_h
                #print('w_b.shape : {0} \n'.format(self.b_h.shape))
                delta_w_out = (grad_w_out + self.l2*self.w_out)
                #print('delta_w_out.shape : {0} \n'.format(delta_w_out.shape))
                delta_b_out = grad_b_out  # bias is not regularized
                #print('delta_b_out.shape : {0} \n'.format(delta_b_out.shape))
                self.w_out -= self.eta * delta_w_out
                self.b_out -= self.eta * delta_b_out

            #evaluation

            #evaluation after each epoch during training
            z_h, a_h, z_out, a_out = self._forward(X_train)
            cost = self._compute_cost(y_enc=y_train_enc,
                                      output=a_out)
            #y_train_pred = self.predict(X_train)
            #print('y_train_pred.shape : {0} \n'.format(y_train_pred.shape))
            #y_valid_pred = self.predict(X_valid)

            #train_acc = ((np.sum(y_train == y_train_pred)).astype(float) /
             #            X_train.shape[0])
            
            #valid_acc = ((np.sum(y_valid == y_valid_pred)).astype(float) /
             #            X_valid.shape[0])
           
            #print('cost : {0} \n'.format(cost))
            #print('train_acc : {0} \n'.format(train_acc*100))
            #print('valid_acc : {0} \n'.format(valid_acc*100))

            #sys.stderr.write('\r%0*d/%d | Cost: %.2f | Train/Valid Acc.: %.2f%%/%.2f%% ' %(epoch_strlen, i+1, self.epochs, cost, train_acc*100, valid_acc*100))
            #sys.stderr.flush()
            
            self.eval_['cost'].append(cost)
            #self.eval_['train_acc'].append(train_acc)
            #self.eval_['valid_acc'].append(valid_acc)

        return self

nn = NeuralNetMLP(n_hidden=250,
                  l2=0.1, # 'l2 != 0' is used for regularization.
                  epochs=100,
                  eta=0.001,
                  shuffle=True, # "shuffle=True" means that the training data will be shuffled in each epoch.
                  minibatch_size=500,
                  seed=1)
#nn._onehot(y_test, n_classes=10)
#print('nn._sigmoid(1000000) = {0}'.format(nn._sigmoid(1000000)))
#print('\n********************\n')
#print('X_train.shape: {0}'.format(X_train[:55000].shape))
#print('X_test.shape: {0}'.format(X_train[55000:].shape))

nn.fit(X_train[:55000], y_train[:55000], X_train[55000:], y_train[55000:])
print('\n********************\n')

plt.plot(range(nn.epochs), nn.eval_['cost'], label='training')
#plt.plot(range(nn.epochs), nn.eval_['valid_acc'], label='test', linestyle='--')
plt.xlabel('epochs')
plt.ylabel('cost')
plt.legend(loc='upper right')
plt.tight_layout()
#plt.title('comparidon')
plt.show()

print('\n********************\n')
y_test_pred = nn.predict(X_test)
acc = (np.sum(y_test == y_test_pred)
       .astype(float) / X_test.shape[0])
print('\neveluating the generalization performance of the MLP: \n')
print('test accuracy: {0}%'.format(acc * 100))
print('\n********************\n')


miscl_img = X_test[y_test != y_test_pred][:25]
miscl_lab = y_test_pred[y_test != y_test_pred][:25]
correct_lab = y_test[y_test != y_test_pred][:25]

fig, ax = plt.subplots(nrows=5,
                       ncols=5,
                       sharex=True,
                       sharey=True)
ax = ax.flatten()

for i in range(25):
    img = miscl_img[i].reshape(28,28)
    ax[i].imshow(img,
                 cmap='Greys',
                 interpolation='nearest')
    ax[i].set_title('%d) t: %d p: %d' % (i+1, correct_lab[i], miscl_lab[i]))
    
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
#plt.title('true vs. predicted class label')
plt.show()

end = time.time()
print('\n\n')
print('total execution time in seconds : {0}'.format(end - start))
