import numpy as np


class TwoLayersNN (object):
    """" TwoLayersNN classifier """

    def __init__ (self, inputDim, hiddenDim, outputDim):
        self.params = dict()
        self.params['w1'] = None
        self.params['b1'] = None
        self.params['w2'] = None
        self.params['b2'] = None
        #########################################################################
        # TODO: 20 points                                                       #
        # - Generate a random NN weight matrix to use to compute loss.          #
        # - By using dictionary (self.params) to store value                    #
        #   with standard normal distribution and Standard deviation = 0.0001.  #
        #########################################################################

        # standard deviation 0.0001, and random the weight matrix
        self.params['w1'] = 0.0001 * np.random.randn(inputDim, hiddenDim)
        self.params['b1'] = np.zeros(hiddenDim)
        self.params['w2'] = 0.0001 * np.random.randn(hiddenDim, outputDim)
        self.params['b2'] = np.zeros(outputDim)



        #########################################################################
        #                       END OF YOUR CODE                                #
        #########################################################################

    def calLoss (self, x, y, reg):
        """
        TwoLayersNN loss function
        D: Input dimension.
        C: Number of Classes.
        N: Number of example.

        Inputs:
        - x: A numpy array of shape (batchSize, D).
        - y: A numpy array of shape (N,) where value < C.
        - reg: (float) regularization strength.

        Returns a tuple of:
        - loss as single float.
        - gradient with respect to each parameter (w1, b1, w2, b2)
        """
        loss = 0.0
        grads = dict()
        grads['w1'] = None
        grads['b1'] = None
        grads['w2'] = None
        grads['b2'] = None
        #############################################################################
        # TODO: 40 points                                                           #
        # - Compute the NN loss and store to loss variable.                         #
        # - Compute gradient for each parameter and store to grads variable.        #
        # - Use Leaky RELU Activation at hidden and output neurons                  #
        # - Use Softmax loss
        # Note:                                                                     #
        # - Use L2 regularization                                                   #
        # Hint:                                                                     #
        # - Do forward pass and calculate loss value                                #
        # - Do backward pass and calculate derivatives for each weight and bias     #
        #############################################################################

        # initialize
        alpha = 0.01
        N = x.shape[0]
        w1 = self.params['w1']
        b1 = self.params['b1']
        w2 = self.params['w2']
        b2 = self.params['b2']

        # foward pass
        z1 = x.dot(w1) + b1

        # leaky relu
        z1 = np.maximum(alpha*z1, z1)

        # backward pass
        z2 = z1.dot(w2) + b2

        # leaky relu
        z2 = np.maximum(alpha*z2, z2)

        #softmax & L2 regularization
        z2 = z2 - np.max(z2, axis=1, keepdims=True)
        exp_z2 = np.exp(z2)
        sum_z2 = np.sum(exp_z2, axis=1, keepdims=True)

        p = exp_z2 / sum_z2
        p_yi = p[np.arange(N), y]
        loss = np.sum(-np.log(p_yi)) / N
        loss += reg * np.sum(w1**2) + reg * np.sum(w2**2)

        # calculate gradient
        ds = p
        ds[range(N), y] -= 1
        ds /= N
        grads['w2'] = z1.T.dot(ds) + 2 * reg * w2
        grads['b2'] = np.sum(ds, axis=0)
        dz1 = ds.dot(w2.T)
        dz1 = np.where(z1 > (z1 * alpha), dz1, dz1 * alpha)
        grads['w1'] = x.T.dot(dz1) + 2 * reg * w1
        grads['b1'] = np.sum(dz1, axis=0)




        #############################################################################
        #                          END OF YOUR CODE                                 #
        #############################################################################

        return loss, grads

    def train (self, x, y, lr=5e-3, reg=5e-3, iterations=100, batchSize=200, decay=0.95, verbose=False):
        """
        Train this linear classifier using stochastic gradient descent.
        D: Input dimension.
        C: Number of Classes.
        N: Number of example.

        Inputs:
        - x: training data of shape (N, D)
        - y: output data of shape (N, ) where value < C
        - lr: (float) learning rate for optimization.
        - reg: (float) regularization strength.
        - iter: (integer) total number of iterations.
        - batchSize: (integer) number of example in each batch running.
        - verbose: (boolean) Print log of loss and training accuracy.

        Outputs:
        A list containing the value of the loss function at each training iteration.
        """

        # Run stochastic gradient descent to optimize W.
        lossHistory = []
        for i in range(iterations):
            xBatch = None
            yBatch = None
            #########################################################################
            # TODO: 10 points                                                       #
            # - Sample batchSize from training data and save to xBatch and yBatch   #
            # - After sampling xBatch should have shape (batchSize, D)              #
            #                  yBatch (batchSize, )                                 #
            # - Use that sample for gradient decent optimization.                   #
            # - Update the weights using the gradient and the learning rate.        #
            #                                                                       #
            # Hint:                                                                 #
            # - Use np.random.choice                                                #
            #########################################################################

            # batch training data
            train = np.random.choice(x.shape[0], batchSize)

            # update x and y Batch and loss and gradient
            xBatch = x[train]
            yBatch = y[train]
            loss, dw = self.calLoss(xBatch, yBatch, reg)

            # update weight & lossHistory
            self.params['w1'] -= lr * dw['w1']
            self.params['b1'] -= lr * dw['b1']
            self.params['w2'] -= lr * dw['w2']
            self.params['b2'] -= lr * dw['b2']
            lossHistory.append(loss)

            #########################################################################
            #                       END OF YOUR CODE                                #
            #########################################################################
            # Decay learning rate
            lr *= decay
            # Print loss for every 100 iterations
            if verbose and i % 100 == 0 and len(lossHistory) is not 0:
                print ('Loop {0} loss {1}'.format(i, lossHistory[i]))

        return lossHistory

    def predict (self, x,):
        """
        Predict the y output.

        Inputs:
        - x: training data of shape (N, D)

        Returns:
        - yPred: output data of shape (N, ) where value < C
        """
        yPred = np.zeros(x.shape[0])
        ###########################################################################
        # TODO: 10 points                                                         #
        # -  Store the predict output in yPred                                    #
        ###########################################################################

        w1 = self.params['w1']
        b1 = self.params['b1']
        w2 = self.params['w2']
        b2 = self.params['b2']
        alpha = 0.01

        # Store the predict output in yPred
        z1 = x.dot(w1) + b1
        z1 = np.maximum(z1 * alpha, z1)
        z2 = z1.dot(w2) + b2
        z2 = np.maximum(z2 * alpha, z2)
        yPred = np.argmax(z2, axis=1)




        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################
        return yPred


    def calAccuracy (self, x, y):
        acc = 0
        ###########################################################################
        # TODO: 10 points                                                         #
        # -  Calculate accuracy of the predict value and store to acc variable    #
        ###########################################################################

        # check if y and predicted value are same and store in acc
        acc = np.mean(y == self.predict(x)) * 100



        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################
        return acc



