# MNIST Digit Recognition
This project is a Deep Learning task which takes a fully connected multi-layer Neural Netwok and trains it on [MNIST](http://yann.lecun.com/exdb/mnist/) dataset. It is implemented in Tensorboard, using Tensorboard for visualization.

The network has three different functions for training the network:
  1. **Train**, e.i. normal trainin. 
  2. **Early_Stopping**, that stops training if validation accuracy stops increasing after a while.
  3. **5_fold_CV**, which folds the data in 5 and uses one of the foldings as validation set each time. In the end it averages the result on all of the folds.

The following parameters of the network were tested to obtain the optimum accuracy:
   * **Standard deviation of initial weights** in the range of \[0, 0.1, 0.3, 1, 2\]
   * **Size of the mini-batches** in the range of \[1, 10, 50, 100, 1000\]
   * **Learning rate** in the range of \[0.00001, 0.001, 0.1, 1, 10, 100, 1000\]
   * **Number of hidden layers** in the range of \[5, 10, 20, 25\]  
You can see some of the results in the **log** folder.
