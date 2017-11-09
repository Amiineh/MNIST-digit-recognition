# MNIST Digit Recognition
This project is a Deep Learning task which takes a fully connected multi-layer Neural Netwok and trains it on [MNIST](http://yann.lecun.com/exdb/mnist/) dataset. It is implemented in Tensorboard, using Tensorboard for visualization.

<p align="center"><img src="https://user-images.githubusercontent.com/19167068/32612287-2607c202-c57d-11e7-97b0-3a21918ffca2.png" width="400"/>

The network has three different functions for training the network:
  * **Train**, e.i. normal training. 
  * **Early_Stopping**, that stops training if the validation accuracy stops increasing after a while.
  
  <p align="center"><img src="https://user-images.githubusercontent.com/19167068/32612297-28554520-c57d-11e7-8ca5-89df3e27c67f.png">
 
  * **5_fold_CV**, which folds the data into 5 sections and uses one of the foldings as validation set each time. In the end it averages the result over all of the folds.

The following parameters of the network were tested to obtain the optimum accuracy:
  * **Standard deviation of initial weights** in the range of \[0, 0.1, 0.3, 1, 2\]
  initial W = 0:
  ```
  accuracy on training data: 0.6674 
  accuracy on test data: 0.671
  ```
  std = 0.1:
  ```
  accuracy on training data: 0.941509 
  accuracy on test data: 0.93
  ```
  std = 0.3:
  ```
  accuracy on training data: 0.9396 
  accuracy on test data: 0.9321
  ```
  std = 1:
  ```
  accuracy on training data: 0.908982 
  accuracy on test data: 0.9047
  ```
  std = 2:
  ```
  accuracy on training data: 0.878745 
  accuracy on test data: 0.8739
  ```
  * **Size of the mini-batches** in the range of \[1, 10, 50, 100, 1000\]
  * **Learning rate** in the range of \[0.00001, 0.001, 0.1, 1, 10, 100, 1000\]
  * **Number of hidden layers** in the range of \[5, 10, 20, 25\]
  
You can see some of the results in the **log** folder.


