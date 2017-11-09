# MNIST Digit Recognition
This project is a Deep Learning task which takes a fully connected multi-layer Neural Netwok and trains it on [MNIST](http://yann.lecun.com/exdb/mnist/) dataset. It uses Sigmoid as its activation function and Cross Entropy as its Loss function. It is implemented in Tensorboard, using Tensorboard for visualization.

<p align="center"><img src="https://user-images.githubusercontent.com/19167068/32612287-2607c202-c57d-11e7-97b0-3a21918ffca2.png" width="500"/>

# Training functions
The network has three different functions for training the network:
  ### Train
  Which trains the network normaly. 
  ### Early_Stopping
  In this case, the network stops training if the validation accuracy stops increasing after a while.
  
  <p align="center"><img src="https://user-images.githubusercontent.com/19167068/32612297-28554520-c57d-11e7-8ca5-89df3e27c67f.png" width="500">
 
  ### 5_fold_CV
  This function folds the data into 5 sections and uses one of the foldings as validation set each time. In the end, it averages the result over all of the folds.

# Parameters
The following parameters of the network were tested to obtain the optimum accuracy:
  ## Standard Deviation of Initial Weights
  in the range of \[0, 0.1, 0.3, 1, 2\]  
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
  ## Mini-batches
  Size of the mini-batches change in the range of \[1, 10, 50, 100, 1000\]  
  <p align="center"><img src="https://user-images.githubusercontent.com/19167068/32612293-28016d74-c57d-11e7-9dda-f4104a33ab83.png" width="500">
 
  ## Learning rate
  Learning rate varies in the range of \[0.00001, 0.001, 0.1, 1, 10, 100, 1000\]
  <p align="center"><img src="https://user-images.githubusercontent.com/19167068/32612294-282b1340-c57d-11e7-9b60-fb961cdffc30.png" width="500">
  
  ## Hidden Layer
  Number of hidden layers change in the range of \[5, 10, 20, 25\]. The result is evaluated with the **5_fold_CV** function explained above.
  
  ```
  hidden layer size = 5
  accuracy on test data: 0.854939985275 
  accuracy on validation data: 0.850800001621
  ```
  ```
  hidden layer size = 10
  accuracy on test data: 0.915699994564 
  accuracy on validation data: 0.909490919113
  ```
  ```
  hidden layer size = 20
  accuracy on test data: 0.935119998455
  accuracy on validation data: 0.927890908718 
  ```
  ```
  hidden layer size = 25
  accuracy on test data: 0.936259996891 
  accuracy on validation data: 0.932127285004
  ```
  
You can see other results in the **log** folder.


