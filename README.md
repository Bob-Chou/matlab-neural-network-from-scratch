# neural-network-from-scratch-mnist
Build a simple 2-layer neural network for MNIST dataset from scratch on MATLAB (without extra third-party libraries)

# Download the MNIST dataset
Download the dataset [here](http://yann.lecun.com/exdb/mnist/), extract all the four packages under `MNIST/`.

# Build network from scratch
The `main.m` file has organized all the logic to load the data, build a neural network, train the network and test it. All the function required by `main.m` is under the folder `utils/`. Go to the folder and implement all the forward and backward functions and the `main.m` would work. 

After finishing all the required functions, you could check it by running `lib\check_gradient.m`, you should get all the numerical output below `1e-5`. 

When all the scripts has been correctly completed, you might get the log like:
```text
step: 3900 average loss: 0.17226 train acc: 0.96875 test acc: 0.906
step: 4000 average loss: 0.48184 train acc: 0.8125 test acc: 0.906
step: 4100 average loss: 0.096835 train acc: 0.96875 test acc: 0.906
step: 4200 average loss: 0.16163 train acc: 0.96875 test acc: 0.906
step: 4300 average loss: 0.15864 train acc: 0.96875 test acc: 0.906
step: 4400 average loss: 0.39957 train acc: 0.875 test acc: 0.906
step: 4500 average loss: 0.1873 train acc: 0.96875 test acc: 0.906
step: 4600 average loss: 0.23546 train acc: 0.90625 test acc: 0.906
step: 4700 average loss: 0.21651 train acc: 0.9375 test acc: 0.906
step: 4800 average loss: 0.60792 train acc: 0.9375 test acc: 0.906
step: 4900 average loss: 0.55192 train acc: 0.8125 test acc: 0.906
step: 5000 average loss: 0.23814 train acc: 0.9375 test acc: 0.906
```

# Run the completed NN directly or get solution files
Solution files provided by me could be found at the branch `solution`. Run `git checkout solution` to get the solution. You could also checkout to this branch to run the directly completed NN by me.

# Optimal Result of 2 layer NN
The optimal result for this 2-layer NN should be about 0.92 mean accuracy on test set. To reach it, you might modify hyperparams and cross-validate the training result.
