# ANN From Scratch
Building an Artificial Neural Network from scratch using pure Python.

**Disclaimer**: This is still a work in progress. It trains and makes predictions. They're just really terrible because the whole process is so slow that training to a high level of performance takes prohibitively long. This whole thing is a learning exercise. If you actually want to build an ANN yourself, you should certainly make use of NumPy and perform the computations using matrices and not for-loops.

## End Goal
Able to demonstrate learning on [MNIST dataset](http://yann.lecun.com/exdb/mnist/) and perform at least better than chance after a relatively short training time.

## Process

Okay, I need some technical understanding about how an ANN works. I'm making a classifier here that takes in MNIST digits as input. The MNIST digits are 28x28 "images", so I'll have 784 input neurons and 10 output neurons (one for each digit).

>During gradient descent, the neural network’s parameters receive an update proportional to the partial derivative of the cost function with respect to the current parameter in each iteration of training. — Andriy Burkov

GD works in **epochs**. In each epoch, you run the whole training set through the network and update the parameters. We initialise the weights and biases to some random values at the beginning of the first epoch. At each epoch, we update them in each node/connection using the partial derivatives with respect to the loss function. We control the learning rate to adjust the sizes of the updates.

In reality, we'd want to use something like minibatch stochastic gradient descent to increase performance and robustness

> Backpropagation is an efficient algorithm for computing gradients on neural networks using the chain rule. — Andriy Burkov

### ANN Learning Algorithm

1. Feed forward through network to generate output
2. Calculation of cost/loss/error
3. Propagation of output activations back through network (backprop.) 
4. Find the gradient of each weight and update the weight by some fraction (the learning rate) of the gradient.

Or, in pseudocode from wikipedia,

```
  initialize network weights (often small random values)
  do
     forEach training example named ex
        prediction = neural-net-output(network, ex)  // forward pass
        actual = teacher-output(ex)
        compute error (prediction - actual) at the output units
        compute delta_w_h for all weights from hidden layer to output layer  // backward pass
        compute delta_w_i for all weights from input layer to hidden layer   // backward pass continued
        update network weights // input layer not modified by error estimate
  until all examples classified correctly or another stopping criterion satisfied
  return the network
```

### Things to address
* Regularisation (L1 vs L2) as a means of preventing overfitting.
* Evaluation: Confusion matrix, precision/recall, accuracy?


