# ANN From Scratch
Building an Artificial Neural Network from scratch using pure Python.

**Disclaimer**: This is still a work in progress. It trains and makes predictions. They're just really terrible because the whole process is so slow that training to a high level of performance takes prohibitively long. This whole thing is a learning exercise. If you actually want to build an ANN yourself, you should certainly make use of NumPy and perform the computations using matrices and not for-loops.

## End Goal
Able to demonstrate learning on [MNIST dataset](http://yann.lecun.com/exdb/mnist/) and perform at least better than chance after a relatively short training time.

## Process

Okay, I need some technical understanding about how an ANN works. I'm making a classifier here that takes in MNIST digits as input. The MNIST digits are 28x28 "images", so I'll have $28 * 28 = 784$ input neurons and 10 output neurons (one for each digit).

### Basic Neural Net

For a simple 3-layer NN:
$y = f_{NN}(x) = f_3(\mathbf{f}_2(\mathbf{f}_1(x)))$

Where $f_1$ and $f_2$ are vector functions of the form:
$\mathbf{f}_l(\mathbf{z}) = \mathbf{g}_l(\mathbf{W}_l\mathbf{z} + \mathbf{b}_l)$

* $\mathbf{g}_l$ is the activation function. This should be a differentiable, non-linear function so that our NN is able to approximate non-linear functions. e.g. TanH and ReLU.
* $\mathbf{W}_l$ is a matrix representing the weights for that layer.
* $\mathbf{b}_l$ is a vector representing the biases for that layer.
* $\mathbf{W}_l$ and $\mathbf{b}_l$ are learned using gradient descent and optimising for some loss function.

### Gradient Descent

>During gradient descent, the neural network’s parameters receive an update proportional to the partial derivative of the cost function with respect to the current parameter in each iteration of training. — Andriy Burkov

Loss function: Mean Squared Error (MSE)
$l = \frac{1}{N} \sum_{i=1}^{N}{(y_i - (wx_i + b))^2}$

GD works in **epochs**. In each epoch, you run the whole training set through the network and update the parameters. We initialise the weights ($w$) and biases ($b$) to zero at the beginning of the first epoch. At each epoch, we update $w$ and $b$ in each node/connection using the partial derivatives with respect to the loss function. We control the learning rate ($\alpha$) to adjust the sizes of the updates.

In reality, we'd want to use something like minibatch stochastic gradient descent to increase performance and robustness.

### Backpropagation

> Backpropagation is an efficient algorithm for computing gradients on neural networks using the chain rule. — Andriy Burkov

For $y=g(x)$ and $z=f(g(x)) = f(y)$, the **chain rule** states:
$\frac{dz}{dx} = \frac{dz}{dy}\frac{dy}{dx}$


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


