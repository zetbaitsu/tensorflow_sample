# A very basic sample ^_^v
# Predict Number is Prime or Not with Tensorflow!

import math

import numpy as np
import tensorflow as tf

NUM_DIGITS = 10


# Represent each input by an array of its binary digits.
def binary_encode(i, num_digits):
    return np.array([i >> d & 1 for d in range(num_digits)])


# The actual method to check is number prime or not, we only use this to generate data
def is_prime(n):
    if n % 2 == 0 and n > 2:
        return False
    return all(n % i for i in range(3, int(math.sqrt(n)) + 1, 2))


# One-hot encode the desired outputs: ["not prime", "prime"]
def prime_encode(i):
    if is_prime(i):
        return np.array([0, 1])
    else:
        return np.array([1, 0])


# Our goal is to produce prime number for the numbers 1 to 100. So it would be
# unfair to include these in our training data. Accordingly, the training data
# corresponds to the numbers 101 to (2 ** NUM_DIGITS - 1).
trX = np.array([binary_encode(i, NUM_DIGITS) for i in range(101, 2 ** NUM_DIGITS)])
trY = np.array([prime_encode(i) for i in range(101, 2 ** NUM_DIGITS)])


# We'll want to randomly initialize weights.
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


# Our model is a standard 1-hidden-layer multi-layer-perceptron with ReLU
# activation. The softmax (which turns arbitrary real-valued outputs into
# probabilities) gets applied in the cost function.
def model(X, w_h, w_o):
    h = tf.nn.relu(tf.matmul(X, w_h))
    return tf.matmul(h, w_o)


# Our variables. The input has width NUM_DIGITS, and the output has width 2.
X = tf.placeholder("float", [None, NUM_DIGITS])
Y = tf.placeholder("float", [None, 2])

# How many units in the hidden layer.
NUM_HIDDEN = 100

# Initialize the weights.
w_h = init_weights([NUM_DIGITS, NUM_HIDDEN])
w_o = init_weights([NUM_HIDDEN, 2])

# Predict y given x using the model.
py_x = model(X, w_h, w_o)

# We'll train our model by minimizing a cost function.
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y))
train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost)

# And we'll make predictions by choosing the largest output.
predict_op = tf.argmax(py_x, 1)


# Finally, we need a way to turn a prediction (and an original number)
# into a prime number output
def predict_prime(i, prediction):
    return [str(i) + " is not prime", str(i) + " is prime"][prediction]


BATCH_SIZE = 128

# Launch the graph in a session
with tf.Session() as sess:
    tf.initialize_all_variables().run()

    for epoch in range(10000):
        # Shuffle the data before each training iteration.
        p = np.random.permutation(range(len(trX)))
        trX, trY = trX[p], trY[p]

        # Train in batches of 128 inputs.
        for start in range(0, len(trX), BATCH_SIZE):
            end = start + BATCH_SIZE
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})

        # And print the current accuracy on the training data.
        print(epoch, np.mean(np.argmax(trY, axis=1) == sess.run(predict_op, feed_dict={X: trX, Y: trY})))

    # And now generate prediction which is prime number or not in range 1 to 100
    numbers = np.arange(1, 101)
    teX = np.transpose(binary_encode(numbers, NUM_DIGITS))
    teY = sess.run(predict_op, feed_dict={X: teX})
    output = np.vectorize(predict_prime)(numbers, teY)

    # Print our prediction
    print(output)

    # Count the correct and wrong answer
    correct_answer = 0
    wrong_answer = 0
    for ans in output:
        i = ans.split(" ")[0]
        if "not" not in ans and is_prime(int(i)):
            correct_answer += 1
        elif "not" in ans and not is_prime(int(i)):
            correct_answer += 1
        else:
            wrong_answer += 1

    # Print how much the correct answer and the wrong answer
    print("Correct answer: ", correct_answer)
    print("Wrong answer: ", wrong_answer)
