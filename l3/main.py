import matplotlib.pyplot as plt
import tensorflow as tf
noise = tf.random.normal(shape=[100], stddev=0.2)

x = tf.random.uniform(shape=[100], minval=0, maxval=10)

k_1_true = 1
k_0_true = 0
b_true = 4

y = k_1_true * x ** 2 + k_0_true * x + b_true + noise
k_1 = tf.Variable(0.3)
k_0 = tf.Variable(0.4)
b = tf.Variable(0.5)
f = k_1 * x ** 2 + k_0 * x + b
loss = tf.reduce_mean(tf.square(y - f))
EPOCHS = 5
learning_rate = 0.0002

for n in range(EPOCHS):
    with tf.GradientTape() as t:
        f = k_1 * x**2 + k_0 * x + b
        loss = tf.reduce_mean(tf.square(y - f))
        print(f, " loss")

    dk_1, dk_2, db = t.gradient(loss, [k_1, k_0, b])
    print(dk_1, dk_2, db)
    k_1.assign_sub(learning_rate * dk_1)
    k_0.assign_sub(learning_rate * dk_2)
    b.assign_sub(learning_rate * db)

print(k_1, k_0, b, sep="\n")

y_pr = k_1 * x ** 2 + k_0 * x + b
plt.scatter(x, y, s=2)
plt.scatter(x, y_pr, c='r', s=2)
plt.show()
