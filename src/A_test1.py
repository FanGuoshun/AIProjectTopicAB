import tensorflow as tf 
from  tensorflow.examples.tutorials.mnist  import  input_data
import numpy as np 
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets('./data/', one_hot = True)

num_classes = 10  # 输出大小
input_size = 784  # 输入大小
hidden_units_size = 30  # 隐藏层1节点数量
# hidden_units1_size = 30  # 隐藏层1节点数量
# hidden_units2_size = 30  # 隐藏层2节点数量
# hidden_units3_size = 30  # 隐藏层3节点数量

batch_size = 100
training_iterations = 5000

X = tf.placeholder(tf.float32, shape = [None, input_size])
Y = tf.placeholder(tf.float32, shape = [None, num_classes])

W1 = tf.Variable(tf.random_normal([input_size, hidden_units_size], stddev = 0.1))
B1 = tf.Variable(tf.constant (0.1), [hidden_units_size])

W12 = tf.Variable(tf.random_normal([hidden_units_size,hidden_units_size], stddev = 0.1))
B12 = tf.Variable(tf.constant(0.1), [hidden_units_size])

# W13 = tf.Variable(tf.random_normal([hidden_units_size,hidden_units_size], stddev = 0.1))
# B13 = tf.Variable(tf.constant(0.1), [hidden_units_size])

# W14 = tf.Variable(tf.random_normal([hidden_units_size,hidden_units_size], stddev = 0.1))
# B14 = tf.Variable(tf.constant(0.1), [hidden_units_size])

# W15 = tf.Variable(tf.random_normal([hidden_units_size,hidden_units_size], stddev = 0.1))
# B15 = tf.Variable(tf.constant(0.1), [hidden_units_size])

W2 = tf.Variable(tf.random_normal ([hidden_units_size, num_classes], stddev = 0.1))
B2 = tf.Variable(tf.constant (0.1), [num_classes])

hidden_opt = tf.matmul(X, W1) + B1  # 输入层到隐藏层正向传播
hidden_opt = tf.nn.relu(hidden_opt)  # 激活函数，用于计算节点输出值

hidden_opt2 = tf.matmul(hidden_opt, W12) +B12
hidden_opt2 = tf.nn.relu(hidden_opt2)

# hidden_opt3 = tf.matmul(hidden_opt2, W13) + B13
# hidden_opt3 = tf.nn.relu(hidden_opt3)

# hidden_opt4 = tf.matmul(hidden_opt3, W14) + B14
# hidden_opt4 = tf.nn.relu(hidden_opt4)

# hidden_opt5 = tf.matmul(hidden_opt4, W15) + B15
# hidden_opt5 = tf.nn.relu(hidden_opt5)

final_opt = tf.matmul(hidden_opt2, W2) + B2  # 隐藏层到输出层正向传播


# 对输出层计算交叉熵损失
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=final_opt))
# 梯度下降算法，这里使用了反向传播算法用于修改权重，减小损失
opt = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
# 初始化变量
init = tf.global_variables_initializer()
# 计算准确率
correct_prediction =tf.equal (tf.argmax (Y, 1), tf.argmax(final_opt, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

sess = tf.Session ()
sess.run (init)

images = mnist.train.images
labels = mnist.train.labels
start = 0
end = 3000
x = images[start:end] 
y = labels[start:end]
LOSS = []
step = []
for i in range (training_iterations) :
    rand_index = np.random.choice(end-batch_size)
    batch_input = x[rand_index:(rand_index+batch_size)]
    batch_labels = y[rand_index:(rand_index+batch_size)]

    # 全数据集
    # batch = mnist.train.next_batch (batch_size)
    # batch_input = batch[0]
    # batch_labels = batch[1]

    training_loss = sess.run ([opt, loss], feed_dict = {X: batch_input, Y: batch_labels})
    if i % 50 == 0 :
        train_accuracy = accuracy.eval (session = sess, feed_dict = {X: batch_input,Y: batch_labels})
        print ("step : %d, training accuracy = %g " % (i, train_accuracy))
        step.append(i)
        LOSS.append(training_loss)

print(len(step),len(LOSS))

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(step, LOSS, label='loss (sample_size=1000)')
# plt.scatter(np.array(step), np.array(LOSS), c='red', label='Loss')
ax.set_xlabel('step')
ax.set_ylabel('loss')
fig.suptitle('loss')
handles, labels = ax.get_legend_handles_labels()
plt.show()
