#본 코드는 Set 12의 코드만 기재하였다(제일 복잡, 나머지는 옵션 하나씩 제외). Setting 별 주요 변화로는,
# Set 4 부터 He weight initilization을 사용,
# Set 5에서만 optimizer = Adadelta 사용
# Set 7~ 10에서 learning rate = 0.01
# Set 6 부터 weight decay = 0
# Set 8 부터 dropout 사용, 정도가 있었다.

import tensorflow as tf
import random
import time
import os
from tensorflow.examples.tutorials.mnist import input_data

##Set 12
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True, validation_size=5000)

X = tf.placeholder(tf.float32, [None, 784], name="X")
Y = tf.placeholder(tf.float32, [None, 10], name="Y")
keep_prob = tf.placeholder(tf.float32, name = "keep_prob") # keep_prob placeholder 를 생성해주어 dropout 비율을 향후에 입력


W1 = tf.get_variable("W1", shape=[784, 400], initializer= tf.initializers.he_normal()) # Setting별로 layer size 다르게 입력 (이후 layer들도 해당), weight initlizaer He Normal 로 설정
b1 = tf.Variable(tf.random_normal([400]))
L1 = tf.nn.relu(tf.matmul(X, W1) + b1)
L1 = tf.nn.dropout(L1, keep_prob=keep_prob) # dropout 위 placehold 한 변수로 설정
L2N = tf.nn.l2_loss(W1) # l2_loss 를 처음 layer 에서 정의해준 뒤에 향후는 +=로 더해줌


W2 = tf.get_variable("W2", shape=[400, 400], initializer= tf.initializers.he_normal()) # *hint* weight initialization을 위해 "initializer" 파라미터에 특정 초기화 기법을 입력
b2 = tf.Variable(tf.random_normal([400]))
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)
L2 = tf.nn.dropout(L2, keep_prob=keep_prob)
L2N += tf.nn.l2_loss(W2)

W3 = tf.get_variable("W3", shape=[400, 400], initializer= tf.initializers.he_normal())
b3 = tf.Variable(tf.random_normal([400]))
L3 = tf.nn.relu(tf.matmul(L2, W3) + b3)
L3 = tf.nn.dropout(L3, keep_prob=keep_prob)
L2N += tf.nn.l2_loss(W3)

W4 = tf.get_variable("W4", shape=[400, 400], initializer= tf.initializers.he_normal())
b4 = tf.Variable(tf.random_normal([400]))
L4 = tf.nn.relu(tf.matmul(L3, W4) + b4)
L4 = tf.nn.dropout(L4, keep_prob=keep_prob)
L2N += tf.nn.l2_loss(W4)


W5 = tf.get_variable("W5", shape=[400, 10], initializer= tf.initializers.he_normal()) #출력 형태 주의하여 10으로 설정
b5 = tf.Variable(tf.random_normal([10]))
L2N += tf.nn.l2_loss(W5)

hypothesis = tf.nn.xw_plus_b(L4, W5, b5, name="hypothesis")
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y)) + L2N*0.005 #cost 에 weight decay 값 더 해주기 (0.01, 0.005)
correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
summary_op = tf.summary.scalar("accuracy", accuracy)


learning_rate = 0.001 # 0.01, 0.001
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# initialize
sess = tf.Session()
sess.run(tf.global_variables_initializer()) # 모든 parameter 값 초기화

training_epochs = 100
batch_size = 200


# ========================================================================
timestamp = str(int(time.time()))
out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
train_summary_dir = os.path.join(out_dir, "summaries", "train")
train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
val_summary_dir = os.path.join(out_dir, "summaries", "dev")
val_summary_writer = tf.summary.FileWriter(val_summary_dir, sess.graph)
checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
checkpoint_prefix = os.path.join(checkpoint_dir, "model")
# ========================================================================

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
saver = tf.train.Saver(tf.global_variables(), max_to_keep=3)

max = 0
early_stopped = 0
start_time = time.time()
for epoch in range(training_epochs):
    avg_cost = 0.0
    avg_acc = 0.0
    total_batch = int(mnist.train.num_examples / batch_size)

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        feed_dict = {X: batch_xs, Y: batch_ys,  keep_prob: 0.9} # train set 에서 dropout = 0.1 (혹은 0.2)
        c, _, a = sess.run([cost, optimizer, accuracy], feed_dict=feed_dict)
        avg_cost += c / total_batch
        avg_acc += a / total_batch

    print('Epoch:', '%04d' % (epoch + 1), 'training cost =', '{:.9f}'.format(avg_cost))
    # 시각화를 위한 accuracy 값 저장, validation accuracy 계산
    # ========================================================================
    train_summary = tf.Summary(value=[tf.Summary.Value(tag="train_accuracy", simple_value=avg_acc)])
    train_summary_writer.add_summary(train_summary, epoch) ##
    val_accuracy, summaries = sess.run([accuracy, summary_op], feed_dict={X: mnist.validation.images, Y: mnist.validation.labels, keep_prob:1}) # valid set 에선 dropout = 0
    val_summary_writer.add_summary(summaries, epoch) ##
    # ========================================================================

    print('Validation Accuracy:', val_accuracy)
    if val_accuracy > max:
        max = val_accuracy
        early_stopped = epoch + 1
        saver.save(sess, checkpoint_prefix, global_step=early_stopped)
training_time = (time.time() - start_time) / 60
print('Learning Finished!')
print('Validation Max Accuracy:', max)
print('Early stopped time:', early_stopped)
print('training time: ', training_time)