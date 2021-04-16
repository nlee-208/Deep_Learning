import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True, validation_size=5000)
# 모델이 저장된 checkpoint 경로
files = {'Set1':'1600571012', 'Set2':'1600572505', 'Set3':'1600573329', 'Set4':'1600575935', 'Set5':'1600576727',
         'Set6':'1600579205', 'Set7':'1600579755'}
files_drop = {'Set8':'1600582008', 'Set9':'1600582807', 'Set10':'1600583570',
         'Set11':'1600584261', 'Set12': '1600584941'}

# for no dropout
tf.flags.DEFINE_string("checkpoint_dir", "./runs/1600571012/checkpoints", "Checkpoint directory from training run")
FLAGS = tf.flags.FLAGS
# ==================================================
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir) #가장 validation accuracy가 높은 시점 load
graph = tf.Graph()
with graph.as_default():
    sess = tf.Session()
    with sess.as_default():
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file) # 저장했던 모델 load

        # Get the placeholders from the graph by name, name을 통해 operation 가져오기
        X = graph.get_operation_by_name("X").outputs[0]
        Y = graph.get_operation_by_name("Y").outputs[0]
        hypothesis = graph.get_operation_by_name("hypothesis").outputs[0]

        correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        test_accuracy = sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels})
        print('Test Max Accuracy:', test_accuracy)

# for dropout, keep prob 추가
# for no dropout
tf.flags.DEFINE_string("checkpoint_dir", "./runs/1600571012/checkpoints", "Checkpoint directory from training run")
FLAGS = tf.flags.FLAGS
# ==================================================
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    sess = tf.Session()
    with sess.as_default():
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file) # 저장했던 모델 load

        # Get the placeholders from the graph by name, name을 통해 operation 가져오기
        X = graph.get_operation_by_name("X").outputs[0]
        Y = graph.get_operation_by_name("Y").outputs[0]
        keep_prob = graph.get_operation_by_name("keep_prob").outputs[0] # keep prob 추가 dropout
        hypothesis = graph.get_operation_by_name("hypothesis").outputs[0]

        correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        test_accuracy = sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1})
        print('Test Max Accuracy:', test_accuracy)



