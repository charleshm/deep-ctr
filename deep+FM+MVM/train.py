#!usr/bin/python
# -*- coding: gbk -*-

import tensorflow as tf
from model import DeepMVM
from sklearn.metrics import roc_auc_score
from dataiter import read_batch
import os

# config
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.001, 'learning rate')
flags.DEFINE_integer('max_epoch', 10, ' max train epochs')
flags.DEFINE_integer("batch_size", 1000, "batch size for sgd")
flags.DEFINE_integer("valid_batch_size", 1000, "validate set batch size")
flags.DEFINE_integer("thread_num", 1, "number of thread to read data")
flags.DEFINE_integer("min_after_dequeue", 100, "min_after_dequeue for shuffle queue")
flags.DEFINE_string("checkpoint_dir", "./checkpoint/", "checkpoint dirctory")
flags.DEFINE_string("tensorboard_dir", "./tensorboard/", "summary data saved for tensorboard")
flags.DEFINE_string("optimizer", "adam", "optimization algorithm")
flags.DEFINE_integer('steps_to_validate', 100, 'steps to validate and print')
flags.DEFINE_bool("train_from_checkpoint", False, "reload model from checkpoint and go on training")

if not os.path.exists(FLAGS.checkpoint_dir):
    os.makedirs(FLAGS.checkpoint_dir)
if not os.path.exists(FLAGS.tensorboard_dir):
    os.makedirs(FLAGS.tensorboard_dir)

# read train data
train_label, train_id, train_value = read_batch("../data/train.tfrecords",
                                                FLAGS.max_epoch,
                                                FLAGS.batch_size,
                                                FLAGS.thread_num,
                                                FLAGS.min_after_dequeue)
# read validate data
valid_label, valid_id, valid_value = read_batch("../data/valid.tfrecords",
                                                FLAGS.max_epoch,
                                                100000,
                                                FLAGS.thread_num,
                                                FLAGS.min_after_dequeue)

feature_size  = 1000000
field_size    = 39
factor_size   = 10
label_num     = 1
model         = DeepMVM(feature_size, field_size, factor_size)

# define loss
train_softmax = model.forward(train_id, train_value)
train_label   = tf.to_int64(train_label)
loss          = tf.losses.log_loss(train_label, train_softmax)
_, train_auc  = tf.contrib.metrics.streaming_auc(train_softmax, train_label)

# define optimizer
print("Optimization algorithm: {}".format(FLAGS.optimizer))
if FLAGS.optimizer   == "sgd":
    optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
elif FLAGS.optimizer == "adadelta":
    optimizer = tf.train.AdadeltaOptimizer(FLAGS.learning_rate)
elif FLAGS.optimizer == "adagrad":
    optimizer = tf.train.AdagradOptimizer(FLAGS.learning_rate)
elif FLAGS.optimizer == "adam":
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
else:
    print("Error: unknown optimizer: {}".format(FLAGS.optimizer))
    exit(1)

global_step = tf.Variable(0, name = 'global_step', trainable = False)
train_op    = optimizer.minimize(loss, global_step = global_step)

# eval acc
tf.get_variable_scope().reuse_variables()
valid_softmax      = model.forward(valid_id, valid_value)
valid_label        = tf.to_int64(valid_label)
valid_loss         = tf.losses.log_loss(valid_label, valid_softmax)

# eval auc
_, auc_op          = tf.contrib.metrics.streaming_auc(valid_softmax, valid_label)

# checkpoint
checkpoint_file = FLAGS.checkpoint_dir + "/checkpoint"
saver = tf.train.Saver()

# summary
tf.summary.scalar('loss', valid_loss)
tf.summary.scalar('auc', auc_op)
summary_op = tf.summary.merge_all()

# train loop
with tf.Session() as sess:
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    writer  = tf.summary.FileWriter(FLAGS.tensorboard_dir, sess.graph)
    sess.run(init_op)

    if FLAGS.train_from_checkpoint:
        checkpoint_state = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if checkpoint_state and checkpoint_state.model_checkpoint_path:
            print("Continue training from checkpoint {}".format(checkpoint_state.model_checkpoint_path))
            saver.restore(sess, checkpoint_state.model_checkpoint_path)

    coord   = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    nbatch  = 0
    try:
        while not coord.should_stop():
            _, loss_value, step = sess.run([train_op, loss, global_step])
            if nbatch % 100 == 0:
                print "batch:", '%6d' % nbatch, "loss=", "{:.9f}".format(loss_value)
            if nbatch % 10000 == 0:
                v_loss, vs, vl = sess.run([valid_loss, valid_softmax, valid_label])
                print("Step: {}, loss: {},auc: {}".format(step, v_loss, roc_auc_score(vl, vs)))
            nbatch += 1
            
    except tf.errors.OutOfRangeError:
        print("training done")
    finally:
        coord.request_stop()

    # wait for threads to exit
    coord.join(threads)
    sess.close()