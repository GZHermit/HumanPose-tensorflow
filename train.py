# coding: utf-8
import os

import tensorflow as tf

from models.single import Single
from utils.data_handle import load_weight, save_weight
from utils.image_reader import ImageReader


def train(args):
    ## set hyparameter
    tf.set_random_seed(args.random_seed)
    coord = tf.train.Coordinator()

    ## load data
    reader = ImageReader(
        args.data_dir,
        args.img_size,
        args.random_scale,
        args.random_mirror,
        args.random_crop,
        args.operate,
        coord)
    image, gtMaps = reader.dequeue(args.batch_size)
    print("数据加载已经完成！")

    ## load model
    # image_batch = tf.placeholder(shape=[None, None, None, 3], dtype=tf.float32, name='image')
    net = Single({'data': image})
    predict = net.get_output()
    print("模型加载已经完成！")

    ## set loss
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=predict, labels=gtMaps),
                          name='cross_entropy_loss')
    loss_var, loss_op = tf.metrics.mean(loss)
    metrics_op = tf.group(loss_op)

    ## set optimizer
    iterstep = tf.placeholder(dtype=tf.float32, shape=[], name='iteration_step')
    lr = tf.train.exponential_decay(args.learning_rate, args.train_step, args.decay_step, args.decay,
                                    staircase=True, name='learning_rate')
    train_op = tf.train.RMSPropOptimizer(learning_rate=lr).minimize(loss, var_list=tf.trainable_variables())

    ## set summary
    tf.summary.scalar('loss_train', loss_var)

    summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(args.log_dir, graph=tf.get_default_graph(), max_queue=5)

    ## set session
    sess = tf.Session()
    global_init = tf.global_variables_initializer()
    local_init = tf.local_variables_initializer()
    sess.run(global_init)
    sess.run(local_init)

    ## set saver
    saver = tf.train.Saver(var_list=tf.global_variables())
    trained_step = 0
    if os.path.exists(args.restore_from + 'checkpoint'):
        trained_step = load_weight(args.restore_from, saver, sess)

    threads = tf.train.start_queue_runners(sess, coord)
    print("所有设置均已完成，训练开始！")

    ## start training
    for step in range(args.num_steps):
        now_step = int(trained_step) + step if trained_step is not None else step
    feed_dict = {iterstep: now_step}
    losses, _, _ = sess.run([loss_var, train_op, metrics_op], feed_dict)
    if step % args.save_pred_every == 0:
        save_weight(args.restore_from, saver, sess, now_step)

    if step % 100 == 0:
        print('step {} \t loss = {}'.format(now_step, losses))
    summary_str = sess.run(summary_op, feed_dict)
    summary_writer.add_summary(summary_str, now_step)
    sess.run(local_init)

    ## end training
    coord.request_stop()
    coord.join(threads)
