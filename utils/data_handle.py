# coding: utf-8
import os
import re

import tensorflow as tf


def load_weight(weight_path, saver, sess):
    try:
        cp_path = tf.train.latest_checkpoint(weight_path)
        print("加载路径：%s" % cp_path)
        saver.restore(sess, cp_path)
        print("加载之前训练的权重，继续训练")
        return int(re.search(r'-(.*)', cp_path).group(1))
    except:
        print("权重文件加载失败！")
        return


def save_weight(weight_path, saver, sess, global_step):
    if not os.path.exists(weight_path):
        os.makedirs(weight_path)
    saver.save(sess, weight_path, global_step)
