# coding: utf-8
import argparse
import os
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '10'
time.sleep(2)

import train
import val


def start(args):
    if args.operate == 'train':
        print("进入训练阶段")
        train.train(args)
    elif args.operate == 'val':
        print("进入验证阶段")
        val.val(args)


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """

    BATCH_SIZE = 4
    # DATA_DIRECTORY = '/data/bo718.wang/mpii/' # remote server address
    DATA_DIRECTORY = '../dataset/mpii/'
    IMG_SIZE = [256, 256]
    LEARNING_RATE = 2.5e-4
    MOMENTUM = 0.9
    NUM_STEPS = 40000
    POWER = 0.9  # 原来我设置的是0.99，v3上面给的是0.9
    RANDOM_SEED = 1234
    OPERATE = 'train'
    SAVE_NUM_IMAGES = 1
    SAVE_PRED_EVERY = 10000
    WEIGHT_DECAY = 0.0005
    RESTORE_FROM = './weights/'
    LOG_DIR = './tblogs/%s/' % OPERATE

    parser = argparse.ArgumentParser(description="DeepLab_ResNet Network")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--data_dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--is_training", action="store_true",
                        help="Whether to updates the running means and variances during the training.")
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--not_restore_last", action="store_true",
                        help="Whether to not restore last (FC) layers.")
    parser.add_argument("--num_steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--img_size", type=list, default=IMG_SIZE,
                        help="The final size of image preprocessing.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random_seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--restore_from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--save_num_images", type=int, default=SAVE_NUM_IMAGES,
                        help="How many images to save.")
    parser.add_argument("--save_pred_every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--log_dir", type=str, default=LOG_DIR,
                        help="Where to save tensorboard log of the model.")
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--operate", type=str, default=OPERATE,
                        help="kind of the operation")
    parser.add_argument("--random_mirror", type=bool, default=True,
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random_scale", type=bool, default=True,
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random_crop", type=bool, default=True,
                        help="Whether to randomly scale the inputs during the training.")

    return parser.parse_args()


if __name__ == '__main__':
    start(get_arguments())
