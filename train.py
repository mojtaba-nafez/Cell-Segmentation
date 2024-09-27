# Copyright 2021 The AI-Medic\Cell-Segmentation Authors. All Rights Reserved.
# License stuff will be written here later...

"""
This is the main module of this project ; here we difine a function to start training process.
"""

import sys
import tensorflow as tf
from utils.metrics import *
import mlflow
from datetime import datetime
from os.path import join
from models import load_model
from params import get_args
from data.data_loader import get_loader
from tensorflow.keras.optimizers import Adam
from utils.callbacks import get_callbacks
from utils.mlflow_handler import MLFlowHandler
from utils.plots import get_plots
from utils.utils import get_gpu_grower

get_gpu_grower()


def train():
    """
    this function is to start trainning process of the desired model.
    it should run in below format:
        python train.py -[option] [value] --[option] [value] ...
    to see available options and arguments see Readme.md file.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    model_name = sys.argv[2]
    print(f"Chosen Model: {model_name}")
    args = get_args(model_name)
    print(f"Arguments: {args}")

    id_ = model_name + "_" + str(datetime.now().date()) + "_" + str(datetime.now().time())
    weight_path = join('weights', id_) + ".h5"
    mlflow_handler = MLFlowHandler(model_name=model_name,
                                   warmup=args.warmup_lr_scheduler,
                                   run_name=id_,
                                   mlflow_source=args.mlflow_source,
                                   run_ngrok=args.run_ngrok)
    mlflow_handler.start_run(args)

    train_loader, val_loader, test_loader = get_loader(batch_size=args.batch_size,
                                                       train_dir=args.train_dir,
                                                       test_dir=args.test_dir,
                                                       val_dir=args.val_dir,
                                                       augmentation_p=args.augmentation_p,
                                                       shuffle=args.shuffle,
                                                       categorical=args.categorical)

    if 'deeplabV3plus_resnet' in args.model:
        model = load_model(args.model, input_shape=args.input_shape, resnet_trainable=args.resnet_trainable,
                           layer_trainable_trainable=args.layer_trainable_trainable)
    elif 'unet_pretrain_cbam' in args.model:
        model = load_model(args.model, input_shape=args.input_shape, resnet_trainable=args.resnet_trainable)
    else:
        model = load_model(args.model, input_shape=args.input_shape)

    print("Loading Model is Done!")

    if args.pretrain:
        model.load_weights(args.pretrain)
        print('pretrain weights loaded.')
    model.summary()

    checkpoint, warmup, early_stopping, tensorboard, reduce_lr = get_callbacks(model_path=weight_path,
                                                                               early_stopping_p=args.early_stopping_p,
                                                                               tb_log_dir=args.tb_dir,
                                                                               save_weights_only=True,
                                                                               epochs=args.epochs,
                                                                               sample_count=len(
                                                                                   train_loader) * args.batch_size,
                                                                               batch_size=args.batch_size,
                                                                               warmup_epoch=args.warmup_epoch,
                                                                               learning_rate_base=args.learning_rate_base,
                                                                               model_name=model_name,
                                                                               plateau_min_lr=1e-6)
    callbacks = [tensorboard, checkpoint, early_stopping, mlflow_handler.mlflow_logger]
    if args.warmup_lr_scheduler:
        callbacks.append(warmup)
        opt = Adam(learning_rate=args.learning_rate)
        print('activate warmup_lr_scheduler.')
    elif args.cosine_decay_lr_scheduler:
        learning_rate_scheduler = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=args.cosine_decay_initial_learning_rate, decay_steps=args.epochs,
            alpha=args.cosine_decay_alpha, name='None')
        opt = Adam(learning_rate=learning_rate_scheduler)
        print('activate cosine_decay_lr_scheduler.')
    elif args.reduce_lr_scheduler:
        callbacks.append(reduce_lr)
        opt = Adam(learning_rate=args.learning_rate)
        print('activate reduce_lr_scheduler.')
    else:
        opt = Adam(learning_rate=args.learning_rate)
        print(f'No lr_scheduler, learning rate fixed at: {args.learning_rate}')
    loss = bce_dice_loss
    model.compile(optimizer=opt, loss=loss, metrics=['acc', dice_coef])
    model.fit(x=train_loader,
              batch_size=args.batch_size,
              epochs=args.epochs,
              validation_data=val_loader,
              validation_batch_size=args.val_batch_size,
              callbacks=callbacks,
              use_multiprocessing=args.multiprocessing,
              workers=args.workers,
              )

    print("Training Model is Done!")

    get_plots(model, val_loader, mlflow_handler)
    mlflow_handler.end_run(weight_path)


if __name__ == '__main__':
    train()
