# Copyright 2021 The AI-Medic\Cell-Classification Authors. All Rights Reserved.
# License stuff will be written here later...

"""
Set main parameters before execution
"""

from argparse import ArgumentParser


def main_args():
    """
    Parameters
    ----------
    None

    Returns
    -------
    Returns some hyper-parameters which are common among all the models.
    """
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, default='unet', help='model name.', required=True)
    parser.add_argument('--epochs', type=int, default=3, help='define number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='define batch size')
    parser.add_argument('--val_batch_size', type=int, default=8, help='define validation batch size')
    parser.add_argument('--train_dir', type=str, default='./dataset/train/', help='train directory')
    parser.add_argument('--test_dir', type=str, default='./dataset/test/', help='test directory')
    parser.add_argument('--val_dir', type=str, default='./dataset/val/', help='validation directory')
    parser.add_argument('--mlflow-source', type=str, default='./mlruns', help='The mlflow direcotry')
    parser.add_argument('--run-ngrok', dest='run_ngrok', action='store_true', help="Run ngrok for colab!")
    parser.add_argument('--no-run-ngrok', dest='run_ngrok', action='store_false', help="Don't run ngrok for colab!")
    parser.add_argument('--augmentation_p', type=float, default=0.5, help='augmentation probability')
    parser.add_argument('--shuffle', dest='shuffle', action='store_true', help='shuffle training && validation dataset')
    parser.add_argument('--no_shuffle', dest='shuffle', action='store_false',
                        help='do not shuffle training && validation dataset')
    parser.add_argument('--categorical', dest='categorical', action='store_true',
                        help='convert dataset to One Hot form')
    parser.add_argument('--no_categorical', dest='categorical', action='store_false',
                        help='no not convert dataset to One Hot form')
    parser.set_defaults(shuffle=True, categorical=False)
    parser.add_argument('--multiprocessing', dest='multiprocessing', action='store_true',
                        help="Run fit with multi-processing")
    parser.add_argument('--no-multiprocessing', dest='multiprocessing', action='store_false',
                        help="Run fit without multi-processing")
    parser.set_defaults(multiprocessing=True)
    parser.add_argument('--workers', type=int, default=4, help="number of workers for model.fit")
    parser.add_argument('--early_stopping_p', type=int, default=2, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--warmup_lr_scheduler', dest='warmup_lr_scheduler', action='store_true',
                        help="use warmup_lr_scheduler(else using reduce_lr callback)")
    parser.set_defaults(warmup_lr_scheduler=False)
    parser.add_argument('--reduce_lr_scheduler', dest='reduce_lr_scheduler', action='store_true',
                        help="use reduce_lr_scheduler")
    parser.set_defaults(reduce_lr_scheduler=False)
    parser.add_argument('--cosine_decay_lr_scheduler', dest='cosine_decay_lr_scheduler', action='store_true',
                        help="use cosine_decay_lr_scheduler(else using reduce_lr callback)")
    parser.set_defaults(cosine_decay_lr_scheduler=False)
    parser.add_argument('--learning_rate_base', type=float, default=0.1,
                        help='maximum lr that warmup lr scheduler will reach to')
    parser.add_argument('--warmup_epoch', type=int, default=3,
                        help='number of epoch to increase the lr to learning_rate_base')
    parser.add_argument('--plateau_min_lr', type=float, default=0.0001, help='lower bound on the learning rate')
    parser.add_argument('--tb_dir', type=str, default='./logs', help='The TensorBoard directory')
    parser.add_argument('--cosine_decay_alpha', type=float, default=0.001,
                        help='Minimum learning rate value as a fraction of initial_learning_rate.')
    parser.add_argument('--cosine_decay_initial_learning_rate', type=float, default=0.1,
                        help='CosineDecay initial learning_rate')

    return parser
