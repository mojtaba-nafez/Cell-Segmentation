# Copyright 2021 The AI-Medic\Cell-Segmentation Authors. All Rights Reserved.
# License stuff will be written here later...

"""
This file contains two modules:
    -   MLFlowLogger
    -   MLFlowHandler
"""

import os
from os.path import join
import mlflow
import mlflow.tensorflow
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend as K
import datetime


class MLFlowLogger(Callback):
    """
    This module contains the following functions:
        -   on_epoch_end
        -   on_train_begin
    """

    def __init__(self, mlflow, warmup):
        super(MLFlowLogger, self).__init__()
        self.mlflow = mlflow
        self.global_step = 0
        self.warmup=warmup

    def on_epoch_end(self, epoch, logs=None):
        """
        Use to log metrics of model on each epoch end.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.mlflow.log_metric('train acc', logs['acc'], step=epoch)
        self.mlflow.log_metric('val acc', logs['val_acc'], step=epoch)
        self.mlflow.log_metric('train loss', logs['loss'], step=epoch)
        self.mlflow.log_metric('val_loss', logs['val_loss'], step=epoch)
        self.mlflow.log_metric('train dice_coef', logs['dice_coef'], step=epoch)
        self.mlflow.log_metric('val dice_coef', logs['val_dice_coef'], step=epoch)

    def on_batch_end(self, batch, logs=None):
        if self.warmup:
            self.global_step += 1
            try:
                lr = round(float(self.model.optimizer.lr(self.global_step).numpy()), 6)
            except:
                lr = K.get_value(self.model.optimizer.lr)
                lr = round(float(lr), 6)
            self.mlflow.log_metric('lr', lr, step=self.global_step)

    def on_train_begin(self, logs=None):
        """
        Use to log parameters of model on beginning of each train.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.mlflow.log_param('optimizer_name', type(self.model.optimizer).__name__)
        self.mlflow.log_param('loss_function', type(self.model.loss).__name__)


class MLFlowHandler:
    """
    This module contains the following functions:
        -   colab_ngrok
        -   start_run
        -   end_run
        -   add_figure
        -   add_report
        -   add_weight
    """

    def __init__(self, model_name, run_name, warmup, mlflow_source='./mlruns', run_ngrok=True):
        """
        Parameters
        ----------
        model_name: str ; name of used model.
        run_name: srt ; name of run.
        mlflow_source: path to save mlflow logs in it; default is './mlruns'
        run_ngrok: bool ; pass False if you are running codes locally.

        Returns
        -------
        None
        """
        self.mlflow = mlflow
        self.run_ngrok = run_ngrok
        self.mlflow_source = mlflow_source
        self.mlflow.set_tracking_uri(mlflow_source)
        self.mlflow_logger = MLFlowLogger(mlflow, warmup)
        if run_name is not None:
            self.run_name = run_name
        else:
            self.run_name = model_name + "_" + str(datetime.datetime.now().date()) + "_" + str(
                datetime.datetime.now().time())
        self.model_name = model_name

    @staticmethod
    def colab_ngrok(mlflow_source):
        """
        Use to help mlflow ui run on colab.

        Parameters
        ----------
        mlflow_source: path to save mlflow logs in it; default is './mlruns'

        Returns
        -------
        URL link: to open MLflow tracking ui
        """
        from pyngrok import ngrok

        # run tracking UI in the background
        os.system(f"cd {os.path.split(mlflow_source)[0]} && mlflow ui --port 5000 &")

        ngrok.kill()

        # Setting the authtoken (optional)
        # Get your authtoken from https://dashboard.ngrok.com/auth
        NGROK_AUTH_TOKEN = ""
        ngrok.set_auth_token(NGROK_AUTH_TOKEN)

        # Open an HTTPs tunnel on port 5000 for http://localhost:5000
        ngrok_tunnel = ngrok.connect(addr="5000", proto="http", bind_tls=True)
        print("MLflow Tracking UI:", ngrok_tunnel.public_url)

    def start_run(self, args):
        """
        Use to start a new run; it saves each run with its date and time.

        Parameters
        ----------
        *args
        *kwargs

        Returns
        -------
        None
        """
        self.mlflow.set_experiment(str(self.model_name))
        experiment = self.mlflow.get_experiment_by_name(str(self.model_name))
        ex_id = experiment.experiment_id
        self.mlflow.start_run(run_name=self.run_name, experiment_id=str(ex_id))
        command_line = "python train.py " + ' '.join([f"--{k} {v}" for k, v in args._get_kwargs()])
        self.mlflow.set_tag("command_line line", command_line)
        for k, v in args._get_kwargs():
            self.mlflow.log_param(k, v)
        if self.run_ngrok:
            self.colab_ngrok(self.mlflow_source)

    def end_run(self, model_path=None):
        """
        Use to end a run; it also saves model weights in a given path.

        Parameters
        ----------
        model_path: str ; a path to save weights; default is None.

        Returns
        -------
        None
        """
        if model_path is not None:
            self.add_weight(model_path, )
        self.mlflow.end_run()

    def add_figure(self, figure, artifact_path):
        """
        Use to log a figure as a mlflow artifact.

        Parameters
        ----------
        figure: the figure to log.
        artifact_path: the run-relative artifact file path to save figure.

        Returns
        -------
        None
        """
        self.mlflow.log_figure(figure, artifact_path)

    def add_report(self, report, artifact_path):
        """
        Use to log a text report as a mlflow artifact.

        Parameters
        ----------
        report: string containing text to log.
        artifact_path: the run-relative artifact file path to save report.

        Returns
        -------
        None
        """
        self.mlflow.log_text(report, artifact_path)

    def add_weight(self, weight_path, artifact_path=None):
        """
        Use to log a model weight file as a mlflow artifact.

        Parameters
        ----------
        weight_path: local path to the weights file.
        artifact_path: if provided, the directory in artifact_uri to log to.

        Returns
        -------
        None
        """
        if artifact_path is None:
            weight_name = os.path.split(weight_path)[-1]
            artifact_path = join('models', weight_name)
        print('model saved: ', mlflow.get_artifact_uri() + '/' + artifact_path + weight_path.replace('weights', ''))

        self.mlflow.log_artifact(weight_path, artifact_path)
