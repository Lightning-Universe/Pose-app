.. _tab_train_status:

############
Train Status
############


This tab displays a dashboard where you can inspect the individual losses for each model throughout
training
(if and when they apply; for example, the unsupervised losses will only be reported for the
semi-supervised model).

.. image:: https://imgur.com/vbbhGKl.png

Important traces are "train_supervised_rmse" (root mean square error between true and predicted
keypoints on training data) and "val_supervised_rmse" (rmse for validation data).

The two models we’ve trained are saved as "YYYY-MM-DD/HH-MM-SS", for example, "2023-12-01/15-30-00"
and "2023-12-01/15-30-01".
The earlier one is the supervised model, and the later one is the semi-supervised.

.. note::

    If you don't see all your models in that tab,
    hit the refresh button on the top right corner of the screen,
    and the other models should appear.

Available metrics
-----------------

The following are the important metrics for all model types 
(supervised, context, semi-supervised, etc.):

* ``train_supervised_loss``: this is the same as ``train_heatmap_mse_loss_weighted``, which is the
  mean square error (MSE) between the true and predicted heatmaps on labeled training data
* ``train_supervised_rmse``: the root mean square error (RMSE) between the true and predicted 
  (x, y) coordinates on labeled training data; scale is in pixels
* ``val_supervised_loss``: this is the same as ``val_heatmap_mse_loss_weighted``, which is the
  MSE between the true and predicted heatmaps on labeled validation data
* ``val_supervised_rmse``: the RMSE between the true and predicted (x, y) coordinates on labeled
  validation data; scale is in pixels

The following are important metrics for the semi-supervised models:

* ``train_pca_multiview_loss_weighted``: the ``train_pca_multiview_loss`` (in pixels), which 
  measures multiview consistency, multplied by the loss weight set in the configuration file.
  This metric is only computed on batches of unlabeled training data.
* ``train_pca_singleview_loss_weighted``: the ``train_pca_singleview_loss`` (in pixels), which 
  measures pose plausibility, multplied by the loss weight set in the configuration file.
  This metric is only computed on batches of unlabeled training data.
* ``train_temporal_loss_weighted``: the ``train_temporal_loss`` (in pixels), which 
  measures temporal smoothness, multplied by the loss weight set in the configuration file.
  This metric is only computed on batches of unlabeled training data.
* ``total_unsupervised_importance``: a weight on all *weighted* unsupervised losses that linearly 
  increases from 0 to 1 over 100 epochs
* ``total_loss``: weighted supervised loss (``train_heatmap_mse_loss_weighted``) plus 
  ``total_unsupervised_importance`` times the sum of all applicable weighted unsupervised losses
  