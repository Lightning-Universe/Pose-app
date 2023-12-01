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
