.. _tab_train_infer:

###########
Train/Infer
###########

This tab is the interface for training models and running inference on new videos.

.. image:: https://imgur.com/uU6LbBZ.png

The left side-bar displays your current labeling progress, and contains a drop-down menu showing
all previously trained models.
The "Train Networks" and "Predict on New Videos" columns are for training and inference,
and detailed below.

Train Networks
==============

Training options
----------------

From the drop-down "Change Defaults" menu,
optionally change the max training epochs and the types of unsupervised losses used for the
semi-supervised models.

.. .. image:: https://imgur.com/LiylXxc.png
    :width: 400

The PCA Multiview option will only appear if your data have more than one view;
the Pose PCA option will only appear if you selected keypoints for the Pose PCA loss during
project creation.

Video handling options
----------------------

After each model has completed training, you can choose to automatically run inference on the set
of videos uploaded for labeling:

* **Do not run inference**: self-explanatory
* **Run inference on videos**: runs on all videos previously uploaded in the "Extract Frames" tab
* **Run inference on videos and make labeled movie**: runs inference and then creates a labeled video with model predictions overlaid on the frames.

.. .. image:: https://imgur.com/8UBY5y9.png
    :width: 400

.. warning::

    Video traces will not be available in the :ref:`Video Diagnostics <tab_video_diagnostics>` tab
    if you choose "Do not run inference".

Select models to train
----------------------

There are currently 4 options to choose from:

* **Supervised**: fully supervised baseline
* **Semi-supervised**: uses labeled frames plus unlabeled video data for training
* **Supervised context**: supervised model with temporal context frames
* **Semi-supervised context**: semi-supervised model plus temporal context frames

.. .. image:: https://imgur.com/x1MdTSk.png
    :width: 400

.. note::

    If you uploaded a DLC project or are using ``demo_app.py`` you will not see the context options.

Train models
------------

Click "Train models" to launch sequential model training - parallel model training coming soon!
A set of progress bars will appear below, one for each model.

.. image:: https://imgur.com/Atekosg.png
    :width: 400

Afer training is complete for each model, inference is run on each video if selected in the
"Video handling options" above.
The progress bar will reset and display inference progress for each video.

Once training is complete for all models you will see
"Training complete; see diagnostics in the following tabs" in green.

.. _tab_train_infer__infer:

Predict on New Videos
=====================

First, select the model you would like to use for inference from the drop-down menu.
Then, drag and drop video file(s) using the provided interface.
You will see an upload progress bar.

.. image:: https://imgur.com/MXHq8hx.png
    :width: 400

You may also choose to create videos overlaid with predictions from the model.

.. image:: https://imgur.com/RBqSZTF.png
    :width: 300

The option "Save labeled video (30 second clip)" will find the 30 second portion of the video
with the highest motion energy in the model predictions.
This is a good option if you want to quickly get a sense of how well the model is performing.

The option "Save labeled video (full video)" will plot predictions for the duration of the entire
video.
This is a good option if you want to search over longer or more diverse periods of the video.

If you check one or both boxes, you will be able to view the resulting videos directly in the app
in the :ref:`"Video Player" tab <tab_video_player>`.

Click "Run inference" once the video uploads are complete,
and another set of progress bars will appear.

.. image:: https://imgur.com/rK2d7ph.png
    :width: 400

Once inference is complete for all videos you will see the
"waiting for existing inference to finish" warning disappear.

See :ref:`Accessing your data <directory_structure>` for the location of inference results.
