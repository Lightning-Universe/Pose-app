.. _tab_train_infer:

###########
Train/Infer
###########

This tab is the interface for training models and running inference on new videos.

.. image:: https://imgur.com/GXhvqXI.png

The left side-bar displays your current labeling progress, and contains a drop-down menu showing
all existing models.
The "Train Networks" and "Predict on New Videos" columns are for training and inference,
and detailed below.

Train Networks
==============

Training options
----------------

Optionally change the max training epochs and the types of unsupervised losses used for the
semi-supervised models.

.. image:: https://imgur.com/LiylXxc.png
    :width: 400

Video handling options
----------------------

After each model has completed training, you can choose to automatically run inference on the set
of videos uploaded for labeling:

* Do not run inference: self-explanatory
* Run inference on videos: runs on all videos previously uploaded in the "Extract Frames" tab
* Run inference on videos and make labeled movie: runs inference and then creates a labeled video with model predictions overlaid on the frames.

.. image:: https://imgur.com/8UBY5y9.png
    :width: 400

.. warning::

    Video traces will not be available in the :ref:`Video Diagnostics <tab_video_diagnostics>` tab
    if you choose "Do not run inference".

Select models to train
----------------------

There are currently 4 options to choose from:

* Supervised: fully supervised baseline
* Semi-supervised: uses labeled frames plus unlabeled video data for training
* Supervised context: supervised model with temporal context frames
* Semi-supervised context: semi-supervised model plus temporal context frames

.. image:: https://imgur.com/x1MdTSk.png
    :width: 400

.. note::

    If you uploaded a DLC project you will not see the context options.

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

Predict on New Videos
=====================

First, select the model you would like to use for inference from the drop-down menu.
Then, drag and drop video file(s) using the provided interface.
You will see an upload progress bar.

.. image:: https://imgur.com/MXHq8hx.png
    :width: 400

Click "Run inference", and another set of progress bars will appear.
After inference is complete for each video a small snippet is extracted
(during the period of highest motion energy)
and a video of raw frames overlaid with model predictions is created for diagnostic purposes.

.. image:: https://imgur.com/rK2d7ph.png
    :width: 400

Once inference is complete for all videos you will see the
"waiting for existing inference to finish" warning disappear.

See :ref:`Accessing your data <directory_structure>` for the location of inference results.
