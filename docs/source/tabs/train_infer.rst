.. _tab_train_infer:

###########
Train/Infer
###########

This tab is the interface for training models and running inference on new videos.

.. image:: https://imgur.com/1lLek6E.png

The left side-bar displays your current labeling progress, and contains a drop-down menu showing
all previously trained models.
The "Train Networks" and "Predict on New Videos" columns are for training and inference,
and detailed below.

Train Networks
==============

Training options
----------------

From the drop-down "Expand to adjust training parameters" menu,
optionally change the max training epochs,
the model seed (different seeds will lead to different model outputs that are useful for ensembling),
and the types of unsupervised losses used for the semi-supervised models.

.. .. image:: https://imgur.com/LiylXxc.png
    :width: 400

The PCA Multiview option will only appear if your data have more than one view;
the Pose PCA option will only appear if you selected keypoints for the Pose PCA loss during
project creation.

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

    If you uploaded a DLC project or are using the example data you will not see the context options.

Train models
------------

Click "Train models" to launch sequential model training - parallel model training coming soon!
A set of progress bars will appear below, one for each model.

.. image:: https://imgur.com/Atekosg.png
    :width: 400

Once training is complete for all models you will see
"Training complete; see diagnostics in the following tabs" in green.

.. _tab_train_infer__infer:

Predict on New Videos
=====================

First, select the model you would like to use for inference from the drop-down menu.

Select videos
-------------

You have three options for video selection:

* **Upload new**:
  upload a new video to the app.
  To do so, drag and drop the video file(s) using the provided interface.
  You will see an upload progress bar.
  If your video is larger than the 200MB default limit, see the :ref:`FAQs<faq_upload_limit>`.
* **Select video(s) previously uploaded to the TRAIN/INFER tab**:
  any video previously uploaded to this tab will be available in the drop down menu; you may
  select multiple videos.
* **Select video(s) previously uploaded to the EXTRACT FRAMES tab**:
  any video previously uploaded in the EXTRACT FRAMES tab for labeling will be available in the
  drop down menu; you may select multiple videos.

Video handling options
----------------------
You may also choose to create videos overlaid with predictions from the model.

* **Save labeled video (30 second clip)**:
  label the 30 second portion of the video with the highest motion energy in the model predictions.
  This is a good option if you want to quickly get a sense of how well the model is performing.
* **Save labeled video (full video)**:
  plot predictions for the duration of the entire video.
  This is a good option if you want to search over longer or more diverse periods of the video.

If you check one or both boxes, you will be able to view the resulting videos directly in the app
in the :ref:`"Video Player" tab <tab_video_player>`.

Run inference
-------------

Click "Run inference" once the video uploads are complete,
and another set of progress bars will appear.

.. image:: https://imgur.com/xHS1D3X.png
    :width: 400

Once inference is complete for all videos you will see the
"waiting for existing inference to finish" warning disappear.

See :ref:`Accessing your data <directory_structure>` for the location of inference results.
