.. _tab_train_infer:

###########
Train/Infer
###########

.. youtube:: C551bzlyKQ8?si=eUUJ54Za4Obz9K53
    :align: center

This tab is the interface for training models and running inference on new videos.

.. image:: https://imgur.com/1lLek6E.png

The left side-bar displays your current labeling progress, and contains a drop-down menu showing
all previously trained models.
The main workflows are:

* :ref:`Train networks<tab_train_infer__train>`
* :ref:`Predict on new videos<tab_train_infer__infer>`
* :ref:`Create an ensemble of models<tab_train_infer__ensemble>`

"Train Networks" and "Predict on New Videos" columns are for training and inference,
and detailed below.

.. _tab_train_infer__train:

Train networks
==============

Training options
----------------

From the drop-down "Expand to adjust training parameters" menu,
optionally change the following:

* Max training epochs (default is 300)
* Model seed; different seeds will lead to different model outputs that are useful for ensembling.
  Enter either a single integer (e.g. ``0``) to train one model of each type (see below), or a
  list of comma-separated integers (e.g. ``0,1,2``) to train multiple models of each type.
* Losses used for the semi-supervised models.
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

Predict on new videos
=====================

First, select the model you would like to use for inference from the drop-down menu.

Select videos
-------------

You have two options for video selection:

* **Upload new**:
  upload a new video to the app.
  To do so, drag and drop the video file(s) using the provided interface.
  You will see an upload progress bar.
  If your video is larger than the 200MB default limit, see the :ref:`FAQs<faq_upload_limit>`.
* **Select previously uploaded video(s)**:
  any video previously uploaded to this tab (located in the ``videos_infer`` directory) or the
  Extract Frames tab (located in the ``videos`` directory) will be available in the drop down menu;
  you may select multiple videos.

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


.. _tab_train_infer__ensemble:

Create an ensemble of models
============================

.. youtube:: MURU9YwBwps?si=lkJi4nQi0Ds_fvf3
    :align: center

Ensembling is a classical machine learning technique that combines predictions from multiple
models to provide enhanced performance.
We offer the `Ensemble Kalman Smoother (EKS) <https://github.com/paninski-lab/eks>`_,
a Bayesian ensembling technique that combines model predictions with a latent smoothing model.

To use EKS, you must first create an ensemble of models.
Then, if you run inference using the ensemble, EKS will automatically be run on the ensemble
output.
The steps are outlined in more detail below.

Select models for ensembling
----------------------------
Select a set of previously trained models to create the ensemble.
We recommend an ensemble size of 4-5 models for a good trade-off between computational efficiency
and accuracy.
An ensemble can be composed in many ways;
one way would be to include models of the same type (supervised, semi-supervised, etc.) using
different random seeds;
another way would be to include models of different types (e.g. one supervised, one
semi-supervised, etc.); a combination of these approaches would work too!

Add ensemble name
-----------------
Give your ensemble a name. This text will be appended to the date and time to form the final
ensemble name (just like the other models), to prevent overwriting previous models/ensembles.

Create ensemble
---------------
Click the "Create ensemble" button; you will see a brief success message.
The newly-created ensemble directory will contain a text file that points to the model directories
of the individual ensemble members.

Running the Ensemble Kalman Smoother post-processor
---------------------------------------------------
Now that the ensemble has been created, you can run inference on videos.
Navigate back to the :ref:`Predict on new videos <tab_train_infer__infer>` part of this tab.
You should now see your ensemble in the drop-down menu of models.

.. note::

    If your model is not in the drop-down menu, click on the three vertical dots in the top right
    of the tab (next to the "Deploy" button) and click "Rerun".

You can now treat the ensemble as any other model: select one or more videos to run inference on,
select any video labeling options you like, and then click "Run inference".
Upon doing so you will see multiple progress bars appear, one for each model/video combination:

.. image:: https://imgur.com/dGktgCm.png
    :width: 400

Inference and labeled video creation will be skipped for any ensemble member that has already
performed these tasks.

After inference and labeled video creation are completed for each ensemble member, a new progress
bar will appear for the EKS model.
You will see the progress of the EKS fitting process, as well as the labeled video creation if you
have selected one of those options.

The outputs of EKS will be stored just like the inference outputs of a single model.
This means that you may inspect the EKS traces in the
:ref:`Video Diagnostics tab<tab_video_diagnostics>`
and view the labeled video (if you have selected one of these options) in the
:ref:`Video Player tab<tab_video_player>`.
