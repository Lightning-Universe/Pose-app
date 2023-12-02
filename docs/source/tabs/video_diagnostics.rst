.. _tab_video_diagnostics:

#################
Video Diagnostics
#################

This tab compares predictions and various metrics computed on unlabeled videos across multiple
models.

First, select models to compare and a specific video on the left panel.

.. warning::

    Video traces will not be available if you chose "Do not run inference" in the Train/Infer tab.

Models can be renamed using the text fields in the left panel.

.. warning::

    Model selections and custom names will be reset when you navigate away from this tab.

Video statistics
================

Select a metric and keypoint; the plot will aggregate the metric over all video frames for each
model.
You can also select the plot style and axis scale.

.. image:: https://imgur.com/ZlGFL79.png

Video traces
============

Select a subset of models and a keypoint.
The trace plot shows various metrics up top (temporal norm, pca errors),
followed by the (x, y) predictions and their confidences.

.. image:: https://imgur.com/RBRsUg0.png
