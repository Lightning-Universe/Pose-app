.. _tab_labeled_diagnostics:

###################
Labeled Diagnostics
###################

This tab compares various metrics on labeled data across multiple models.

.. image:: https://imgur.com/9h9cmTl.png

Select models on the left panel.
Models can be renamed using the text fields in the left panel.

.. warning::

    Model selections and custom names will be reset when you navigate away from this tab.

Select data to plot
-------------------

.. image:: https://imgur.com/8C7JShk.png

Filter results using various criteria:

* keypoint: mean computes metric average across all keypoints on each frame
* metric: choose from available metrics like pixel error and confidence
* train/val/test: data split to plot

Compare multiple models
-----------------------

Plot selected metric/keypoint across all models:

* plot style: choose from box, violin, or strip
* metric threshold: ignore any values below this threshold
* y-axis scale: choose from log or linear

Compare two models
------------------

A scatter plot provides a more in-depth comparison between two models.
Hover over individual points to see frame information.
