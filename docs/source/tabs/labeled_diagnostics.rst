.. _tab_labeled_diagnostics:

###################
Image Diagnostics
###################

This tab compares various metrics on labeled data across multiple models.

.. image:: https://imgur.com/9h9cmTl.png

Select models on the left panel.
Models can be renamed using the text fields in the left panel.

.. warning::

    Model selections and custom names will be reset when you navigate away from this tab.

Select data to plot
-------------------

.. .. image:: https://imgur.com/8C7JShk.png

Filter results using various criteria:

* **Keypoint**: mean computes metric average across all keypoints on each frame
* **Metric**: choose from available metrics like pixel error and confidence
* **Train/Val/Test**: data split to plot

Compare multiple models
-----------------------

Plot selected metric/keypoint across all models:

* **Plot style**: choose from box, violin, or strip
* **Metric threshold**: ignore any values below this threshold
* **Y-axis scale**: choose from log or linear

Compare two models
------------------

A scatter plot provides a more in-depth comparison between two models.
Hover over individual points to see frame information.
