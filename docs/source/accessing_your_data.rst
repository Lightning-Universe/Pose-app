.. _directory_structure:

###################
Accessing your data
###################

Lightning Pose project structure
================================

Data for the project named ``<PROJ_NAME>`` will be stored in a directory with the following structure:

.. code-block::

    ~/Pose-app/.shared/data/<PROJ_NAME>
      ├── labeled-data/
      ├── models/
      ├── videos/
      ├── videos_infer/
      ├── videos_tmp/
      ├── CollectedData.csv
      ├── label_studio_config.xml
      ├── label_studio_metadata.yaml
      ├── label_studio_tasks.pkl
      └── model_config_<PROJ_NAME>.yaml

* ``labeled-data/``: contains one subdirectory for each video used during labeling; each of these subdirectories contains images named ``img<n>.png`` that correspond to the labels. The subdirectory also contains images preceding and succeeding each labeled frame for training context models.

* ``models/``: contains one subdirectory for each model trained, with the naming convention ``YYYY-MM-DD/HH-MM-SS``. See below for more information on the files inside each model directory.

* ``videos/``: videos uploaded for labeling. When training models, after training completion inference is automatically run on each of these videos, and the results are saved in the model-specific directory

* ``videos_infer/``: videos uploaded for inference

* ``videos_tmp/``: temporary directory that stores original videos before they are automatically reformatted to be compatible with DALI video readers (h.264 encoding, yuv420p pixel format). ORIGINAL VIDEOS ARE NOT SAVED.

* ``CollectedData.csv``: hand labels in human-readable format.

* ``label_studio_config.xml``: config info for label studio, including keypoint names and labeler instructions.

* ``label_studio_metadata.yaml``: info on number of labeled and total tasks.

* ``label_studio_tasks.pkl``: reformatted version of labeling tasks, also stored in label studio database.

* ``model_config_<PROJ_NAME>.yaml``: config file for pose estimation networks; contains editable information about min/max epochs, train/val splits, network architecture, etc.

Model directory structure
=========================

Each model and its associated outputs will be stored in a directory with the following structure:

.. code-block::

    ~/Pose-app/.shared/data/<PROJ_NAME>/models/YYYY-MM-DD/HH-MM-SS
      ├── tb_logs/
      ├── video_preds/
      ├── video_preds_infer/
      ├── config.yaml
      ├── predictions.csv
      ├── predictions_pca_multiview_error.csv
      ├── predictions_pca_singleview_error.csv
      └── predictions_pixel_error.csv

* ``tb_logs/``: model weights
* ``video_preds/``: predictions and metrics from all videos in ``videos`` directory of the project
* ``video_preds_infer/``: predictions and metrics from all videos in ``videos_infer`` directory of the project
* ``config.yaml``: copy of the config file used to train the model
* ``predictions.csv``: predictions on labeled data in ``CollectedData.csv``
* ``predictions_pca_multiview_error.csv``: if applicable, pca multiview reprojection error for labeled data
* ``predictions_pca_singleview_error.csv``: if applicable, pca singleview reprojection error for labeled data
* ``predictions_pixel_error.csv``: pixel error for labeled data
