.. _guide_inference_on_large_videos:

###############################
Run inference on large datasets
###############################

.. youtube:: NvVUsaRon_E?si=pOm_Jt3gki5pcCCM
    :align: center

Once you are satisfied with your model's performance, you will often want to use that model to
run inference on a large number of videos.
While the app allows you to upload videos through the browser and run inference, this is not the
most efficient way to process a large number of videos (or even a small number of very large videos).
To do so, we recommend using a provided script that is called from the command line.

Step 1: Upload data
===================

Uploading large videos directly into the app is inefficient due to the way the Streamlit widget
handles data in the browser. Instead, we recommend the following:

**If using a local workstation**:

Copy your videos to the directory ``Pose-app/data/<project_name>/videos_infer``.

.. note::

    It is good practice to make *copies* of your videos into the Lightning Pose project, and keep
    the originals saved elsewhere.

**If using a Lightning Studio**:

First, in the file browser on the left-hand side of the Studio verify if you have a directory named
``Pose-app/data/<project_name>/videos_infer``.
If not, right-click on the ``<project_name>`` directory and select ``New Folder...``.
Name the new folder ``videos_infer``.

Next, you will need to create a temporary file in this directory so that it will sync with your
Lightning Drive filesystem (empty directories are not synced).

Right click on ``videos_infer``, select ``New File...``, and name your file ``tmp.txt``.

Now, click the Drive button on the right-hand menu in the Studio:

.. image:: https://imgur.com/HQpoqth.png

You can then navigate to the directory
``this_studio > Pose-app > data > <project_name>``.
If you do not see ``videos_infer`` you can re-sync your Drive by clicking the sync button in the
upper right corner:

.. image:: https://imgur.com/qC3h2YI.png
    :width: 300

Once ``videos_infer`` appears, click on this directory, then click the "Add data" button in the
upper right corner:

.. image:: https://imgur.com/MVOiyUb.png
    :width: 300

Now you can drag and drop your videos to upload to Lightning, using a more efficient uploading
mechanism than the Streamlit uploader available in the app.

If your videos are very large (>1GB) we recommend uploading one video at a time.


Step 2: Run inference from the command line
===========================================

Now that the videos you wish to run inference on are all in one place, you can run a script
that will loop over all videos and perform inference with a specific model.
The bash script is provided in ``Pose-app/scripts``.
The bash script requires 5 positional arguments:

#. absolute path to the model directory (usually in YY-MM-DD/HH-MM-SS format)
#. absolute path to the directory of videos to be processed
   (note that *every* video in the directory will be processed; if you'd like to run inference on
   a subset of videos you may first save the new videos in a temporary directory, run this script
   (pointing it to the temporary directory),
   then move the videos into ``videos_infer``)
#. compute metrics flag; if true, compute metrics like temporal norm and pca errors
   (if these options were selected during project creation)
#. label full video flag; if true, save a copy of the original video with model predictions overlaid
   (caution: if your original videos are large these will also take up lots of storage!)
#. label video snippet flag: if true, create a 30 second video snippet with the highest motion
   energy with model predictions overlaid

Here is an example when running from a Lightning Studio (inside the ``Pose-app`` directory):

.. code-block:: console

    bash scripts/process_videos.sh /this_studio/Pose-app/data/<project_name>/2024-03-28/23-29-04_super-0 /this_studio/Pose-app/data/<project_name>/videos_infer true true true

This call will use the model located in ``2024-03-28/23-29-04_super-0`` to run inference on all videos
in ``<project_name>/videos_infer``,
compute metrics on each video,
and create both a full labeled video and a labeled video snippet.

If you wish to run inference with an *ensemble* of models, and optionally run EKS on the outputs,
we provide another script name ``process_videos_ensemble.sh``.

This script requires 7 positional arguments.
The first 5 arguments are the same as above, except the first argument is a *list* of mmodel
directories, separated by colons (``:``); see the example below.
The final two arguments are:

* compute eks flag: if ``True``, run EKS on the outputs of the listed models.
  The previous three flags for computing metrics and label videos will be applied to the EKS
  outputs as well, where applicable.
* eks save directory: absolute path to directory to save eks outputs

Here is an example using two networks for the ensemble when running from a Lightning Studio:

.. code-block:: console

    bash scripts/process_videos_ensemble.sh /this_studio/Pose-app/data/<project_name>/2024-03-28/23-29-04_super-0:/this_studio/Pose-app/data/<project_name>/2024-03-28/23-29-04_super-1 /this_studio/Pose-app/data/<project_name>/videos_infer true true true true /this_studio/Pose-app/data/<project_name>/eks_outputs
