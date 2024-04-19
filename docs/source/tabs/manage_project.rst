.. _tab_manage_project:

##############
Manage Project
##############

The first tab of the app is a project manager that allows you to create, load, and delete projects.

* :ref:`Create new project <create_new_project>`
* :ref:`Create new project from source (e.g. existing DLC project) <create_new_project_from_source>`
* :ref:`Load existing project <load_existing_project>`
* :ref:`Delete existing project <delete_existing_project>`

.. _create_new_project:

Create new project
==================

.. image:: https://imgur.com/HfrAmUW.png

First, enter your project name (at least 3 characters).

After you hit "Enter", you will be asked to record the number of camera views.
Currently, more than one view is only supported if the views are fused into single frames,
or a mirror is used to create multiple views from a single camera.
In this example we have a side and bottom view of a mouse, so we enter "2".

.. image:: https://imgur.com/SENrE3W.png

Next, you will define the set of keypoints that you would like to label.
In the example below we enter four keypoint names: two bodyparts ("nose" and "tailbase")
seen from two views ("top" and "bottom").
If you are using more than one view, we recommend listing all keypoints from one view first,
then all keypoints from the next view, etc.

.. image:: https://imgur.com/m0a6TRy.png

You will then be prompted to select a subset of keypoints for the Pose PCA loss.
Including static keypoints (e.g. those marking a corner of an arena) are generally not helpful.
Also be careful to not include keypoints that are often occluded, like the tongue.
If these keypoints are included the Pose PCA loss will try to localize them even when they are
occluded, which might be unhelpful if you want to use the confidence of the outputs as a lick
detector.

.. image:: https://imgur.com/1BtsrWG.png

Finally, if you chose more than one camera view, you will select which keypoints correspond to the
same body part from different views.
This table will be filled out automatically, but check to make sure it is correct!

.. image:: https://imgur.com/0Nb7hCp.png

Click "Create project"; you will see "Request submitted".
Once the project is created the text will update to
"Proceed to the next tab to extract frames for labeling" in green,
and a new set of tabs will appear at the top of the app.

.. .. image:: https://imgur.com/J2IEZrm.png

.. _create_new_project_from_source:

Create new project from source
==============================

.. image:: https://imgur.com/499rk2a.png

.. warning::

    The app currently only supports conversion of DLC projects.
    If you have another type of project that needs conversion support (SLEAP, DPK, etc.) please
    `raise an issue <https://github.com/Lightning-Universe/Pose-app/issues>`_.

The standard DLC project directory looks like the following:

.. code-block::

    <dlc-project>
      ├── dlc-models/
      ├── labeled-data/
      ├── training-datasets/
      ├── videos/
      └── config.yaml

You will need to create a zip file of this project directory to upload to the app.
The upload process can take some time, so we recommend first creating a version of the dlc project
that **only** contains the directories ``labeled-data`` and ``videos``.
Make sure the videos are not symlinks!
Once you have created this project copy, compress it into a zip file.

.. code-block::

    <dlc-project-copy>
      ├── labeled-data/
      └── videos/

In the Lightning Pose App project manager, select "Create new project from source" and give your
project a name (can be the same as the DLC name or different).
You will then select the uploaded project format, and upload your zip file.

.. note::

    If your zip file is larger than the 200MB limit, :ref:`see the FAQ <faq_upload_limit>`.
    You may also replace many large video files with smaller video snippets for faster uploading.
    Whatever video files are in the ``videos`` directory will be used for unsupervised losses.

Once the zip file upload is complete you will need to walk through the steps covered in
:ref:`Create new project <create_new_project>` (though note the keypoint names are now provided).
Once you click "Create project" your DLC project will be successfully converted!
If you have many hundreds or thousands of labeled images in your project it may take
several minutes to upload all of the data into LabelStudio.

.. _load_existing_project:

Load existing project
=====================

.. image:: https://imgur.com/O8Jdd54.png

Enter project name; you will see a list of available projects (like 'mirror-mouse' above) -
you **must** select one of the available projects, or you will see an error message.
Once you enter the project name click "Load project".

You will see the previously entered project data appear (camera views, keypoint names, etc.).
You can then navigate to other project tabs.

.. _delete_existing_project:

Delete existing project
=======================

.. image:: https://imgur.com/aEprJF3.png

Enter a name from the list of available projects.
When you click "Delete project" all project data will be deleted.
