.. _using_the_app:

#############
Using the app
#############

We provide three different apps:

* ``demo_app.py``: using provided example data, train and evaluate pose estimation models
* ``labeling_app.py``: stand-alone labeling app where you can upload videos, extract frames, and annotate keypoints on extracted frames using LabelStudio
* ``app.py``: full app that includes labeling, training, and evaluation

**Launch the app**

Once you have opened the Lightning Pose App Studio, you will see the following:

.. image:: https://imgur.com/N351izy.png
    :width: 400

Make sure you are :ref:`connected to a GPU machine <faq_change_machine>` for non-labeling tasks.

First, open a terminal by using the drop-down menu at the top left and select
``Terminal > New Terminal``.

.. image:: https://imgur.com/ZqhpAhE.png
    :width: 400

To launch an app from the terminal, first move to the ``Pose-app`` directory:

.. code-block:: console

    cd Pose-app

and run:

.. code-block:: console

    lightning run app <app_name>.py

.. note::

    You may get a message alerting you that a new version of Lightning is available.
    There is a possibility that upgrading will cause breaking changes, and you can always decline.
    If you **do** upgrade and an error occurs, please notify the maintainers by
    `raising an issue <https://github.com/Lightning-Universe/Pose-app/issues>`_.

Once the app launches you will begin to see printouts in the terminal.
Navigate to the app output by clicking on the "port" plugin on the right-hand tool bar
(see image below), type in port number 7501, and click "update".

.. image:: https://imgur.com/0XxDcpZ.png
    :width: 400

IMPORTANT! You will need to click on the "Share" button, which will open the app in a separate
browser window.
This is crucial to getting all of the components visualized properly.
In the original browser page you can return to the terminal to see printouts by clicking on the
VS Code icon in the right-hand tool bar.

.. image:: https://imgur.com/lINajyE.png
    :width: 200

Click on the links below to find more information about specific tabs;
remember that ``demo_app.py`` and ``labeleing_app.py`` only utilize a subset of the tabs.

.. toctree::
   :maxdepth: 1
   :caption: App tabs:

   tabs/manage_project
   tabs/extract_frames
   tabs/label_frames
   tabs/train_infer
   tabs/train_status
   tabs/labeled_diagnostics
   tabs/video_diagnostics
   tabs/prepare_fiftyone
   tabs/fiftyone

**Close the app**

**To shut down the app**: return to the terminal/VS Code view of the Studio by clicking on
the appropriate icon in the right-hand tool bar (see image below), and type "ctrl+c" in the
terminal.

.. image:: https://imgur.com/lINajyE.png
    :width: 200

**To shut down the studio**: click on the compute icon in the right-hand tool bar and then click the power button.

.. image:: https://imgur.com/jsygRpO.png
    :width: 400
