###############
Getting started
###############

There are several options for getting started with the Lightning Pose app:

* :ref:`Duplicate Lightning Studio <lightning_studio>` is a no-install option - simply clone a cloud-based environment that comes with the app already installed. Requires creating a Lightning account.

* :ref:`Install app from github <conda_from_source>` is for installation on a local machine or a fresh Lightning Studio. This option is mostly used for development purposes.

The :ref:`bottom of this page <update_app>` also describes how to update an already-installed version of the app.


.. _lightning_studio:

Duplicate Lightning Studio with pre-installed app
-------------------------------------------------

First, you will need to `create a Lightning account <https://lightning.ai/>`_
if you have not already signed up.

Once you have set up your Lightning account, follow
`this link <https://lightning.ai/themattinthehatt/studios/lightning-pose-app?section=all>`_
to the Lightning Pose App Studio.
When you click the **Get** button you will be taken to a Lightning Studio environment with access
to a terminal, VSCode IDE, Jupyter IDE, and more.
The app and all dependencies are already installed.

Once you have opened the Studio environment you can proceed to
:ref:`Using the app <using_the_app>`
to learn how to launch the app and navigate the various tabs.

.. _conda_from_source:

Install app from github
-----------------------

.. warning::

    The Lightning Pose app currently requires a Linux operating system, at least one GPU,
    and **CUDA 11.0-12.x** installed.
    This is a requirement for **NVIDIA DALI**.

Step 1: Install ffmpeg
**********************

First, check to see if you have ``ffmpeg`` installed by typing the following in the terminal:

.. code-block:: console

    ffmpeg -version

If not, install:

.. code-block:: console

    sudo apt install ffmpeg

Step 2: Create a conda environment
**********************************

.. note::

    If you are installing the software in a Lightning Studio, you can skip this step.

First, `install anaconda <https://docs.anaconda.com/free/anaconda/install/index.html>`_.

Next, create and activate a conda environment:

.. code-block:: console

    conda create --yes --name lai python=3.10
    conda activate lai

Step 3: Download the app from github
************************************

Move to your home directory (or wherever you would like to download the code)
and install the ``Pose-app`` repo:

.. code-block:: console

    cd ~
    git clone --recursive https://github.com/Lightning-Universe/Pose-app
    cd Pose-app
    pip install -e .

While still inside of the ``Pose-app`` directory, install the lightning pose package
(with frozen requirements).

.. code-block:: console

    pip install -r requirements_litpose.txt -e lightning-pose

If you are using Ubuntu 22.04 or newer (or using a Lightning Studio),
you'll need an additional update for the FiftyOne package:

.. code-block:: console

    pip install fiftyone-db-ubuntu2204


.. _update_app:

Update an already-installed app
-------------------------------

First, move into the ``Pose-app`` directory and pull the newest updates for 
``Pose-app`` and ``lightning-pose`` from github:

.. code-block:: console

    cd Pose-app
    git pull --recurse-submodules

The ``git pull`` command will provide the most up-to-date app features.
You might want to do this every month or so.

Updating all of the app dependencies requires a few more steps, but generally does not need happen
nearly as often.
If you cloned/installed the app before July 2024, we recommend following this procedure.

Update the ``lightning-pose`` dependencies:

.. code-block:: console

    pip install -U -r requirements_litpose.txt
    pip install -U torchaudio

.. note::

    You will see various dependency issue errors arise after installing the packages in
    ``requirements_litpose.txt`` - this is fine, we will deal with these below.

Next, reinstall ``lightning-pose`` locally:

.. code-block:: console

    pip install -e lightning-pose

Update the ``Pose-app`` dependencies and reinstall locally:

.. code-block:: console

    pip install -U -e .

Finally, update the Lightning Studio SDK package:

.. code-block:: console

    pip install -U lightning-sdk
