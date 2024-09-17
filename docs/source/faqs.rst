####
FAQs
####

General
-------

.. _faq_can_i_import:

.. dropdown:: Can I import a pose estimation project in another format into the app?

    We currently support conversion from DLC or SLEAP projects into Lightning Pose projects
    (if you would like support for another format,
    please `open an issue <https://github.com/Lightning-Universe/Pose-app/issues>`_).

    **For DLC**: First, make sure that all videos in your DLC project are actual files, not
    symbolic links.
    Symlinked videos in the DLC project will not be uploaded.
    Next, compress your DLC project into a zip file - you will upload this into the app later.

    **For SLEAP**: Make sure that you export both frames and keypoints in a ``.pkg.slp`` file.

    Finally, when you run the app select “Create new project from source” and upload your zip file.
    As of 07/2024 context datasets are not automatically created upon import;
    if this is a feature you would like to see, please
    `open an issue <https://github.com/Lightning-Universe/Pose-app/issues>`_.


.. _faq_upload_limit:

.. dropdown:: How do I increase the file upload size limit?

    Set the following environment variable (in units of MB);
    we recommend using videos less than 500MB for best performance.

    .. code-block:: console

        lightning_app run app <app_name.py> --env STREAMLIT_SERVER_MAX_UPLOAD_SIZE=500

    In general we recommend uploading shorter snippets of video to extract frames for labeling.
    If you are attempting to run inference a some large videos (or even a large number of shorter
    videos) we recommend
    :ref:`running inference from the command line using a provided script<guide_inference_on_large_videos>`.


.. _faq_update_app:

.. dropdown:: How do I update the app to the most recent version?

    The process is outlined at the bottom of the :ref:`Getting started page <update_app>`.


.. _faq_update_lightning_oops:

.. dropdown:: I accidentally upgraded my lightning version, what should I do?

    Have no fear, all you need to do is quit the app (ctrl + c twice),
    then do the following: uninstall lightning

    .. code-block:: console

        pip uninstall lightning

    and then update to the correct version:

    .. code-block:: console

        pip install lightning==2.2.5


.. _faq_connection_error:

.. dropdown:: What does it mean if I see a "Connection error" in one of the Streamlit tabs?

    .. image:: https://imgur.com/Ty5jJeQ.png
        :width: 400

    If you see some form of connection error like the above image, this typically means that an
    error has occurred somewhere in the app, the app has closed down, and the Streamlit tab is no
    longer connected. Navigate to the command line where you launched the app and you should see a
    more specific error message.


Lightning Studios
-----------------

.. _faq_change_machine:

.. dropdown:: How can I switch the cloud machine type (CPU to/from GPU)?

    In the upper right corner of the Lightning Studio, click on the compute icon
    (which will read ``1 T4`` if you are connected to the default T4 GPU, or ``4 CPU`` if you are
    connected to a CPU-only machine).
    Select the GPU or CPU box to see available options.
    We recommend a default CPU machine (not data prep) for labeling tasks.

    .. image:: https://imgur.com/HGtYm0g.png
        :width: 400


.. _faq_missing_port_viewer_api_builder:

.. dropdown:: What should I do if I can't find the Port Viewer and API Builder plugins?

    If you are using an older version of the studio or for any other reason cannot find the Port
    Viewer or API Builder plugins, you will need to install them manually using the "Add Plugin" 
    feature. To do this, click on the + button in the plugins panel. Under the "Serving" menu, you 
    can find and install the API Builder. Similarly, under the "Web Apps" menu, you will find the 
    Port Viewer—install it as well.


Data labeling
-------------

.. _faq_ls_login:

.. dropdown:: What LabelStudio username and password should I use?

    The app uses generic login info:

    * email: user@localhost
    * password: pw


.. _faq_how_many_frames:

.. dropdown:: How many frames do I need to label?

    The typical workflow we recommend is to start with ~100 labeled frames from 2-4 different
    videos
    (this is preferred to labeling 100 frames from a single video, which will not generalize as
    well).
    With this you should be able to train models that provide somewhat reasonable predictions that
    you can then do some preliminary analyses with.
    This is a good regime if you think you might later change the experimental setup, or the
    specific keypoints you're labeling, etc.

    Once you are happy with your experimental setup and plan to acquire a lot of data, then it is
    time to reassess how good the pose tracking is, and how good it actually needs to be for your
    scientific question of interest.
    If all you end up analyzing with the pose estimates is where an animal is located in an open
    field, then maybe super precise tracking of the keypoints isn't necessary.
    But if you care about very subtle changes in pose then precision is much more important.

    If you decide you need better predictions, we recommend labeling another 100-200 frames across
    multiple videos (~20-50 frames per video), training another model, and reassessing the output
    (we recommend looking at snippets of labeled video).
    Repeat this process until you are happy with the results.


Model training
--------------

.. _faq_oom:

.. dropdown:: What if I encounter a CUDA out of memory error?

    We recommend a GPU with at least 8GB of memory.
    Note that both semi-supervised and context models will increase memory usage
    (with semi-supervised context models needing the most memory).
    If you encounter this error, reduce batch sizes during training or inference.
    This feature is currently not supported in the app, so you will need to manually open the config
    file, located at ``Pose-app/data/<proj_name>/model_config_<proj_name>.yaml``, update bactch
    sizes, save the file, then close.
    We also recommend restarting the app after config updates.
    You can find the relevant parameters to adjust
    `here <https://lightning-pose.readthedocs.io/en/latest/source/user_guide/config_file.html>`_
    (this link takes you to another set of docs specifically for Lightning Pose).

.. _faq_epoch:

.. dropdown:: How many epochs should I use for training?

    **What is an epoch?**
    An epoch refers to one complete pass through the entire training dataset. During an epoch, 
    the model is trained on every sample in the dataset exactly once. Find more
    `info here <https://lightning-pose.readthedocs.io/en/latest/source/user_guide/config_file.html#model-training-parameters>`_
    (this link takes you to another set of docs specifically for Lightning Pose).

    **With what value should I start?**
    To train a full model, we recommend starting with the default - 300. To get a baseline
    understanding of how the model performs, we recommend 50 epochs as the minimum number to get
    a valid model to check.

    **What are the trade-offs for increasing or decreasing the number of epochs?**
    Increasing the epochs may enhance convergence and accuracy but raises the risk of overfitting. 
    Conversely, fewer epochs might speed up training but risk underfitting. Balancing epochs is
    crucial to minimize both underfitting and overfitting.


Post-processing
---------------

.. _faq_post_processing:

.. dropdown:: Does the Lightning Pose app perform post-processing of the predictions?

    We offer the `Ensemble Kalman Smoother (EKS) <https://github.com/paninski-lab/eks>`_
    post-processor, which we have found superior to other forms of post-processing.
    To run EKS, see the :ref:`Create an ensemble of models<tab_train_infer__ensemble>` section.
