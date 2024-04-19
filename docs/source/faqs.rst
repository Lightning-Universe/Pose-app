####
FAQs
####

.. _faq_can_i_import:

.. dropdown:: Can I import a pose estimation project in another format into the app?

    We currently support conversion from DLC projects into Lightning Pose projects
    (if you would like support for another format,
    please `open an issue <https://github.com/Lightning-Universe/Pose-app/issues>`_).
    First, make sure that all videos in your DLC project are actual files, not symbolic links.
    Symlinked videos in the DLC project will not be uploaded.
    Next, compress your DLC project into a zip file - you will upload this into the app later.
    Finally, when you run the app select "Create new project from source" and upload your zip file.
    As of 07/2023 context datasets are not automatically created upon import; if this is a feature
    you would like to see, please
    `open an issue <https://github.com/Lightning-Universe/Pose-app/issues>`_.


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


.. _faq_change_machine:

.. dropdown:: How can I switch the cloud machine type (CPU to/from GPU)?

    In the upper right corner of the Lightning Studio, click on the compute icon
    (which will read ``1 T4`` if you are connected to the default T4 GPU, or ``4 CPU`` if you are
    connected to a CPU-only machine).
    Select the GPU or CPU box to see available options.
    We recommend a default CPU machine (not data prep) for labeling tasks.

    .. image:: https://imgur.com/HGtYm0g.png
        :width: 400


.. _faq_upload_limit:

.. dropdown:: How do I increase the file upload size limit?

    Set the following environment variable (in units of MB);
    we recommend using videos less than 500MB for best performance.

    .. code-block:: console

        lightning run app <app_name.py> --env STREAMLIT_SERVER_MAX_UPLOAD_SIZE=500


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
