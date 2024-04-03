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
    Finally, when you run the labeling app or full app select "Create new project from source" and
    upload your zip file. As of 07/2023 context datasets are not automatically created upon import;
    if this is a feature you would like to see,
    please `open an issue <https://github.com/Lightning-Universe/Pose-app/issues>`_.


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
    file, located at `Pose-app/data/<proj_name>/model_config_<proj_name>.yaml`, update bactch
    sizes, save the file, then close.
    We also recommend restarting the app after config updates.
    You can find the relevant parameters to adjust
    `here <https://lightning-pose.readthedocs.io/en/latest/source/user_guide/config_file.html>`_.

.. _faq_ls_login:

.. dropdown:: What LabelStudio username and password should I use?

    The app uses generic login info:

    * email: user@localhost
    * password: pw

.. _faq_change_machine:

.. dropdown:: How can I switch the cloud machine type (CPU to/from GPU)?

    In the upper right corner of the Lightning Studio, click on the compute icon
    (which will read ``1 T4`` if you are connected to the default T4 GPU, or ``4 CPU`` if you are
    connected to a CPU-only machine).
    Select the GPU or CPU box to see available options.
    We recommend a default CPU machine (not data prep) for labeling tasks.

    .. image:: https://imgur.com/HGtYm0g.png
        :width: 400
