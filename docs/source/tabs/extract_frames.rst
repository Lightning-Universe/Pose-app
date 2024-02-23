.. _tab_extract_frames:

##############
Extract Frames
##############

The Extract Frames tab allows you to select frames for labeling using various methods.

.. note::

    This tab does not appear in the demo app.

* :ref:`Upload videos and automatically extract random frames <upload_video_random>`
* :ref:`Upload zipped files of frames <upload_zipped_frames>`

.. _upload_video_random:

Upload videos and automatically extract random frames
=====================================================

Select the appropriate option from the list.

.. image:: https://imgur.com/szEHHtw.png

Drag and drop video file(s) using the provided interface. You will see an upload progress bar.

.. image:: https://imgur.com/GjR2Jb4.png

Choose number of frames to label per video - these frames will be automatically selected to
maximize the diversity of poses from each video.

You can also select the portion of the video to extract frames from.
If the beginning and/or end of your videos do not contain the animals or contain extra objects
(e.g. experimenter hands) we recommend excluding these portions.

Click "Extract frames" once the video upload is complete, and another progress bar will appear.

.. image:: https://imgur.com/U258Vah.png

Once all frames have been extracted you will see "Proceed to the next tab to label frames" in green.

.. image:: https://imgur.com/F9y1aPv.png


.. _upload_zipped_frames:

Upload zipped files of frames
=============================

Select the appropiate option from the list.

.. image:: https://imgur.com/64ZjGu3.png

Drag and drop zipped files(s) of frames using the provided interface.
You will see an upload progress bar.

.. warning::

    At the moment this feature of the app requires a strict file structure!

As an example, let's say you have a video named ``subject023_session0.mp4`` and you have extracted
frames 143, 1156, and 34567, which you want to label.

You will need to create a single zip file named ``subject023_session0.zip``.
The zip file must contain png files, and they must follow the naming convention ``img%08.png``,
for example ``img00000143.png``
(such that there are 8 digits for the frame number, with leading zeros).

If you would like to fit context models, you must also include context frames for each labeled
frame. Again using frame 143 as an example, you must include five files:

* img00000141.png
* img00000142.png
* img00000143.png
* img00000144.png
* img00000145.png

Including context frames is recommended, though not required.

Finally, you must include a csv file named ``selected_frames.csv`` that is simply a list of the
file names of the frames you wish to *label* (not the context frames),
so that LabelStudio knows which frames to upload into its database.
For the example above, the csv file should look like:

.. code-block::

    img00000143.png
    img00001156.png
    img00034567.png

Therefore, the final set of files that must be zipped into ``subject023_session0.zip`` for this
example is:

* img00000141.png
* img00000142.png
* img00000143.png
* img00000144.png
* img00000145.png
* img00001154.png
* img00001155.png
* img00001156.png
* img00001157.png
* img00001158.png
* img00034565.png
* img00034566.png
* img00034567.png
* img00034568.png
* img00034569.png
* selected_frames.csv

If you would like to upload frames for multiple videos, make one zip file per video.

Click "Extract frames" once the zip file upload is complete.

Once all frames have been extracted you will see "Proceed to the next tab to label frames" in green.

.. image:: https://imgur.com/F9y1aPv.png
