from lightning.app import CloudCompute
import numpy as np
import os


def test_extract_frames_work(video_file, tmpdir):
    """Test private methods here; test run method externally from the UI object."""

    from lightning_pose_app import LABELED_DATA_DIR, SELECTED_FRAMES_FILENAME
    from lightning_pose_app.ui.extract_frames import ExtractFramesWork

    work = ExtractFramesWork(
        cloud_compute=CloudCompute("default"),
    )

    # read frame function
    resize_dims = 8
    frames = work._read_nth_frames(video_file, n=10, resize_dims=resize_dims)
    assert frames.shape == (100, resize_dims, resize_dims)

    # select indices
    n_clusters = 5
    idxs = work._select_frame_idxs(
        video_file, resize_dims=resize_dims, n_clusters=n_clusters, frame_skip=1,
    )
    assert len(idxs) == n_clusters

    # export frames
    save_dir_0 = os.path.join(str(tmpdir), 'labeled-frames-0')
    work._export_frames(
        video_file=video_file,
        save_dir=save_dir_0,
        frame_idxs=idxs,
        context_frames=0,  # no context
    )
    assert len(os.listdir(save_dir_0)) == len(idxs)

    save_dir_1 = os.path.join(str(tmpdir), 'labeled-frames-1')
    idxs = np.array([5, 10, 15, 20])
    work._export_frames(
        video_file=video_file,
        save_dir=save_dir_1,
        frame_idxs=idxs,
        context_frames=2,  # 2-frame context
    )
    assert len(os.listdir(save_dir_1)) == 5 * len(idxs)

    # extract frames
    proj_dir = os.path.join(str(tmpdir), 'proj-dir-0')
    video_name = os.path.splitext(os.path.basename(str(video_file)))[0]
    video_dir = os.path.join(proj_dir, LABELED_DATA_DIR, video_name)
    os.makedirs(video_dir, exist_ok=True)  # need to create this here for path purposes
    n_frames_per_video = 10
    work._extract_frames(
        video_file=video_file,
        proj_dir=proj_dir,
        n_frames_per_video=n_frames_per_video,
        frame_range=[0, 1],
    )
    assert os.path.exists(video_dir)
    assert len(os.listdir(video_dir)) > n_frames_per_video
    assert os.path.exists(os.path.join(video_dir, SELECTED_FRAMES_FILENAME))
    assert work.work_is_done_extract_frames

    # TODO: test work._unzip_frames
