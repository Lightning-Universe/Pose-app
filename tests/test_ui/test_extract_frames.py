from lightning.app import CloudCompute
import numpy as np
import os
import pandas as pd
import shutil


from lightning_pose_app import LABELED_DATA_DIR, SELECTED_FRAMES_FILENAME, MODEL_VIDEO_PREDS_INFER_DIR
from lightning_pose_app import MODELS_DIR, VIDEOS_TMP_DIR, VIDEOS_DIR


def test_extract_frames_work(video_file, video_file_pred_df, video_file_pca_singleview_df ,tmpdir):
    """Test private methods here; test run method externally from the UI object."""

    from lightning_pose_app.ui.extract_frames import ExtractFramesWork

    work = ExtractFramesWork(
        cloud_compute=CloudCompute("default"),
    )

    # -----------------
    # read frame function
    # -----------------
    resize_dims = 8
    frames = work._read_nth_frames(video_file, n=10, resize_dims=resize_dims)
    assert frames.shape == (100, resize_dims, resize_dims)

    # -----------------
    # select indices
    # -----------------
    n_clusters = 5
    idxs = work._select_frame_idxs(
        video_file, resize_dims=resize_dims, n_clusters=n_clusters, frame_skip=1,
    )
    assert len(idxs) == n_clusters

    # -----------------
    # select indices w/ model
    # -----------------
    # TODO: make sure to update test by making dummy prediction/metric files
    # prediction: use video_file_pred_df
    proj_dir = os.path.join(str(tmpdir), 'proj-dir-0')
    model_dir = os.path.join(proj_dir, MODELS_DIR, 'dd-mm-yy/hh-mm-ss')

    video_name = os.path.splitext(os.path.basename(str(video_file)))[0]
    path = os.path.join(model_dir, MODEL_VIDEO_PREDS_INFER_DIR, video_name +"_pca_singleview_error.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    video_file_pca_singleview_df.to_csv(path)
    # save predictions
    # should be saved in os.path.join(model_dir, <video_name>.csv)
    # save metrics
    n_frames_per_video = 7
    idxs = work._select_frame_idxs_using_model(
        video_file=video_file,
        proj_dir=proj_dir,
        model_dir=os.path.join(model_dir, MODEL_VIDEO_PREDS_INFER_DIR), ##TO FIX
        n_frames_per_video=n_frames_per_video,
        frame_range=[0, 1],
    )
    assert len(idxs) == n_frames_per_video

    # -----------------
    # export frames
    # -----------------
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

    save_dir_2 = os.path.join(str(tmpdir), 'labeled-frames-2')
    idxs = np.array([10])  # try with single frame
    work._export_frames(
        video_file=video_file,
        save_dir=save_dir_2,
        frame_idxs=idxs,
        context_frames=2,  # 2-frame context
    )
    assert len(os.listdir(save_dir_2)) == 5 * len(idxs)

    # -----------------
    # extract frames
    # -----------------
    # use "random" method
    proj_dir = os.path.join(str(tmpdir), 'proj-dir-1')
    video_name = os.path.splitext(os.path.basename(str(video_file)))[0]
    video_dir = os.path.join(proj_dir, LABELED_DATA_DIR, video_name)
    os.makedirs(os.path.dirname(video_dir), exist_ok=True)  # need to create for path purposes
    n_frames_per_video = 10
    work._extract_frames(
        method="random",
        video_file=video_file,
        proj_dir=proj_dir,
        n_frames_per_video=n_frames_per_video,
        frame_range=[0, 1],
    )
    assert os.path.exists(video_dir)
    assert len(os.listdir(video_dir)) > n_frames_per_video
    assert os.path.exists(os.path.join(video_dir, SELECTED_FRAMES_FILENAME))
    assert work.work_is_done_extract_frames

    # use "active" method
    # TODO: make sure to update test by making dummy prediction/metric files in model dir
    proj_dir = os.path.join(str(tmpdir), 'proj-dir-2')
    model_dir = os.path.join(proj_dir, MODELS_DIR, 'dd-mm-yy/hh-mm-ss')

    video_name = os.path.splitext(os.path.basename(str(video_file)))[0]
    path = os.path.join(model_dir, MODEL_VIDEO_PREDS_INFER_DIR, video_name +"_pca_singleview_error.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    video_file_pca_singleview_df.to_csv(path)

    video_dir = os.path.join(proj_dir, LABELED_DATA_DIR, video_name)
    os.makedirs(os.path.dirname(video_dir), exist_ok=True)  # need to create for path purposes
    n_frames_per_video = 10
    work._extract_frames(
        method="active",
        video_file=video_file,
        proj_dir=proj_dir,
        n_frames_per_video=n_frames_per_video,
        frame_range=[0, 1],
        model_dir=os.path.join(model_dir,MODEL_VIDEO_PREDS_INFER_DIR), ##TO FIX - remove all part and leave "model_dir"
    )
    assert os.path.exists(video_dir)
    assert len(os.listdir(video_dir)) > n_frames_per_video
    assert os.path.exists(os.path.join(video_dir, SELECTED_FRAMES_FILENAME))
    assert work.work_is_done_extract_frames

    # -----------------
    # unzip frames
    # -----------------
    # zip up a subset of the frames extracted from the previous test
    n_frames_to_zip = 5
    frame_files = os.listdir(save_dir_1)
    new_vid_name = "TEST_VID_ZIPPED_FRAMES"
    dst = os.path.join(tmpdir, new_vid_name)
    os.makedirs(dst, exist_ok=True)
    files = []
    for f in range(n_frames_to_zip):
        src = os.path.join(save_dir_1, frame_files[f])
        shutil.copyfile(src, os.path.join(dst, frame_files[f]))
        files.append(frame_files[f])
    # make a csv file to accompany frames
    np.savetxt(
        os.path.join(dst, SELECTED_FRAMES_FILENAME),
        np.sort(files),
        delimiter=",",
        fmt="%s",
    )
    # zip it all up
    new_video_name = new_vid_name + "_NEW"
    new_video_path = os.path.join(tmpdir, new_video_name)
    zipped_file = new_video_path + ".zip"
    shutil.make_archive(new_video_path, "zip", dst)

    # test unzip frames
    proj_dir = os.path.join(str(tmpdir), 'proj-dir-1')
    video_dir = os.path.join(proj_dir, LABELED_DATA_DIR, new_video_name)
    os.makedirs(os.path.dirname(video_dir), exist_ok=True)  # need to create for path purposes
    work.work_is_done_extract_frames = False
    work._unzip_frames(
        video_file=zipped_file,
        proj_dir=proj_dir,
    )
    assert os.path.exists(video_dir)
    assert len(os.listdir(video_dir)) == (n_frames_to_zip + 1)
    idx_file_abs = os.path.join(video_dir, SELECTED_FRAMES_FILENAME)
    assert os.path.exists(idx_file_abs)
    df = pd.read_csv(idx_file_abs, header=None)
    assert df.shape[0] == n_frames_to_zip
    assert work.work_is_done_extract_frames

    # -----------------
    # cleanup
    # -----------------
    del work


def test_extract_frames_ui(root_dir, tmp_proj_dir):

    from lightning_pose_app.ui.extract_frames import ExtractFramesUI

    video_name = "test_vid_copy"
    video_file_ = video_name + ".mp4"
    video_file = os.path.join(tmp_proj_dir, VIDEOS_TMP_DIR, video_file_)

    flow = ExtractFramesUI()

    # set attributes
    flow.proj_dir = tmp_proj_dir
    flow.st_extract_status[video_file] = "initialized"

    # -------------------
    # extract frames
    # -------------------
    n_frames_per_video = 10
    flow.run(
        action="extract_frames",
        video_files=[video_file],
        n_frames_per_video=n_frames_per_video,
        testing=True,
    )

    # make sure flow attributes are properly cleaned up
    assert flow.st_extract_status[video_file] == "complete"
    assert len(flow.works_dict) == 0
    assert flow.work_is_done_extract_frames

    # make sure frames were extracted
    proj_dir_abs = os.path.join(root_dir, tmp_proj_dir)
    frame_dir_abs = os.path.join(proj_dir_abs, LABELED_DATA_DIR, video_name)
    idx_file_abs = os.path.join(frame_dir_abs, SELECTED_FRAMES_FILENAME)
    assert os.path.isfile(os.path.join(proj_dir_abs, VIDEOS_DIR, video_file_))
    assert os.path.isdir(frame_dir_abs)
    assert os.path.isfile(idx_file_abs)

    df = pd.read_csv(idx_file_abs, header=None)
    assert df.shape[0] == n_frames_per_video

    # -------------------
    # unzip frames
    # -------------------
    # TODO

    # -----------------
    # cleanup
    # -----------------
    del flow


def test_run_kmeans():
    from lightning_pose_app.ui.extract_frames import run_kmeans

    n_samples = int(50)
    n_features = int(5)
    n_clusters = 10

    data_to_cluster = np.random.rand(n_samples,n_features)
    cluster = run_kmeans(data_to_cluster,n_clusters)
  
    assert len(cluster) == n_samples
    assert len(np.unique(cluster)) == n_clusters


def test_select_max_frame_per_cluster():
    from lightning_pose_app.ui.extract_frames import select_max_frame_per_cluster

    df = pd.DataFrame({"frames index":[1,2,3,4,5,6],
                        "error score":[10,10,15,15,10,10],
                        "cluster_labels":[1,1,2,1,2,2]
    })

    list_of_frames = select_max_frame_per_cluster(df)
    assert list_of_frames[0] == 3
    assert list_of_frames[1] == 4










def compute_motion_energy_from_predection_df(df, likelihood_thresh):
    kps_and_conf = df.to_numpy().reshape(df.shape[0], -1, 3)
    kps = kps_and_conf[:, :, :2]
    conf = kps_and_conf[:, :, -1]
    conf2 = np.concatenate([conf[:, :, None], conf[:, :, None]], axis=2)
    kps[conf2 < likelihood_thresh] = np.nan
    me = np.nanmean(np.linalg.norm(kps[1:] - kps[:1], axis=2), axis=-1)
    me = np.concatenate([[0], me])
    return me












def compute_motion_energy(preds, likelihood_thresh, metrics):

    # Convert predictions to numpy array and reshape
    kps_and_conf = preds.to_numpy().reshape(preds.shape[0], -1, 3)  # n_frames x n_keypoints x 3 (x, y, conf)
    kps = kps_and_conf[:, :, :2]
    conf = kps_and_conf[:, :, -1]

    # Store likelihood scores in metrics dictionary
    metrics["likelihood"] = pd.DataFrame(conf)

    # Duplicate likelihood scores for x and y coordinates
    conf2 = np.concatenate([conf[:, :, None], conf[:, :, None]], axis=2)

    # Apply likelihood threshold
    kps[conf2 < likelihood_thresh] = np.nan  # Filter out low-likelihood keypoints

    # Compute motion energy
    me = np.nanmean(np.linalg.norm(kps[1:] - kps[:-1], axis=2), axis=-1)
    me = np.concatenate([[0], me])  # Prepend 0 to maintain frame consistency

    me_prctile = 50 if preds.shape[0] < 1e5 else 75  # take fewer frames if there are many
    # Select index of high ME frames
    idxs_high_me = np.where(me > np.percentile(me, me_prctile))[0]

    #Select high ME frames from metrics
    for metric, val in metrics.items():
        metrics[metric] = val.loc[idxs_high_me]

    return me, idxs_high_me, metrics


