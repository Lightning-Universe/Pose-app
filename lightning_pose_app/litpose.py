from lightning import LightningWork
from lightning.app.storage.drive import Drive
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities import rank_zero_only
import lightning.pytorch as pl
import os
import yaml


class TrainingProgress(Callback):

    def __init__(self, work):
        self.work = work
        self.progress_delta = 0.5

    @rank_zero_only
    def on_train_epoch_end(self, trainer, *args, **kwargs) -> None:
        progress = 100 * (trainer.current_epoch + 1) / float(trainer.max_epochs)
        if self.work.progress is None:
            if progress > self.progress_delta:
                self.work.progress = round(progress, 4)
        elif round(progress, 4) - self.work.progress >= self.progress_delta:
            if progress > 100:
                self.work.progress = 100.0
            else:
                self.work.progress = round(progress, 4)


class LitPose(LightningWork):

    def __init__(
        self,
        *args,
        drive_name,
        component_name="litpose",
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.pwd = os.getcwd()
        self.progress = 0.0

        self._drive = Drive(drive_name, component_name=component_name)

        self.work_is_done_extract_frames = True
        self.work_is_done_training = True
        self.work_is_done_inference = True
        self.count = 0

    def get_from_drive(self, inputs, component_name=None):
        for i in inputs:
            print(f"drive get {i}")
            try:  # file may not be ready
                self._drive.get(i, overwrite=True, component_name=component_name)
                print(f"drive data saved at {os.path.join(os.getcwd(), i)}")
            except Exception as e:
                print(e)
                print(f"did not load {i} from drive")
                pass

    def put_to_drive(self, outputs, directory=True):
        for o in outputs:
            print(f"drive try put {o}")
            # make sure dir ends with / so that put works correctly
            if directory and o[-1] != "/":
                o = os.path.join(o, "")
            self._drive.put(o)
            print(f"drive success put {o}")

    def _reformat_videos(self, video_files=None, **kwargs):

        from lightning_pose.utils.video_ops import check_codec_format, reencode_video

        # pull videos from Drive; these will come from "root." component
        self.get_from_drive(video_files, component_name="root.")

        video_files_new = []
        for video_file in video_files:

            video_file_abs = os.path.join(os.getcwd(), video_file)

            # check 1: does file exist?
            video_file_exists = os.path.exists(video_file_abs)
            if not video_file_exists:
                continue

            # check 2: is file in the correct format for DALI?
            video_file_correct_codec = check_codec_format(video_file_abs)
            ext = os.path.splitext(os.path.basename(video_file))[1]
            video_file_new = video_file.replace(f"_tmp{ext}", ".mp4")
            video_file_abs_new = os.path.join(os.getcwd(), video_file_new)
            if not video_file_correct_codec:
                print("re-encoding video to be compatable with Lightning Pose video reader")
                reencode_video(video_file_abs, video_file_abs_new)
                # remove local version of old video
                # cannot remove Drive version of old video, created by other Work
                os.remove(video_file_abs)
                # record
                video_files_new.append(video_file_new)
            else:
                # rename
                os.rename(video_file_abs, video_file_abs_new)
                # record
                video_files_new.append(video_file_new)

            # push possibly reformated, renamed videos to Drive
            self.put_to_drive(video_files_new, directory=False)

        return video_files_new

    def _start_extract_frames(self, video_files=None, proj_dir=None, n_frames_per_video=20):

        import numpy as np
        from lightning_pose.utils.video_ops import select_frame_idxs, export_frames
        from lightning_pose.utils.video_ops import check_codec_format

        self.work_is_done_extract_frames = False

        # pull videos from drive; these will come from "litpose"
        self.get_from_drive(video_files, component_name="litpose")

        data_dir_rel = os.path.join(proj_dir, "labeled-data")
        data_dir = os.path.join(os.getcwd(), data_dir_rel)
        n_digits = 8
        extension = "png"
        context_frames = 2

        for video_file in video_files:

            video_file_abs = os.path.join(os.getcwd(), video_file)

            print(f"============== extracting frames from {video_file} ================")

            # check 1: does file exist?
            video_file_exists = os.path.exists(video_file_abs)
            print(f"video file exists? {video_file_exists}")
            if not video_file_exists:
                continue

            # check 2: is file in the correct format for DALI?
            video_file_correct_codec = check_codec_format(video_file_abs)
            print(f"video file codec and pix format correct? {video_file_correct_codec}")

            # create folder to save images
            video_name = os.path.splitext(os.path.basename(video_file))[0]
            save_dir = os.path.join(data_dir, video_name)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            idxs_selected = select_frame_idxs(
                video_file=video_file_abs, resize_dims=64, n_clusters=n_frames_per_video)

            # save csv file inside same output directory
            frames_to_label = np.array([
                "img%s.%s" % (str(idx).zfill(n_digits), extension) for idx in idxs_selected])
            np.savetxt(
                os.path.join(save_dir, "selected_frames.csv"),
                np.sort(frames_to_label),
                delimiter=",",
                fmt="%s"
            )

            export_frames(
                video_file=video_file_abs, save_dir=save_dir, frame_idxs=idxs_selected, 
                format=extension, n_digits=n_digits, context_frames=context_frames)

        # push extracted frames to drive
        self.put_to_drive([data_dir_rel])

        # update flag
        self.work_is_done_extract_frames = True

    def _train(self, inputs, outputs, cfg_overrides, results_dir):
        
        from omegaconf import DictConfig
        from lightning_pose.utils import pretty_print_str, pretty_print_cfg
        from lightning_pose.utils.io import (
            check_video_paths,
            return_absolute_data_paths,
            return_absolute_path,
        )
        from lightning_pose.utils.predictions import predict_dataset
        from lightning_pose.utils.scripts import (
            export_predictions_and_labeled_video,
            get_data_module,
            get_dataset,
            get_imgaug_transform,
            get_loss_factories,
            get_model,
            get_callbacks,
            calculate_train_batches,
            compute_metrics,
        )

        self.work_is_done_training = False

        # ----------------------------------------------------------------------------------
        # Pull data from drive
        # ----------------------------------------------------------------------------------

        # pull config, frames, labels, and videos (relative paths)
        self.get_from_drive(inputs)

        # load config (absolute path)
        for i in inputs:
            if i.endswith(".yaml"):
                config_file = i
        cfg = DictConfig(yaml.safe_load(open(os.path.join(os.getcwd(), config_file), "r")))

        # update config with user-provided overrides
        for key1, val1 in cfg_overrides.items():
            for key2, val2 in val1.items():
                cfg[key1][key2] = val2

        # mimic hydra, change dir into results dir
        os.makedirs(results_dir, exist_ok=True)
        os.chdir(results_dir)

        # ----------------------------------------------------------------------------------
        # Set up data/model objects
        # ----------------------------------------------------------------------------------

        pretty_print_cfg(cfg)

        data_dir, video_dir = return_absolute_data_paths(data_cfg=cfg.data)

        # imgaug transform
        imgaug_transform = get_imgaug_transform(cfg=cfg)

        # dataset
        dataset = get_dataset(cfg=cfg, data_dir=data_dir, imgaug_transform=imgaug_transform)

        # datamodule; breaks up dataset into train/val/test
        data_module = get_data_module(cfg=cfg, dataset=dataset, video_dir=video_dir)

        # build loss factory which orchestrates different losses
        loss_factories = get_loss_factories(cfg=cfg, data_module=data_module)

        # model
        model = get_model(cfg=cfg, data_module=data_module, loss_factories=loss_factories)

        if (
            ("temporal" in cfg.model.losses_to_use)
            and model.do_context
            and not data_module.unlabeled_dataloader.context_sequences_successive
        ):
            raise ValueError(
                f"Temporal loss is not compatible with non-successive context sequences. "
                f"Please change cfg.dali.context.train.consecutive_sequences=True."
            )

        # ----------------------------------------------------------------------------------
        # Set up and run training
        # ----------------------------------------------------------------------------------

        # logger
        logger = pl.loggers.TensorBoardLogger("tb_logs", name=cfg.model.model_name)

        # early stopping, learning rate monitoring, model checkpointing, backbone unfreezing
        callbacks = get_callbacks(cfg)
        # add callback to log progress
        callbacks.append(TrainingProgress(self))

        # calculate number of batches for both labeled and unlabeled data per epoch
        limit_train_batches = calculate_train_batches(cfg, dataset)

        # set up trainer
        trainer = pl.Trainer(
            accelerator="gpu",
            devices=1,
            max_epochs=cfg.training.max_epochs,
            min_epochs=cfg.training.min_epochs,
            check_val_every_n_epoch=cfg.training.check_val_every_n_epoch,
            log_every_n_steps=cfg.training.log_every_n_steps,
            callbacks=callbacks,
            logger=logger,
            limit_train_batches=limit_train_batches,
            accumulate_grad_batches=cfg.training.accumulate_grad_batches,
            profiler=cfg.training.profiler,
        )

        # train model!
        trainer.fit(model=model, datamodule=data_module)

        # ----------------------------------------------------------------------------------
        # Post-training analysis
        # ----------------------------------------------------------------------------------
        hydra_output_directory = os.getcwd()
        print("Hydra output directory: {}".format(hydra_output_directory))
        # get best ckpt
        best_ckpt = os.path.abspath(trainer.checkpoint_callback.best_model_path)
        # check if best_ckpt is a file
        if not os.path.isfile(best_ckpt):
            raise FileNotFoundError("Cannot find checkpoint. Have you trained for too few epochs?")

        # make unaugmented data_loader if necessary
        if cfg.training.imgaug != "default":
            cfg_pred = cfg.copy()
            cfg_pred.training.imgaug = "default"
            imgaug_transform_pred = get_imgaug_transform(cfg=cfg_pred)
            dataset_pred = get_dataset(
                cfg=cfg_pred, data_dir=data_dir, imgaug_transform=imgaug_transform_pred
            )
            data_module_pred = get_data_module(cfg=cfg_pred, dataset=dataset_pred, video_dir=video_dir)
            data_module_pred.setup()
        else:
            data_module_pred = data_module

        # predict on all labeled frames (train/val/test)
        pretty_print_str("Predicting train/val/test images...")
        # compute and save frame-wise predictions
        preds_file = os.path.join(hydra_output_directory, "predictions.csv")
        predict_dataset(
            cfg=cfg,
            trainer=trainer,
            model=model,
            data_module=data_module_pred,
            ckpt_file=best_ckpt,
            preds_file=preds_file,
        )
        # compute and save various metrics
        try:
            compute_metrics(cfg=cfg, preds_file=preds_file, data_module=data_module_pred)
        except Exception as e:
            print(f"Error computing metrics\n{e}")

        # predict folder of videos
        if cfg.eval.predict_vids_after_training:
            pretty_print_str("Predicting videos...")
            if cfg.eval.test_videos_directory is None:
                filenames = []
            else:
                filenames = check_video_paths(return_absolute_path(cfg.eval.test_videos_directory))
                pretty_print_str(
                    f"Found {len(filenames)} videos to predict on (in cfg.eval.test_videos_directory)")
            for video_file in filenames:
                assert os.path.isfile(video_file)
                pretty_print_str(f"Predicting video: {video_file}...")
                # get save name for prediction csv file
                video_pred_dir = os.path.join(hydra_output_directory, "video_preds")
                video_pred_name = os.path.splitext(os.path.basename(video_file))[0]
                prediction_csv_file = os.path.join(video_pred_dir, video_pred_name + ".csv")
                # get save name labeled video csv
                if cfg.eval.save_vids_after_training:
                    labeled_vid_dir = os.path.join(video_pred_dir, "labeled_videos")
                    labeled_mp4_file = os.path.join(labeled_vid_dir, video_pred_name + "_labeled.mp4")
                else:
                    labeled_mp4_file = None
                # predict on video
                export_predictions_and_labeled_video(
                    video_file=video_file,
                    cfg=cfg,
                    ckpt_file=best_ckpt,
                    prediction_csv_file=prediction_csv_file,
                    labeled_mp4_file=labeled_mp4_file,
                    trainer=trainer,
                    model=model,
                    gpu_id=cfg.training.gpu_id,
                    data_module=data_module_pred,
                    save_heatmaps=cfg.eval.get("predict_vids_after_training_save_heatmaps", False),
                )
                # compute and save various metrics
                try:
                    compute_metrics(
                        cfg=cfg, preds_file=prediction_csv_file, data_module=data_module_pred
                    )
                except Exception as e:
                    print(f"Error predicting on video {video_file}:\n{e}")
                    continue

        # ----------------------------------------------------------------------------------
        # Push results to drive, clean up
        # ----------------------------------------------------------------------------------
        os.chdir(self.pwd)
        self.put_to_drive(outputs)  # IMPORTANT! must come after changing directories
        self.work_is_done_training = True

    def _run_inference(self, model, video):
        import time
        self.work_is_done_inference = False
        print(f"launching inference for video {video} using model {model}")
        time.sleep(5)
        self.work_is_done_inference = True

    def run(self, action=None, **kwargs):

        if action == "start_extract_frames":
            video_files_new = self._reformat_videos(**kwargs)
            kwargs["video_files"] = video_files_new
            self._start_extract_frames(**kwargs)
        elif action == "train":
            self._train(**kwargs)
        elif action == "run_inference":
            self._run_inference(**kwargs)
