from lightning import LightningWork
from lightning.app.storage.drive import Drive
from lightning.pytorch.callbacks import Callback
import lightning.pytorch as pl
import os
import yaml


class TrainingProgress(Callback):

    def __init__(self, state, key="progress"):
        super().__init__()
        self.state = state
        self.key = key
    
    def on_train_epoch_end(self, *args, **kwargs):
        old_val = self.state.__getattr__(self.key)
        self.state.__setattr__(self.key, old_val + 1)


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
        self.progress = 0

        self._drive = Drive(drive_name, component_name=component_name)

        self.work_is_done_extract_frames = True
        self.work_is_done_training = True
        self.work_is_done_inference = True
        self.count = 0

    def get_from_drive(self, inputs):
        for i in inputs:
            print(f"drive get {i}")
            try:  # file may not be ready
                self._drive.get(i, overwrite=True)
                print(f"drive data saved at {os.path.join(os.getcwd(), i)}")
            except Exception as e:
                print(e)
                print(f"did not load {i} from drive")
                pass

    def put_to_drive(self, outputs):
        for o in outputs:
            print(f"drive try put {o}")
            # make sure dir end with / so that put works correctly
            if os.path.isdir(o):
                o = os.path.join(o, "")
            # check to make sure file exists locally
            if not os.path.exists(o):
                continue
            self._drive.put(o)
            print(f"drive success put {o}")

    # def start_extract_frames(self, video_files=None, proj_dir=None, n_frames_per_video=20):

    #     print(f"launching extraction for video {video_files[0]}")
    #     self.work_is_done_extract_frames = False

    #     print(f"launching extraction for video {video_files[0]}")
    #     self.work_is_done_extract_frames = False

    #     # set videos to select frames from
    #     vid_file_args = ""
    #     for vid_file in video_files:
    #         vid_file_ = os.path.join(os.getcwd(), vid_file)
    #         vid_file_args += f" --video_files={vid_file_}"

    #     data_dir = os.path.join(os.getcwd(), proj_dir, "labeled-data")

    #     cmd = "python" \
    #           + " scripts/extract_frames.py" \
    #           + vid_file_args \
    #           + f" --data_dir={data_dir}" \
    #           + f" --n_frames_per_video={n_frames_per_video}" \
    #           + f" --context_frames=2" \
    #           + f" --export_idxs_as_csv"
    #     self.work.run(
    #         cmd,
    #         wait_for_exit=True,
    #         cwd=lightning_pose_dir,
    #         inputs=video_files,
    #         outputs=[os.path.join(proj_dir, "labeled-data")],
    #     )
    #     self.work_is_done_extract_frames = True

    def train(self, inputs, outputs, cfg_overrides, results_dir):
        
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
        callbacks.append(TrainingProgress(self, "progress"))

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
        # Push results to drive
        # ----------------------------------------------------------------------------------
        self.put_to_drive(outputs)

        os.chdir(self.pwd)

    # def run_inference(self, model, video):
    #     import time
    #     self.work_is_done_inference = False
    #     print(f"launching inference for video {video} using model {model}")
    #     time.sleep(5)
    #     self.work_is_done_inference = True

    def run(self, action=None, **kwargs):

        if action == "start_extract_frames":
            self.start_extract_frames(**kwargs)
        elif action == "train":
            self.train(**kwargs)
        elif action == "run_inference":
            self.run_inference(**kwargs)
