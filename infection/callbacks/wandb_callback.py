import os
import os.path as osp
from copy import deepcopy
from datetime import datetime
from typing import Dict
import torch
from deepdiff import DeepDiff
import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback
from omegaconf import DictConfig, OmegaConf
try:
    import wandb as wandb_logger
except ModuleNotFoundError:
    pass


def pretty_print_diff(diff):
    texts = []
    for type_key in diff.keys():
        for config_key in diff[type_key].keys():
            if type_key == "values_changed":
                texts.append(
                    config_key
                    + ": "
                    + str(diff[type_key][config_key]["old_value"])
                    + "-->"
                    + str(diff[type_key][config_key]["new_value"])
                )
            elif "item_removed" in type_key:
                texts.append(config_key + ": " + str(diff[type_key][config_key]))
            elif "item_added" in type_key:
                texts.append(config_key + ": " + str(diff[type_key][config_key]))

    return "\n".join(texts)

class Wandb_Logger:
    """
    Logger for wandb intergration
    :param log_dir: Path to save checkpoint
    """

    def __init__(
        self,
        unique_id: str,
        username: str,
        project_name: str,
        run_name: str,
        group_name: str = None,
        save_dir: str = None,
        config_dict: Dict = None,
        **kwargs,
    ):
        self.project_name = project_name
        self.username = username
        self.run_name = run_name
        self.config_dict = config_dict
        self.id = unique_id
        self.save_dir = save_dir
        self.group_name = group_name

        wandb_logger.init(
            id=self.id,
            dir=self.save_dir,
            config=config_dict,
            entity=username,
            group=self.group_name,
            project=project_name,
            name=run_name,
            resume="allow",
            reinit=kwargs.get("reinit", False),
            tags=kwargs.get("tags", None),
        )

        wandb_logger.watch_called = False

    def load_state_dict(self, path):
        if wandb_logger.run.resumed:
            state_dict = torch.load(wandb_logger.restore(path))
            return state_dict
        else:
            return None

    def log_file(self, tag, value, base_folder=None, **kwargs):
        """
        Write a file to wandb
        :param tag: (str) tag
        :param value: (str) path to file

        :param base_folder: (str) folder to save file to
        """
        wandb_logger.save(value, base_path=base_folder)

    def log_scalar(self, tag, value, step, **kwargs):
        """
        Write a log to specified directory
        :param tags: (str) tag for log
        :param values: (number) value for corresponding tag
        :param step: (int) logging step
        """

        # define our custom x axis metric
        wandb_logger.define_metric("iterations")
        # define which metrics will be plotted against it
        wandb_logger.define_metric(tag, step_metric="iterations")

        wandb_logger.log({tag: value, "iterations": step})

    def log_figure(self, tag, value, step=0, **kwargs):
        """
        Write a matplotlib fig to wandb
        :param tags: (str) tag for log
        :param value: (image) image to log. torch.Tensor or plt.fire.Figure
        :param step: (int) logging step
        """

        try:
            if isinstance(value, torch.Tensor):
                image = wandb_logger.Image(value)
                wandb_logger.log({tag: image, "iterations": step})
            else:
                wandb_logger.log({tag: value, "iterations": step})
        except Exception as e:
            pass

    def log_torch_module(self, tag, value, log_freq, **kwargs):
        """
        Write a model graph to wandb
        :param value: (nn.Module) torch model
        :param inputs: sample tensor
        """
        wandb_logger.watch(value, log="gradients", log_freq=log_freq)

    def log_spec_text(self, tag, value, step, **kwargs):
        """
        Write a text to wandb
        :param value: (str) captions
        """
        texts = wandb_logger.Html(value)
        wandb_logger.log({tag: texts, "iterations": step})

    def log_table(self, tag, value, columns, step, **kwargs):
        """
        Write a table to wandb
        :param value: list of column values
        :param columns: list of column names

        Examples:
        value = [
            [0, fig1, 0],
            [1, fig2, 8],
            [2, fig3, 7],
            [3, fig4, 1]
        ]
        columns=[
            "id",
            "image",
            "prediction"
        ]
        """

        # Workaround for tensor image, have not figured out how to use plt.Figure :<
        new_value = []
        for record in value:
            new_record = []
            for val in record:
                if isinstance(val, torch.Tensor):
                    val = wandb_logger.Image(val)
                new_record.append(val)
            new_value.append(new_record)

        table = wandb_logger.Table(data=new_value, columns=columns)
        wandb_logger.log({tag: table, "iterations": step})

    def log_video(self, tag, value, step, fps, **kwargs):
        """
        Write a video to wandb
        :param value: numpy array (time, channel, height, width)
        :param fps: int
        """
        # axes are
        wandb_logger.log({tag: wandb_logger.Video(value, fps=fps), "iterations": step})

    def log_html(self, tag, value, step=0, **kwargs):
        """
        Display a html
        :param value: path to html file
        """
        table = wandb_logger.Table(columns=[tag])
        table.add_data(wandb_logger.Html(value))
        wandb_logger.log({tag: table, "iterations": step})

    def log_embedding(
        self,
        tag,
        value,
        label_img=None,
        step=0,
        metadata=None,
        metadata_header=None,
        **kwargs,
    ):
        """
        Write a embedding projection to tensorboard
        :param value: embeddings array (N, D)
        :param label_img: (torch.Tensor) normalized image tensors (N, 3, H, W)
        :param metadata: (List) zipped list of metadata
        :param metadata_header: (List) list of metadata names according to the metadata provided
        """

        import pandas as pd

        df_dict = {"embeddings": [e for e in value.tolist()]}
        if metadata is not None and metadata_header is not None:
            for meta in metadata:
                for idx, item in enumerate(meta):
                    if metadata_header[idx] not in df_dict.keys():
                        df_dict[metadata_header[idx]] = []
                    df_dict[metadata_header[idx]].append(item)
        if label_img is not None:
            df_dict["images"] = [wandb_logger.Image(i.values) for i in label_img]

        df = pd.DataFrame(df_dict)

        table = wandb_logger.Table(columns=df.columns.to_list(), data=df.values)
        wandb_logger.log({tag: table, "iterations": step})

    def __del__(self):
        wandb_logger.finish()


def find_run_id(dirname):
    """
    Read a .txt file which contains wandb run id
    """

    wandb_id_file = osp.join(dirname, "wandb_id.txt")

    if not osp.isfile(wandb_id_file):
        raise ValueError(f"Wandb ID file not found in {wandb_id_file}")
    else:
        with open(wandb_id_file, "r") as f:
            wandb_id = f.read().rstrip()
        return wandb_id

class WandbCallback(Callback):
    """
    Callbacks for logging running loss/metric/time while training to wandb server
    Features:
        - Only do logging

    username: `str`
        username of Wandb
    project_name: `str`
        project name of Wandb
    resume: `bool`
        whether to resume project
    """

    def __init__(
        self,
        username: str,
        project_name: str,
        group_name: str = None,
        save_dir: str = None,
        resume: str = None,
        config_dict: DictConfig = None,
        **kwargs,
    ) -> None:
        super().__init__()

        self.username = username
        self.project_name = project_name
        self.resume = resume
        self.save_dir = save_dir
        self.config_dict = config_dict

        # A hack, not good
        if self.save_dir is None:
            self.run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.run_name = osp.basename(save_dir)

        if self.resume is None:
            self.id = wandb_logger.util.generate_id()
        else:
            try:
                # Get run id
                run_id = find_run_id(os.path.dirname(os.path.dirname(self.resume)))

                # Load the config from that run
                try:
                    old_config_path = wandb_logger.restore(
                        "pipeline.yaml",
                        run_path=f"{self.username}/{self.project_name}/{run_id}",
                        root=f".cache/{run_id}/",
                        replace=True,
                    ).name
                except Exception:
                    raise ValueError(
                        f"Falid to load run id={run_id}, due to pipeline.yaml is missing or run is not existed"
                    )

                # Check if the config remains the same, if not, create new run id
                old_config_dict = OmegaConf.load(old_config_path)
                tmp_config_dict = deepcopy(self.config_dict)
                ## strip off global key because `resume` will always different
                old_config_dict.pop("global", None)
                OmegaConf.set_struct(tmp_config_dict, False)
                tmp_config_dict.pop("global", None)
                if old_config_dict == tmp_config_dict:
                    self.id = run_id
                    print(
                        "Run configuration remains unchanged. Resuming wandb run...",
                    )
                else:
                    diff = DeepDiff(old_config_dict, tmp_config_dict)
                    diff_text = pretty_print_diff(diff)

                    print(
                        f"Config values mismatched: {diff_text}",
                    )
                    print(
                        """Run configuration changes since the last run. Decide:
                        (1) Terminate run
                        (2) Create new run
                        (3) Override run (not recommended)
                        """,
                    )

                    answer = int(input())
                    assert answer in [1, 2], "Wrong input"
                    if answer == 2:
                        print(
                            "Creating new wandb run...",
                        )
                        self.id = wandb_logger.util.generate_id()
                    elif answer == 1:
                        print("Terminating run...")
                        raise InterruptedError()
                    else:
                        print(
                            "Overriding run...",
                        )
                        self.id = run_id

            except ValueError as e:
                print(
                    f"Can not resume wandb due to '{e}'. Creating new wandb run...",
                )
                self.id = wandb_logger.util.generate_id()

        # All the logging stuffs have been done in LoggerCallbacks.
        # Here we just register the wandb logger to the main logger

        self.wandb_logger = Wandb_Logger(
            unique_id=self.id,
            save_dir=self.save_dir,
            username=self.username,
            project_name=self.project_name,
            run_name=self.run_name,
            config_dict=self.config_dict,
            group_name=group_name,
            **kwargs,
        )

    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str) -> None:
        """
        Before going to the main loop. Save run id
        """
        wandb_id_file = osp.join(self.save_dir, "wandb_id.txt")
        with open(wandb_id_file, "w") as f:
            f.write(self.id)

        # Save all config files
        self.wandb_logger.log_file(
            tag="configs",
            base_folder=self.save_dir,
            value=osp.join(self.save_dir, "*.yaml"),
        )

    def teardown(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str):
        """
        After finish training
        """
        base_folder = osp.join(self.save_dir, "checkpoints")
        self.wandb_logger.log_file(
            tag="checkpoint",
            base_folder=self.save_dir,
            value=osp.join(base_folder, "*.ckpt"),
        )

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """
        On validation batch (iteration) end
        """
        base_folder = osp.join(self.save_dir, "checkpoints")
        self.wandb_logger.log_file(
            tag="checkpoint",
            base_folder=self.save_dir,
            value=osp.join(base_folder, "*.ckpt"),
        )
