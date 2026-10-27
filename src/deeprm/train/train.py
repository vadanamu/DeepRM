"""
DeepRM Training Module

This module provides the training functionality for the DeepRM Transformer model.
It includes the Trainer class, which handles the training loop, evaluation, and checkpointing.
"""

import argparse
import gc
import glob
import os
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from deeprm.inference.inference import _build_model, _model_name, load_model_file
from deeprm.model import deeprm_v12
from deeprm.train.train_dataloader import load_dataset
from deeprm.utils.logging import get_logger, is_terminal, use_color

log = get_logger(__name__)

try:
    import torchmetrics.classification as cm

    TORCHMETRICS_AVAILABLE = True
except Exception:
    TORCHMETRICS_AVAILABLE = False


# The architecture values that deeprm train run does not offer as options. They are written into the model file, as
# in the bundled v1.2 weights, so that deeprm call run builds the same model (the v1.2 model has no dwell/base-quality
# input and one output, a logit).
DEEPRM_V12_FIXED = {
    "model_type": deeprm_v12.ARCHITECTURE,
    "no_bq": True,
    "no_dwell": True,
    "n_label": 1,
    "n_class": 1,
    "output_logits": True,
    "kmer_mode": "5mer",
    "t_act": "gelu",  # not used by deeprm_v12 (SwiGLU); deeprm call run reads the key
}

# Progress-bar colour of each rank (repeated after 8 ranks)
_BAR_COLOURS = ("red", "green", "blue", "yellow", "magenta", "cyan", "white", "black")


def _bar_kwargs(rank: int) -> dict:
    """The tqdm arguments that place and colour the progress bar of `rank`. Both write terminal control codes, so
    they are given only on a terminal: position (one line per rank) when standard error is a terminal, colour when
    it is a terminal and NO_COLOR is unset. In a log file every bar is plain text."""
    kwargs = {}
    if is_terminal(sys.stderr):
        kwargs["position"] = rank
    if use_color(sys.stderr):
        kwargs["colour"] = _BAR_COLOURS[rank % len(_BAR_COLOURS)]
    return kwargs


def add_arguments(parser: argparse.ArgumentParser):
    """
    Adds command-line arguments.

    Args:
        parser (argparse.ArgumentParser): Argument parser to which arguments will be added.
    Returns:
        None
    """
    parser.add_argument(
        "--gpu",
        dest="num_gpu",
        type=int,
        default=None,
        help="Number of GPUs to use (default: all GPUs, or as many as --gpu-pool names)",
    )
    parser.add_argument("--batch", dest="batch_size", type=int, default=1024, help="Batch size on each GPU")
    parser.add_argument(
        "--eval_batch",
        dest="eval_batch_size",
        type=int,
        default=None,
        help="Has no effect in this version (validation uses --batch)",
    )
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of epochs to train")
    parser.add_argument(
        "--data", dest="data_path", type=str, required=True, help="Folder written by deeprm train compile"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output folder; each run writes to a subfolder named after the run (--model-name)",
    )
    parser.add_argument(
        "--tb",
        dest="tb_path",
        type=str,
        default=None,
        help="Folder for the TensorBoard log; each run writes to a subfolder named after the run (default: "
        "tensorboard_log in the run folder)",
    )
    # Kept, and hidden, so that "--model deeprm_model" is refused: without it argparse would read "--model" as an
    # abbreviation of "--model-name".
    parser.add_argument("--model", dest="model_type", type=str, default=deeprm_v12.ARCHITECTURE, help=argparse.SUPPRESS)
    parser.add_argument(
        "--es-delta", type=float, default=1e-5, help="Smallest drop in validation loss that counts as an improvement"
    )
    parser.add_argument(
        "--es-patience",
        type=int,
        default=50,
        help="Stop when the validation loss has not improved for more than this many epochs (after --es-start)",
    )
    parser.add_argument(
        "--es-start", type=int, default=1000, help="Epoch after which early stopping can stop the training"
    )
    parser.add_argument(
        "--disk-shard-size",
        type=int,
        default=None,
        help="Rows per file of the compiled dataset (default: read from the first file)",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed (default: a random seed, written to the log)"
    )
    parser.add_argument("--enc-dim", type=int, default=512, help="Width of the transformer layers")
    parser.add_argument("--lin-dim", type=int, default=1024, help="Feed-forward size of the transformer layers")
    parser.add_argument("--head", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--enc-layer", type=int, default=6, help="Number of transformer layers")
    parser.add_argument(
        "--lin-layer", type=int, default=6, help="Depth of the output head (the number of its full-width layers)"
    )
    parser.add_argument("--enc-dropout", type=float, default=0.1, help="Dropout rate in the transformer layers")
    parser.add_argument("--lin-dropout", type=float, default=0.1, help="Dropout rate in the output head")
    parser.add_argument("--period", type=int, default=30, help="Has no effect in this version")
    parser.add_argument(
        "--buffer_size", dest="shuffle_buffer_size", type=int, default=160000, help="Rows held in the shuffle buffer"
    )
    parser.add_argument("--kmer-size", type=int, default=5, help="Bases per k-mer token")
    parser.add_argument(
        "--signal-size", type=int, default=30, help="Signal samples per window; must equal --stride x --kmer-size"
    )
    parser.add_argument("--block-len", type=int, default=17, help="Bases per row; the model scores the centre base")
    parser.add_argument("--seq-len", type=int, default=200, help="Signal windows per row")
    parser.add_argument(
        "--lin-act",
        type=str,
        default="gelu",
        choices=list(deeprm_v12.HEAD_ACTIVATIONS),
        help="Activation function of the output head (the v1.2 weights use gelu)",
    )
    parser.add_argument(
        "--lr-step",
        type=int,
        default=4000,
        help="Length of the cosine learning-rate schedule, in scheduler steps (with --rlrop: its patience)",
    )
    parser.add_argument(
        "--lr-interval", type=int, default=100, help="Training steps between two learning-rate scheduler steps"
    )
    parser.add_argument("--weight-decay", type=float, default=0.1, help="Weight decay of the AdamW optimizer")
    parser.add_argument(
        "--class-ratio",
        type=int,
        default=None,
        help="Negatives per positive in training (default: the ratio of negative to positive data files)",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="Training steps between two training-loss entries in the TensorBoard log",
    )
    parser.add_argument("--eval-interval", type=int, default=1000, help="Training steps between two validation runs")
    parser.add_argument(
        "--save-interval",
        type=int,
        default=None,
        help="Training steps between two checks that save the model if the validation loss improved (default: "
        "--eval-interval)",
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
        help="Largest gradient norm; larger gradients are scaled down (0: no clipping)",
    )
    parser.add_argument("--profiler", type=int, default=0, help="Has no effect in this version")
    parser.add_argument("--pin-memory", type=int, default=1, help="Use pinned memory in the data loader (1) or not (0)")
    parser.add_argument(
        "--yield_period",
        type=int,
        default=None,
        help="Rows taken from the shuffle buffer at a time; at most --buffer_size (default: --disk-shard-size)",
    )
    parser.add_argument(
        "--rlrop",
        type=float,
        default=None,
        help="Halve the learning rate when the validation loss stops improving by this relative amount, instead of "
        "the cosine schedule",
    )
    parser.add_argument(
        "--loss",
        type=str,
        choices=("BCEWL",),
        default="BCEWL",
        help="Loss function: BCEWL (binary cross-entropy on logits), the only choice",
    )
    parser.add_argument("--score-feature", type=bool, default=False, help="Has no effect in this version")
    parser.add_argument(
        "--gpu-pool", type=int, nargs="+", default=None, help="IDs of the GPUs to use (default: the first --gpu GPUs)"
    )
    parser.add_argument("--cut-overlap", type=bool, default=False, help="Has no effect in this version")
    parser.add_argument(
        "--load-checkpoint",
        type=str,
        default=None,
        help="Model file or checkpoint to start from; a file without optimizer state (such as the bundled weights) "
        "loads its weights only",
    )
    parser.add_argument(
        "--workers", dest="num_workers", type=int, default=8, help="Data-loading worker processes per GPU"
    )
    parser.add_argument(
        "--prefetch", type=int, default=512, help="Batches each data-loading worker prepares in advance"
    )
    parser.add_argument(
        "--stride", dest="signal_stride", type=int, default=6, help="Signal samples between the starts of two windows"
    )
    parser.add_argument(
        "--load-weight-only",
        action="store_true",
        default=False,
        help="With --load-checkpoint: load the model weights only, not the optimizer and learning-rate state",
    )
    parser.add_argument(
        "--override-lr",
        action="store_true",
        default=False,
        help="With --load-checkpoint: use --lr instead of the saved learning rate, and do not load the saved "
        "learning-rate schedule",
    )
    parser.add_argument(
        "--comment",
        type=str,
        default="None",
        help="Text in the default run name, <architecture>-<comment>-<date and time>",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Name of the run: its subfolder of --output and the start of its model file names (default: "
        "deeprm_v12-<comment>-<date and time>)",
    )
    return None


def main(args: argparse.Namespace):
    """
    Main function to start the training process (the v1.2 architecture deeprm_v12).

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        None
    """
    _deeprm_v12_args(args)
    strfttime = time.strftime("%Y%m%d-%H%M%S")

    if args.num_gpu is None:
        if args.gpu_pool is None:
            args.num_gpu = torch.cuda.device_count()
        else:
            args.num_gpu = len(args.gpu_pool)

    if args.num_gpu < 1:
        raise ValueError(
            "deeprm train needs at least one CUDA GPU (it trains with NCCL), but PyTorch sees none. "
            "Run `deeprm check` to see why."
        )
    if args.gpu_pool is None:
        args.gpu_pool = list(range(args.num_gpu))
    else:
        if len(args.gpu_pool) < args.num_gpu:
            raise ValueError("GPU Pool should be the same or larger than the number of GPUs to use.")

    if args.disk_shard_size is None:
        sample_file = glob.glob(os.path.join(args.data_path, "train", "pos", "*.npz"))[0]
        with np.load(sample_file) as f:
            args.disk_shard_size = f["kmer_token"].shape[0]
        log.info(f"Setting disk shard size to {args.disk_shard_size}.")

    if args.eval_batch_size is None:
        args.eval_batch_size = args.batch_size * 4
    if args.model_name is None:
        args.model_name = f"{args.model_type}-{args.comment}-{strfttime}"
    if args.yield_period is None:
        args.yield_period = args.disk_shard_size
    if args.save_interval is None:
        args.save_interval = args.eval_interval

    args.output = os.path.join(args.output, args.model_name)

    if args.tb_path is None:
        args.tb_path = os.path.join(args.output, "tensorboard_log")
    else:
        args.tb_path = os.path.join(args.tb_path, args.model_name)

    args_dict = vars(args)
    torch.multiprocessing.set_sharing_strategy("file_system")
    os.makedirs(args_dict["output"], exist_ok=True)
    os.makedirs(args_dict["tb_path"], exist_ok=True)
    if args_dict["seed"] is None:
        args_dict["seed"] = np.random.randint(0, 10000000)
    log.info("Training Program Started.")
    log.info(f"Seed: {args_dict['seed']}")
    log.info(f"Using {args_dict['num_gpu']} GPUs.")
    try:
        mp.spawn(main_worker, nprocs=args_dict["num_gpu"], args=(args_dict,))
    except Exception as e:
        log.error("Training Program Failed.")
        raise e
    log.info("Training Program Complete.")

    return None


def _deeprm_v12_args(args: argparse.Namespace) -> argparse.Namespace:
    """Checks the model options and adds DEEPRM_V12_FIXED. Runs before the GPU check, so a refusal needs no GPU."""
    if args.model_type == "deeprm_model":
        raise ValueError(
            "--model deeprm_model: deeprm train run trains only the v1.2 architecture, deeprm_v12. To train the v1.1 "
            "model, use DeepRM v1.1 (pip install deeprm==1.1.3); for deeprm_v12, leave --model out."
        )
    if args.model_type != deeprm_v12.ARCHITECTURE:
        raise ValueError(
            f"--model {args.model_type}: deeprm train run trains only the v1.2 architecture, deeprm_v12. "
            "Leave --model out."
        )
    if args.signal_size != args.signal_stride * args.kmer_size:
        raise ValueError(
            f"--signal-size {args.signal_size} must equal --stride x --kmer-size "
            f"({args.signal_stride} x {args.kmer_size} = {args.signal_stride * args.kmer_size})."
        )
    if args.load_checkpoint is not None:
        save_dict = load_model_file(args.load_checkpoint)
        # a v1.1 file is refused
        _model_name(save_dict.get("model_config", {}), None, save_dict.get("model_state_dict"), "--load-checkpoint")
        if "optimizer_state_dict" not in save_dict and not args.load_weight_only:
            log.info("%s holds no optimizer state (a weight file): loading its weights only.", args.load_checkpoint)
            args.load_weight_only = True
        del save_dict
    vars(args).update(DEEPRM_V12_FIXED)
    return args


class Trainer:
    def __init__(
        self,
        rank: int,
        gpu_id: int,
        model: torch.nn.Module,
        train_loader,
        val_loader,
        optimizer: torch.optim.Optimizer,
        scheduler,
        loss_func: torch.nn.Module,
        grad_clip: float,
        metric_func_dict: dict,
        checkpoint_path: str,
        tb_path: str,
        es_start: int,
        es_patience: int,
        es_delta: float,
        model_name: str,
        num_gpu: int,
        lr_interval: int,
        eval_interval: int,
        log_interval: int,
        save_interval: int,
        model_config: dict = None,
        soft_label: float = None,
        score_feature: bool = False,
        cut_overlap: bool = False,
        signal_stride: int = 6,
        **kwargs,
    ) -> None:
        """
        Initializes the Trainer class.

        Args:
            rank (int): Rank of the current process.
            gpu_id (int): GPU ID to use.
            model (torch.nn.Module): Model to train.
            train_loader (torch.utils.data.DataLoader): DataLoader for training data.
                (deeprm.train.train_dataloader.NanoporeDataLoader)
            val_loader (torch.utils.data.DataLoader): DataLoader for validation data.
                (deeprm.train.train_dataloader.NanoporeDataLoader)
            optimizer (torch.optim.Optimizer): Optimizer for training.
            scheduler: Learning rate scheduler.
            loss_func (torch.nn.Module): Loss function.
            grad_clip (float): Gradient clipping value.
            metric_func_dict (dict): Dictionary of metric functions.
            checkpoint_path (str): Path to save checkpoints.
            tb_path (str): Path for TensorBoard logs.
            es_start (int): Epoch to start early stopping.
            es_patience (int): Patience for early stopping.
            es_delta (float): Delta for early stopping.
            model_name (str): Name of the model.
            num_gpu (int): Number of GPUs to use.
            lr_interval (int): Interval for learning rate updates.
            eval_interval (int): Interval for evaluation.
            log_interval (int): Interval for logging.
            save_interval (int): Interval for saving checkpoints.
            model_config (dict): Model configuration dictionary. Defaults to None. (optional)
            soft_label (float): Soft label value. Defaults to None. (optional)
            score_feature (bool): Score feature flag. Defaults to False. (optional)
            cut_overlap (bool): Cut overlap flag. Defaults to False. (optional)
            signal_stride (int): Signal stride. Defaults to 6. (optional)
            **kwargs: Additional keyword arguments.
        """
        self.rank = rank
        self.gpu_id = gpu_id
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.grad_clip = grad_clip
        self.scheduler = scheduler
        self.lr_interval = lr_interval
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.eval_interval = eval_interval
        self.metric_func_dict = metric_func_dict
        self.checkpoint_path = checkpoint_path
        self.es_start = es_start
        self.es_patience = es_patience
        self.es_delta = es_delta
        self.continue_training = 1
        self.best_val_loss = np.inf
        self.best_val_loss_epoch = 0
        self.current_val_loss = 0
        self.current_val_metric_dict = {}
        self.current_epoch = 0
        self.current_batch = 0
        self.current_step = 0
        self.current_batch_loss = 0
        self.current_interval_loss = 0
        self.current_interval_losses = []
        self.current_lr = 0
        self.pbar = None
        self.model_name = model_name
        self.num_gpu = num_gpu
        self.model_config = model_config
        self.devname = f"{os.uname()[1]}-{self.gpu_id}"
        self.tb_path = tb_path
        self.soft_label = soft_label
        self.score_feature = score_feature
        self.cut_overlap = cut_overlap
        self.signal_stride = signal_stride
        self.histogram = False
        if self.rank == 0 and self.tb_path is not None:
            self.tb_writer = SummaryWriter(tb_path)
        else:
            self.tb_writer = None

        self.eval_sources, self.eval_targets = self._cache_eval_data()

    def _cache_eval_data(self):
        """
        Caches evaluation data for faster evaluation.

        Returns:
            tuple: Cached sources and targets for evaluation.
        """
        sources = []
        targets = []
        with torch.no_grad():
            for source, target in self.val_loader:
                sources.append(source)
                targets.append(target)
        return sources, targets

    def _feed_model(self, source, target):
        """
        Feeds data to the model and returns the output and target.

        Args:
            source (dict): Source data.
            target (torch.Tensor): Target data.

        Returns:
            tuple: Model output and target.
        """
        target = target.to(torch.float32).to(self.gpu_id)
        src_kmer = source["kmer_token"].to(self.gpu_id)
        src_signal = source["signal_token"].to(self.gpu_id)
        src_seg_len = source["segment_len"].to(self.gpu_id)

        # deeprm_v12 takes no dwell/base-quality input and returns (rows, 1) logits; the target is (rows,)
        output = self.model(src_kmer, src_signal, src_seg_len).squeeze(-1)
        return output, target

    def _run_batch(self, source, target):
        """
        Runs a single batch of training.

        Args:
            source (dict): Source data.
            target (torch.Tensor): Target data.

        Returns:
            None
        """
        self.optimizer.zero_grad()
        output, target = self._feed_model(source, target)
        loss = self.loss_func(output, target)
        loss.backward()
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()
        self.current_batch_loss = loss.item()
        dist.barrier()
        time.sleep(0.0001 * self.gpu_id)
        evaltext = (
            f"LR {self.current_lr:.3E} | T-Loss {self.current_batch_loss:.3E} | V-Loss {self.current_val_loss:.3E} | "
        )
        evaltext += " | ".join(
            [f"{k.upper()} {v:.3E}" for k, v in self.current_val_metric_dict.items() if (k in ["auroc", "ap"])]
        )
        self.pbar.update(1)
        self.pbar.set_postfix_str(evaltext)
        return None

    def _run_epoch(self):
        """
        Runs a single epoch of training.

        Returns:
            None
        """
        self.train_loader.set_epoch(self.current_epoch)
        self.val_loader.set_epoch(self.current_epoch)
        self.current_batch = 0
        self.model.train()
        self.current_lr = self.optimizer.param_groups[0]["lr"]
        dist.barrier()
        time.sleep(0.03 * self.gpu_id)
        with tqdm.tqdm(
            total=len(self.train_loader),
            desc=f"[GPU {self.gpu_id}] Epoch {self.current_epoch}",
            smoothing=0,
            **_bar_kwargs(self.rank),
        ) as self.pbar:
            for source, targets in self.train_loader:
                self._run_batch(source, targets)
                self.current_interval_losses.append(self.current_batch_loss)

                if self.current_step % self.log_interval == 0:
                    current_interval_loss = np.mean(self.current_interval_losses)
                    self.current_interval_losses = []
                    dist.barrier()
                    current_interval_loss = torch.tensor(current_interval_loss).to(self.gpu_id)
                    dist.all_reduce(current_interval_loss, op=dist.ReduceOp.SUM)
                    self.current_interval_loss = current_interval_loss / self.num_gpu
                    if self.rank == 0:
                        self.tb_writer.add_scalar("Loss", self.current_interval_loss, self.current_step)
                        self.tb_writer.add_scalar("Learning_Rate", self.current_lr, self.current_step)
                    dist.barrier()

                if self.current_step % self.eval_interval == 0 and self.current_step > 0:
                    self._run_eval()

                if self.current_step % self.save_interval == 0 and self.current_step > 0:
                    dist.barrier()
                    if self.rank == 0:
                        if self.histogram:
                            try:
                                for name, parameter in self.model.named_parameters():
                                    self.tb_writer.add_histogram(
                                        name, parameter.clone().cpu().data.numpy(), self.current_step
                                    )
                            except Exception as e:
                                log.warning(f"Error adding histogram: {e}")
                                pass
                        self._save_checkpoint()
                    dist.barrier()

                if self.current_step % self.lr_interval == 0:
                    if self.scheduler.__class__.__name__ == "ReduceLROnPlateau":
                        self.scheduler.step(self.current_val_loss)
                    else:
                        self.scheduler.step()
                    self.current_lr = self.optimizer.param_groups[0]["lr"]

                self.current_batch += 1
                self.current_step += 1

        gc.collect()
        return None

    def _run_eval(self):
        """
        Runs evaluation on the validation dataset.

        Returns:
            None
        """
        self.model.eval()
        val_loss = []
        outputs = []
        with torch.no_grad():
            for source, target in zip(self.eval_sources, self.eval_targets):
                output, target = self._feed_model(source, target)
                loss = self.loss_func(output, target)
                val_loss.append(loss.item())
                outputs.append(output)

        val_loss = np.mean(val_loss)
        outputs = torch.cat(outputs, dim=0)
        targets = torch.cat(self.eval_targets, dim=0)
        targets = targets.to(torch.long).to(self.gpu_id)
        metric_dict = {}

        for metric_name, metric_func in self.metric_func_dict.items():
            # the model returns logits; torchmetrics would read logits that all lie in [0, 1] as probabilities
            metric_dict[metric_name] = metric_func(torch.sigmoid(outputs), targets)

        dist.barrier()
        val_loss = torch.tensor(val_loss).to(self.gpu_id)
        dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)
        val_loss = val_loss / self.num_gpu
        for key, value in metric_dict.items():
            metric = value.clone().detach().to(self.gpu_id)
            dist.all_reduce(metric, op=dist.ReduceOp.SUM)
            metric_dict[key] = metric / self.num_gpu
        self.current_val_loss = val_loss
        self.current_val_metric_dict = metric_dict

        if self.rank == 0:
            self.tb_writer.add_scalar("Val_Loss", val_loss, self.current_step)
            for key, value in metric_dict.items():
                self.tb_writer.add_scalar(f"Val_{key}", value, self.current_step)
        dist.barrier()

        self.model.train()
        return None

    def _save_checkpoint(self):
        """
        Saves the model checkpoint if validation loss improves.

        Returns:
            None
        """

        if self.best_val_loss - self.current_val_loss > self.es_delta:
            ## Save Model if Improved
            self.best_val_loss = self.current_val_loss
            self.best_val_loss_epoch = self.current_epoch
            torch.save(
                {
                    "model_state_dict": self.model.module.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scheduler_state_dict": self.scheduler.state_dict(),
                    "val_loss": self.current_val_loss,
                    "metric_dict": self.current_val_metric_dict,
                    "model_config": self.model_config,
                },
                f"{self.checkpoint_path}/{self.model_name}-{self.current_epoch}-{self.current_step}.pt",
            )
            self.continue_training = 1

        elif self.current_epoch > self.es_start and self.current_epoch - self.best_val_loss_epoch > self.es_patience:
            self.continue_training = 0

        else:
            self.continue_training = 1

        return None

    def train(self, max_epochs: int):
        """
        Trains the model for a specified number of epochs.

        Args:
            max_epochs (int): The maximum number of epochs to train the model.

        Returns:
            None
        """
        for epoch in range(max_epochs):
            dist.barrier()
            self.current_epoch = epoch
            self._run_epoch()
            if self.continue_training == 0:
                log.error(f"Early Stopping at Epoch {self.current_epoch}")
                break
        if self.rank == 0:
            self.tb_writer.flush()
        return None

    ## END of Class NanoporeTrainer


def setup_ddp(rank, world_size, gpu_id):
    """
    Sets up Distributed Data Parallel (DDP) for multi-GPU training.

    Args:
        rank (int): Rank of the current process.
        world_size (int): Total number of processes.
        gpu_id (int): GPU ID to use.

    Returns:
        None
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(gpu_id)
    return None


def prepare_dataloader(data_path, rank, num_gpu, num_workers, **kwargs):
    """
    Prepares the DataLoader for training and validation datasets.

    Args:
        data_path (str): Path to the dataset directory.
        rank (int): Rank of the current process.
        num_gpu (int): Number of GPUs to use.
        num_workers (int): Number of worker processes.
        **kwargs: Additional keyword arguments.

    Returns:
        tuple: A tuple containing the training and validation DataLoaders.
    """
    if rank == 0:
        log.info(f"Total number of dataloader workers: {num_workers * num_gpu}")
    train_pos_data_path = f"{data_path}/train/pos"
    train_neg_data_path = f"{data_path}/train/neg"
    val_pos_data_path = f"{data_path}/val/pos"
    val_neg_data_path = f"{data_path}/val/neg"

    train_loader = load_dataset(
        pos_data_path=train_pos_data_path,
        neg_data_path=train_neg_data_path,
        rank=rank,
        num_replicas=num_gpu,
        num_workers=num_workers,
        shuffle=True,
        drop_last=True,
        **kwargs,
    )

    val_loader = load_dataset(
        pos_data_path=val_pos_data_path,
        neg_data_path=val_neg_data_path,
        rank=rank,
        num_replicas=num_gpu,
        num_workers=num_workers,
        shuffle=False,
        drop_last=False,
        **kwargs,
    )

    return train_loader, val_loader


def main_worker(rank, args_dict):
    """
    Main worker function for training the model.

    Args:
        rank (int): Rank of the current process.
        args_dict (dict): Dictionary of command-line arguments.

    Returns:
        None
    """
    gpu_id = args_dict["gpu_pool"][rank]
    setup_ddp(rank, args_dict["num_gpu"], gpu_id)
    model = _build_model(deeprm_v12.TransformerModel, args_dict)  # the same builder as deeprm call run
    if rank == 0:
        total_params = 0
        for name, parameter in model.named_parameters():
            params = parameter.numel()
            total_params += params
        log.info(f"Total Params: {total_params:,}")

    model = model.to(gpu_id)

    if args_dict["load_checkpoint"] is not None:
        # every tensor loads onto this process's GPU, whatever GPU the file was saved from
        save_dict = load_model_file(args_dict["load_checkpoint"], map_location=f"cuda:{gpu_id}")
        model.load_state_dict(state_dict=save_dict["model_state_dict"])
        if args_dict["load_weight_only"]:
            for param in model.parameters():
                param.requires_grad = True
    else:
        save_dict = {}

    model = DDP(model, device_ids=[gpu_id], output_device=gpu_id, find_unused_parameters=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args_dict["lr"], weight_decay=args_dict["weight_decay"])

    if args_dict["load_checkpoint"] is not None:
        if not args_dict["load_weight_only"]:
            optimizer.load_state_dict(save_dict["optimizer_state_dict"])
            if args_dict["override_lr"]:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = args_dict["lr"]

    if args_dict["rlrop"] is not None:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=args_dict["lr_step"],
            threshold=args_dict["rlrop"],
            threshold_mode="rel",
            cooldown=0,
            min_lr=1e-6,
            eps=1e-8,
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args_dict["lr_step"], eta_min=1e-6)

    if args_dict["load_checkpoint"] is not None:
        if not (args_dict["load_weight_only"] or args_dict["override_lr"]):
            scheduler.load_state_dict(save_dict["scheduler_state_dict"])

    if args_dict["load_checkpoint"] is not None:
        save_dict.clear()
        del save_dict
        gc.collect()

    loss_func = torch.nn.BCEWithLogitsLoss()  # the model returns logits (output_logits=True)

    if TORCHMETRICS_AVAILABLE:
        metric_func_dict = {
            "acc": cm.BinaryAccuracy().to(gpu_id),
            "auroc": cm.BinaryAUROC().to(gpu_id),
            "ap": cm.BinaryAveragePrecision().to(gpu_id),
            "f-1": cm.BinaryF1Score().to(gpu_id),
        }
    else:
        log.warning("torchmetrics is not available. Some metrics will not be computed.")
        metric_func_dict = {}

    args_dict["checkpoint_path"] = args_dict["output"]

    train_loader, val_loader = prepare_dataloader(rank=rank, gpu_id=gpu_id, **args_dict)
    trainer = Trainer(
        rank=rank,
        gpu_id=gpu_id,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_func=loss_func,
        metric_func_dict=metric_func_dict,
        model_config=args_dict,
        **args_dict,
    )

    log.info(f"[GPU {gpu_id}] Trainer Setup Complete.")
    trainer.train(args_dict["epochs"])
    log.info(f"[GPU {gpu_id}] Training Loop Complete.")
    dist.destroy_process_group()
    return None
