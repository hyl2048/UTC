# coding=utf-8
import argparse
import os
from typing import (Any, Callable, Dict, Iterable, List, NamedTuple, Optional,
                    Tuple, Union)

import numpy as np
import torch
import torch.nn as nn
from lightning.fabric import Fabric
from lightning.fabric.accelerators import Accelerator
from lightning.fabric.loggers import Logger
from lightning.fabric.strategies import DeepSpeedStrategy, Strategy
from lightning_utilities import apply_to_collection
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchmetrics import F1Score, MeanMetric, Metric, MetricCollection
from torchmetrics.utilities.data import to_categorical
from transformers import AdamW, get_linear_schedule_with_warmup


class EvalPrediction(NamedTuple):
    """
    Evaluation output (always contains labels), to be used to compute metrics.

    Parameters:
        predictions (`np.ndarray`): Predictions of the model.
        label_ids (`np.ndarray`): Targets to be matched.
    """

    predictions: Union[np.ndarray, Tuple[np.ndarray]]
    label_ids: Union[np.ndarray, Tuple[np.ndarray]]


class Trainer:
    def __init__(
        self,
        args: argparse.Namespace,
        fabric: Fabric,
        optimizer: Optimizer,
        train_metrics: Metric,
        test_metrics: Metric,
        scheduler: Optional[Callable] = None,
    ) -> None:
        self.fabric = fabric

        self.optimizer = optimizer
        self.scheduler = scheduler

        self.max_epochs = args.max_epochs
        self.max_steps = args.max_steps
        self.max_grad_norm = args.max_grad_norm
        self.grad_accum_steps = args.grad_accum_steps

        self.logging_steps = args.logging_steps
        # self.eval_logging_steps = args.eval_logging_steps
        self.eval_steps = args.eval_steps
        self.save_checkpoint_steps = args.save_checkpoint_steps
        self.checkpoint_dir = args.checkpoint_dir

        self.task_type = args.task_type
        self.num_classes = args.num_classes
        self.ignore_index = args.ignore_index

        self.current_step = 0
        self.global_step = 0
        self.current_epoch = 0

        self.train_loss = self.fabric.to_device(MeanMetric())
        self.train_metrics = self.fabric.to_device(train_metrics)
        self.test_metrics = self.fabric.to_device(test_metrics)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # input_ids, token_type_ids, position_ids, attention_mask, omask_positions, cls_positions, labels = batch

        output = self.model(**batch)
        # output.update({'labels': batch['labels']})
        return output

    def train_step(self, batch: Dict[str, torch.Tensor]):
        # import pdb

        # pdb.set_trace()
        output = self.forward(batch)

        metric_output = apply_to_collection(output, torch.Tensor, lambda x: x.detach())
        self.train_loss.update(metric_output["loss"])
        if self.task_type == "multiclass":
            metric_output["option_logits"] = to_categorical(
                metric_output["option_logits"]
            )
            metric_output["labels"] = to_categorical(batch["labels"])
        else:
            metric_output["labels"] = batch["labels"]

        self.train_metrics(metric_output["option_logits"], metric_output["labels"])
        return output["loss"]

    def train_epoch(self, train_loader: DataLoader, val_loader: DataLoader):
        for i, batch in enumerate(train_loader):
            # Accumulate gradient 8 batches at a time
            is_accumulating = (self.global_step + 1) % self.grad_accum_steps != 0

            with self.fabric.no_backward_sync(self.model, enabled=is_accumulating):

                loss = self.train_step(batch)
                if self.grad_accum_steps > 1:
                    loss = loss / self.grad_accum_steps
                self.fabric.backward(loss)

            if not is_accumulating:
                # Step the optimizer after accumulation phase is over
                if (
                    self.max_grad_norm is not None
                    and self.max_grad_norm > 0
                    and not isinstance(self.fabric.strategy, DeepSpeedStrategy)
                ):
                    self.fabric.clip_gradients(
                        self.model,
                        optimizer=self.optimizer,
                        max_norm=self.max_grad_norm,
                    )
                self.optimizer.step()
                if self.scheduler is not None:
                    self.fabric.log(
                        "lr", self.scheduler.get_last_lr()[0], step=self.current_step
                    )
                    self.scheduler.step()
                else:
                    self.fabric.log(
                        "lr",
                        self.optimizer.param_groups[0]["lr"],
                        step=self.current_step,
                    )

                self.optimizer.zero_grad()
                self.current_step += 1

                if (
                    self.current_step % self.logging_steps == 0
                    and self.fabric.is_global_zero
                ):
                    self.log_info(self.train_loss, self.train_metrics, "train")

                if (
                    self.current_step % self.eval_steps == 0
                    and self.fabric.is_global_zero
                ):
                    self.eval(val_loader)

                if self.current_step % self.save_checkpoint_steps == 0:
                    state = {
                        "model": self.model,
                        "current_step": self.current_step,
                        "current_epoch": self.current_epoch,
                    }
                    self.save(state)

            self.global_step += 1

    def compute_metrics(self, eval_tuple: EvalPrediction):
        # import pdb

        # pdb.set_trace()
        labels = torch.as_tensor(eval_tuple.label_ids, dtype=torch.int64)
        preds = torch.as_tensor(eval_tuple.predictions)
        preds = torch.nn.functional.sigmoid(preds)
        preds = preds[labels != -100].numpy()
        labels = labels[labels != -100].numpy()
        preds = preds > 0.5
        from sklearn.metrics import f1_score

        micro_f1 = f1_score(y_pred=preds, y_true=labels, average="micro")
        macro_f1 = f1_score(y_pred=preds, y_true=labels, average="macro")

        return {"micro_f1": micro_f1, "macro_f1": macro_f1}

    def eval(self, val_loader: DataLoader):
        torch.set_grad_enabled(False)
        test_loss = self.fabric.to_device(MeanMetric())
        sk_logits, sk_lable = torch.Tensor([]).cuda(), torch.Tensor([]).cuda()
        for i, batch in enumerate(val_loader):
            output = self.forward(batch)
            output = apply_to_collection(output, torch.Tensor, lambda x: x.detach())
            test_loss.update(output["loss"])
            if self.task_type == "multiclass":
                output["option_logits"] = to_categorical(output["option_logits"])
                output["labels"] = to_categorical(batch["labels"])
            else:
                output["labels"] = batch["labels"]
            sk_logits = torch.cat((sk_logits, output["option_logits"]), 0)
            sk_lable = torch.cat((sk_lable, output["labels"]), 0)
            self.test_metrics.update(output["option_logits"], output["labels"])

        # import pdb

        # pdb.set_trace()
        eval_tuple = EvalPrediction(
            predictions=sk_logits.cpu().numpy(), label_ids=sk_lable.cpu().numpy()
        )
        res = self.compute_metrics(eval_tuple)
        self.fabric.print(
            "eval_macro_f1:{}, eval_mircro_f1:{}".format(
                res["macro_f1"], res["micro_f1"]
            )
        )
        self.log_info(test_loss, self.test_metrics, "eval")

        torch.set_grad_enabled(True)

    def fit(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ):
        train_loader = self.fabric.setup_dataloaders(train_loader)
        if val_loader is not None:
            val_loader = self.fabric.setup_dataloaders(
                val_loader, use_distributed_sampler=False
            )
        self.model, self.optimizer = self.fabric.setup(model, self.optimizer)

        self.model.train()
        for epoch in range(1, self.max_epochs + 1):
            self.current_epoch = epoch
            self.train_epoch(train_loader, val_loader)

    def test(self, model: nn.Module, val_loader: DataLoader):
        val_loader = self.fabric.setup_dataloaders(
            val_loader, use_distributed_sampler=False
        )
        self.model = self.fabric.setup(model)
        self.eval(val_loader)

    def progbar_wrapper(self, iterable: Iterable, total: int, **kwargs: Any):
        if self.fabric.is_global_zero:
            return tqdm(iterable, total=total, **kwargs)
        return iterable

    def log_info(
        self, loss_metric: MeanMetric, f1_metrics: MetricCollection, mode: str = "train"
    ):

        loss = loss_metric.compute()
        metrics = f1_metrics.compute()
        log_metrics = {
            f"{mode}_loss": loss,
            f"{mode}_micro_f1": metrics["micro_f1"],
            f"{mode}_macro_f1": metrics["macro_f1"],
        }
        loss_metric.reset()
        f1_metrics.reset()
        self.fabric.log_dict(log_metrics, self.current_step)
        log_metrics = apply_to_collection(log_metrics, torch.Tensor, lambda x: x.item())
        self.fabric.print(
            "{} steps: {}, loss: {}, micro_f1: {}, macro_f1: {}".format(
                mode, self.current_step, *log_metrics.values()
            )
        )

    def save(self, state):
        self.fabric.save(
            os.path.join(self.checkpoint_dir, f"step-{self.current_step:04d}.ckpt"),
            state,
        )
