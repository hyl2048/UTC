# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field

import paddle
import paddle.nn as nn
import paddle.optimizer
import paddlenlp
import ray
from paddle.metric import Accuracy
from paddle.optimizer import (
    LBFGS,
    SGD,
    Adadelta,
    Adagrad,
    Adamax,
    AdamW,
    Lamb,
    Momentum,
    RMSProp,
    adam,
)
from paddle.optimizer.lr import (
    CosineAnnealingDecay,
    CyclicLR,
    ExponentialDecay,
    InverseTimeDecay,
    LambdaDecay,
    LinearWarmup,
    MultiplicativeDecay,
    MultiStepDecay,
    NaturalExpDecay,
    NoamDecay,
    OneCycleLR,
    PiecewiseDecay,
    PolynomialDecay,
    ReduceOnPlateau,
    StepDecay,
)
from paddle.static import InputSpec
from paddlenlp.datasets import load_dataset
from paddlenlp.prompt import (
    PromptModelForSequenceClassification,
    PromptTrainer,
    PromptTuningArguments,
    UTCTemplate,
)
from paddlenlp.trainer import PdArgumentParser
from paddlenlp.trainer.integrations import (
    AutoNLPCallback,
    TrainerCallback,
    VisualDLCallback,
)
from paddlenlp.trainer.trainer_callback import TrainerControl, TrainerState
from paddlenlp.trainer.training_args import TrainingArguments
from paddlenlp.transformers import UTC, AutoTokenizer, export_model
from ray import train, tune
from ray.air.integrations.wandb import WandbLoggerCallback, setup_wandb
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.bayesopt import BayesOptSearch
from sklearn.metrics import f1_score
from utils import HLCLoss, UTCLoss, read_local_dataset
from visualdl import LogWriter
from visualdl.server import app

import wandb


class InspectStateCallback(TrainerCallback):

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        if state.log_history:

            train.report(
                {
                    "loss": state.log_history[-2]["loss"],
                    "eval_loss": state.log_history[-1]["eval_loss"],
                }
            )


@dataclass
class DataArguments:
    dataset_path: str = field(
        default="/root/UTC/utc_paddle/data/cail_mul_label_mul_classify",
        metadata={
            "help": "Local dataset directory including train.txt, dev.txt and label.txt (optional)."
        },
    )
    train_file: str = field(
        default="test.txt", metadata={"help": "Train dataset file name."}
    )
    dev_file: str = field(
        default="dev.txt", metadata={"help": "Dev dataset file name."}
    )
    threshold: float = field(
        default=0.5, metadata={"help": "The threshold to produce predictions."}
    )
    single_label: str = field(
        default=False, metadata={"help": "Predict exactly one label per sample."}
    )


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="/root/UTC/utc_paddle/models/utc-base",
        metadata={
            "help": "The build-in pretrained UTC model name or path to its checkpoints, such as "
            "`utc-xbase`, `utc-base`, `utc-medium`, `utc-mini`, `utc-micro`, `utc-nano` and `utc-pico`."
        },
    )
    export_type: str = field(
        default="paddle",
        metadata={"help": "The type to export. Support `paddle` and `onnx`."},
    )
    export_model_dir: str = field(
        default="/root/UTC/utc_paddle/checkpoint/model_best",
        metadata={"help": "The export model path."},
    )


def train_function(config):

    constant_config = {
        "device": "gpu",
        "logging_steps": 1,
        "save_steps": 500,
        "eval_steps": 1,
        "model_name_or_path": "models/utc-base",
        "output_dir": "./checkpoint/model_best",
        "dataset_path": "data/cail_mul_label_mul_classify",
        "max_seq_length": 512,
        "per_device_eval_batch_size": 32,
        "export_model_dir": "./checkpoint/model_best",
        "disable_tqdm": True,
        "metric_for_best_model": "macro_f1",
        "load_best_model_at_end": True,
        "save_total_limit": 1,
        "evaluation_strategy": "steps",
        "save_strategy": "steps",
        "do_train": True,
        "do_export": True,
        "seed": 42,
        "gradient_accumulation_steps": 1,
        "per_device_train_batch_size": 64,
        "optimizer": "AdamW",
        "scheduler": "LinearWarmup",
        "num_train_epochs": 3,
        "max_grad_norm": 1,
        "warmup_steps": 20,
        "lr_end": 1e-7,
        "start_lr": 0,
    }
    config.update(constant_config)
    # Parse the arguments.
    parser = PdArgumentParser((ModelArguments, DataArguments, PromptTuningArguments))
    model_args, data_args, training_args = parser.parse_dict(config)

    training_args.print_config(model_args, "Model")
    training_args.print_config(data_args, "Data")
    # paddle.set_device(training_args.device)

    # Load the pretrained language model.
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    model = UTC.from_pretrained(model_args.model_name_or_path)
    print("gpu using", paddle.device.get_device())
    # Define template for preprocess and verbalizer for postprocess.
    template = UTCTemplate(tokenizer, training_args.max_seq_length)

    # Load and preprocess dataset.
    train_ds = load_dataset(
        read_local_dataset,
        data_path=data_args.dataset_path,
        data_file=data_args.train_file,
        lazy=False,
    )
    dev_ds = load_dataset(
        read_local_dataset,
        data_path=data_args.dataset_path,
        data_file=data_args.dev_file,
        lazy=False,
    )

    # Define the criterion.
    criterion = UTCLoss()

    # Initialize the prompt model.
    prompt_model = PromptModelForSequenceClassification(
        model,
        template,
        None,
        freeze_plm=training_args.freeze_plm,
        freeze_dropout=training_args.freeze_dropout,
    )

    # Define the metric function.
    def compute_metrics_single_label(eval_preds):
        labels = paddle.to_tensor(eval_preds.label_ids, dtype="int64")
        preds = paddle.to_tensor(eval_preds.predictions)
        preds = paddle.nn.functional.softmax(preds, axis=-1)
        labels = paddle.argmax(labels, axis=-1)
        metric = Accuracy()
        correct = metric.compute(preds, labels)
        metric.update(correct)
        acc = metric.accumulate()
        return {"accuracy": acc}

    def compute_metrics(eval_preds):
        labels = paddle.to_tensor(eval_preds.label_ids, dtype="int64")
        preds = paddle.to_tensor(eval_preds.predictions)
        preds = paddle.nn.functional.sigmoid(preds)
        preds = preds[labels != -100].numpy()
        labels = labels[labels != -100].numpy()
        preds = preds > data_args.threshold

        micro_f1 = f1_score(y_pred=preds, y_true=labels, average="micro")
        macro_f1 = f1_score(y_pred=preds, y_true=labels, average="macro")

        return {"micro_f1": micro_f1, "macro_f1": macro_f1}

    decay_parameters = [
        p.name
        for n, p in model.named_parameters()
        if not any(nd in n for nd in ["bias", "norm"])
    ]

    def apply_decay_param_fun(x):
        return x in decay_parameters

    if config["scheduler"] == "LinearWarmup":
        scheduler = LinearWarmup(
            learning_rate=config["lr"],
            warmup_steps=config["warmup_steps"],
            start_lr=config["start_lr"],
            end_lr=config["lr_end"],
        )

    if config["optimizer"] == "AdamW":
        optimizer = AdamW(
            learning_rate=scheduler,
            parameters=model.parameters(),
            apply_decay_param_fun=apply_decay_param_fun,
            grad_clip=(
                nn.ClipGradByGlobalNorm(config["max_grad_norm"])
                if config["max_grad_norm"] > 0
                else None
            ),
        )

    trainer = PromptTrainer(
        model=prompt_model,
        tokenizer=tokenizer,
        args=training_args,
        criterion=criterion,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        callbacks=[InspectStateCallback],
        compute_metrics=(
            compute_metrics_single_label if data_args.single_label else compute_metrics
        ),
        optimizers=(optimizer, scheduler),
    )

    # Training.
    if training_args.do_train:

        train_results = trainer.train(
            resume_from_checkpoint=training_args.resume_from_checkpoint
        )
        metrics = train_results.metrics
        trainer.save_model()
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Export.
    if training_args.do_export:
        input_spec = [
            InputSpec(shape=[None, None], dtype="int64", name="input_ids"),
            InputSpec(shape=[None, None], dtype="int64", name="token_type_ids"),
            InputSpec(shape=[None, None], dtype="int64", name="position_ids"),
            InputSpec(
                shape=[None, None, None, None], dtype="float32", name="attention_mask"
            ),
            InputSpec(shape=[None, None], dtype="int64", name="omask_positions"),
            InputSpec(shape=[None], dtype="int64", name="cls_positions"),
        ]
        export_model(
            trainer.pretrained_model,
            input_spec,
            model_args.export_model_dir,
            model_args.export_type,
        )


def tune_with_callback():
    """使用callback"""

    search_space = {
        "lr": (1e-5, 2e-5),
    }

    search_algorithm = BayesOptSearch(space=search_space, metric="loss", mode="min")

    tuner = tune.run(
        train_function,
        resources_per_trial={"cpu": 5, "gpu": 1},
        callbacks=[WandbLoggerCallback(project="Wandb_example_two")],
        search_alg=search_algorithm,
        scheduler=ASHAScheduler(
            metric="loss", mode="min", max_t=4380, grace_period=1, reduction_factor=2
        ),
    )

    print(tuner)


if __name__ == "__main__":

    tune_with_callback()
