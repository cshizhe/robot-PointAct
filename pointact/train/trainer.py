# Copyright 2025 EO-Robotics Team. All rights reserved.
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

import torch
import torch.nn as nn
from transformers import Trainer
from transformers.trainer import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
    is_sagemaker_mp_enabled,
    logger,
)
from transformers.trainer_pt_utils import nested_gather


class MetaLossesTrainerState(TrainerCallback):
    def __init__(self, meta_losses: list[str]):
        self.meta_losses = meta_losses

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        control.meta_losses = {k: torch.tensor(0.0, device=args.device) for k in self.meta_losses}
        control.meta_losses_last_logged_step = state.global_step
        return control


class VLATrainer(Trainer):
    """Trainer with meta-loss logging and optional module-specific learning rates."""

    meta_losses: tuple[str, ...] = ("text_loss", "action_loss")
    special_lr_parameter_patterns = (
        ("vision_lr", "visual", lambda name: "visual" in name and "merger" not in name),
        ("merger_lr", "merger", lambda name: "visual" in name and "merger" in name),
        ("llm_lr", "language_model", lambda name: "language_model" in name),
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_callback(MetaLossesTrainerState(list(self.meta_losses)))

    def _has_special_lrs(self):
        return any(getattr(self.args, arg_name, None) is not None for arg_name, _, _ in self.special_lr_parameter_patterns)

    def _special_lr_parameter_names(self, opt_model):
        parameter_names = [name for name, _ in opt_model.named_parameters()]
        special_lr_parameter_names = {}

        for arg_name, label, matcher in self.special_lr_parameter_patterns:
            lr = getattr(self.args, arg_name, None)
            if lr is None:
                continue

            matched_names = [name for name in parameter_names if matcher(name)]
            special_lr_parameter_names[arg_name] = matched_names
            log_fn = logger.info if matched_names else logger.warning
            log_fn(f"{arg_name}={lr} matched {len(matched_names)} {label} parameters")

        return special_lr_parameter_names

    def create_optimizer(self):
        if is_sagemaker_mp_enabled() or not self._has_special_lrs():
            return super().create_optimizer()

        opt_model = self.model
        if self.optimizer is None:
            decay_parameters = set(self.get_decay_parameter_names(opt_model))
            special_lr_parameter_names = self._special_lr_parameter_names(opt_model)
            special_lr_parameters = {
                name for names in special_lr_parameter_names.values() for name in names
            }

            optimizer_grouped_parameters = [
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if n in decay_parameters and n not in special_lr_parameters and p.requires_grad
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if n not in decay_parameters and n not in special_lr_parameters and p.requires_grad
                    ],
                    "weight_decay": 0.0,
                },
            ]

            for arg_name, _, _ in self.special_lr_parameter_patterns:
                lr = getattr(self.args, arg_name, None)
                parameter_names = set(special_lr_parameter_names.get(arg_name, []))
                if lr is None or not parameter_names:
                    continue

                optimizer_grouped_parameters.extend(
                    [
                        {
                            "params": [
                                p
                                for n, p in opt_model.named_parameters()
                                if n in decay_parameters and n in parameter_names and p.requires_grad
                            ],
                            "weight_decay": self.args.weight_decay,
                            "lr": lr,
                        },
                        {
                            "params": [
                                p
                                for n, p in opt_model.named_parameters()
                                if n not in decay_parameters and n in parameter_names and p.requires_grad
                            ],
                            "weight_decay": 0.0,
                            "lr": lr,
                        },
                    ]
                )

            if self.optimizer_cls_and_kwargs is not None:
                optimizer_cls, optimizer_kwargs = self.optimizer_cls_and_kwargs
            else:
                optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args, opt_model)

            if "params" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("params")
            if "model" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("model")
            if "optimizer_dict" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("optimizer_dict")

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped / 2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped / 2**20}M params")

        return self.optimizer

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if hasattr(self.control, "meta_losses") and model.training:
            loss, outputs = super().compute_loss(
                model,
                inputs,
                return_outputs=True,
                num_items_in_batch=num_items_in_batch,
            )

            if not isinstance(outputs, dict):
                raise ValueError(
                    "The model output should be a dictionary or ModelOutput and not a tuple or list."
                )

            for k, v in outputs.items():
                if k in self.control.meta_losses and v is not None:
                    self.control.meta_losses[k] += v.detach().mean() / self.args.gradient_accumulation_steps

            return (loss, outputs) if return_outputs else loss

        return super().compute_loss(
            model,
            inputs,
            return_outputs=return_outputs,
            num_items_in_batch=num_items_in_batch,
        )

    def log(self, logs: dict[str, float], *args, **kwargs):
        if hasattr(self.control, "meta_losses") and "loss" in logs:
            step_delta = self.state.global_step - self.control.meta_losses_last_logged_step
            if step_delta > 0:
                for k, v in self.control.meta_losses.items():
                    gathered_loss = nested_gather(v, self.args.parallel_mode).mean().item()
                    logs[k] = round(gathered_loss / step_delta, 4)
                    v.zero_()
                self.control.meta_losses_last_logged_step = self.state.global_step

        return super().log(logs, *args, **kwargs)
