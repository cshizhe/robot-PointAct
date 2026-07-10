import os
import sys
from pathlib import Path

from accelerate.utils import broadcast_object_list
from transformers import HfArgumentParser

from pointact.train.pipeline_config import TrainPipelineConfig


def parse_training_args(logger=None) -> TrainPipelineConfig:
    parser = HfArgumentParser(TrainPipelineConfig)
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        (training_args,) = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        (training_args,) = parser.parse_args_into_dataclasses()

    training_args.output_dir = broadcast_object_list([training_args.output_dir])[0]
    if logger is not None:
        logger.info(f"set output-dir to {training_args.output_dir}")
    return training_args


def log_trainable_parameters(model, logger) -> None:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_pct = trainable_params / total_params * 100 if total_params else 0
    logger.warning(
        f"{total_params=}, {trainable_params=}, [{trainable_pct}%]",
        main_process_only=True,
    )


def has_resume_checkpoint(output_dir: str | Path) -> bool:
    return bool(list(Path(output_dir).glob("checkpoint-*")))


def train_or_resume(
    trainer,
    training_args: TrainPipelineConfig,
    logger,
    resume_from_checkpoint: bool | None = None,
) -> None:
    output_dir = Path(training_args.output_dir)

    # save training args to output dir for future reference
    if trainer.accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "training_args.json", "w") as outf:
            outf.write(training_args.to_json_string())

    if resume_from_checkpoint is None:
        resume_from_checkpoint = has_resume_checkpoint(output_dir)

    if resume_from_checkpoint:
        logger.info("resume from checkpoint")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
