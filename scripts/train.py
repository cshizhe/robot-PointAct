import importlib

import torch
from accelerate.logging import get_logger

from pointact.model.backbone.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from pointact.train.pipeline_config import TrainPipelineConfig
from pointact.train.script_utils import (
    has_resume_checkpoint,
    log_trainable_parameters,
    parse_training_args,
    train_or_resume,
)
from pointact.train.train_utils import (
    add_handler_to_logger,
    configure_vlm,
    configure_processor,
    find_target_linear_names,
    safe_save_model_for_hf_trainer,
    smart_tokenizer_and_embedding_resize,
)
from train_registry import TrainRecipe, resolve_recipe

logger = get_logger(__name__, log_level="INFO")
logger = add_handler_to_logger(logger)


def _import_object(dotted_path: str):
    module_name, object_name = dotted_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, object_name)


def _compute_dtype(training_args: TrainPipelineConfig) -> torch.dtype:
    if training_args.bf16:
        return torch.bfloat16
    if training_args.fp16:
        return torch.float16
    return torch.float32


def build_fresh_model(recipe: TrainRecipe, training_args: TrainPipelineConfig, compute_dtype: torch.dtype):
    config_class = _import_object(recipe.config_class)
    model_class = _import_object(recipe.model_class)

    config = config_class.from_pretrained(
        training_args.vlm_name_or_path,
        dtype=compute_dtype,
        attn_implementation=training_args.attn_implementation,
        **recipe.config_kwargs_fn(training_args),
    )

    vlm_backbone = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        training_args.vlm_name_or_path,
        dtype=compute_dtype,
        attn_implementation=training_args.attn_implementation,
    )
    return model_class(config, vlm_backbone=vlm_backbone)


def build_model(recipe: TrainRecipe, training_args: TrainPipelineConfig, compute_dtype: torch.dtype):
    if training_args.model_name_or_path is None:
        return build_fresh_model(recipe, training_args, compute_dtype)

    model_class = _import_object(recipe.model_class)
    model = model_class.from_pretrained(
        training_args.model_name_or_path,
        dtype=compute_dtype,
        attn_implementation=training_args.attn_implementation,
    )
    # Rewrite the action chunk size
    model.config.action_chunk_size = training_args.chunk_size
    return model


def load_processor(recipe: TrainRecipe, training_args: TrainPipelineConfig):
    processor_class = _import_object(recipe.processor_class)
    return processor_class.from_pretrained(
        training_args.processor_name_or_path,
        padding_side="right",
    )



def maybe_load_ptv3_checkpoint(model, training_args: TrainPipelineConfig) -> None:
    if training_args.ptv3_init_ckpt_file is None:
        return
    
    ptv3_module = getattr(model, "ptv3_model", None)
    if ptv3_module is not None:
        ptv3_module = model.ptv3_model
        ptv3_module = getattr(ptv3_module, "ptv3_model", ptv3_module)

    if ptv3_module is None:
        logger.warning(
            f"ignoring ptv3_init_ckpt_file for model without ptv3_model: {training_args.ptv3_init_ckpt_file}",
            main_process_only=True,
        )
        return
    
    def _maybe_slice_input_stem(state_dict: dict[str, torch.Tensor], target_state: dict[str, torch.Tensor]) -> None:
        key = "embedding.stem.linear.weight"
        if key not in state_dict or key not in target_state:
            return

        source = state_dict[key]
        target = target_state[key]
        if source.shape == target.shape:
            return
        if source.ndim == target.ndim == 2 and source.shape[0] == target.shape[0] and source.shape[1] >= target.shape[1]:
            state_dict[key] = source[:, : target.shape[1]]

    checkpoint = torch.load(training_args.ptv3_init_ckpt_file, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    target_state = ptv3_module.state_dict()
    _maybe_slice_input_stem(state_dict, target_state)

    compatible_state = {}
    skipped_shape = []
    for name, value in state_dict.items():
        if name not in target_state:
            continue
        if value.shape == target_state[name].shape:
            compatible_state[name] = value
        else:
            skipped_shape.append((name, tuple(value.shape), tuple(target_state[name].shape)))

    ptv3_module.load_state_dict(compatible_state, strict=False)
    logger.info(
        f"resumed {len(compatible_state)} ptv3 parameters from {training_args.ptv3_init_ckpt_file}; "
        f"skipped {len(skipped_shape)} shape mismatches",
        main_process_only=True,
    )


def apply_lora(model, training_args: TrainPipelineConfig, compute_dtype: torch.dtype):
    if not training_args.lora_enable:
        return model

    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        raise ImportError("LoRA training requires the `peft` package.") from exc

    peft_config = LoraConfig(
        r=training_args.lora_rank,
        lora_alpha=training_args.lora_alpha,
        target_modules=find_target_linear_names(
            model,
            lora_namespan_exclude=training_args.lora_namespan_exclude,
            num_lora_modules=training_args.num_lora_modules,
        ),
        lora_dropout=training_args.lora_dropout,
        bias=training_args.lora_bias,
    )
    if compute_dtype != torch.float32:
        model.to(compute_dtype)
    logger.info("adding LoRA to the model...", main_process_only=True)
    return get_peft_model(model, peft_config)


def preview_sample(trainer, processor, data_module) -> None:
    if not trainer.accelerator.is_main_process:
        return

    dataset = data_module["train_dataset"]
    dataset.info_qwen_vision_fetch()
    input_ids = dataset[0]["input_ids"]
    print(f"sample: {processor.tokenizer.decode(input_ids)}")


def train():
    training_args = parse_training_args(logger=logger)
    recipe = resolve_recipe(training_args.model_class)
    resume_from_checkpoint = has_resume_checkpoint(training_args.output_dir)

    compute_dtype = _compute_dtype(training_args)
    model = build_model(recipe, training_args, compute_dtype)
    if not resume_from_checkpoint:
        maybe_load_ptv3_checkpoint(model, training_args)

    processor = load_processor(recipe, training_args)
    # Add new tokens to processor and model
    smart_tokenizer_and_embedding_resize(processor, model)
    # Freeze the vlm or not
    configure_vlm(model.vlm_backbone, training_args, compute_dtype, training_args.device)
    
    model = apply_lora(model, training_args, compute_dtype)

    create_data_module = _import_object(recipe.data_module_fn)
    data_module = create_data_module(processor=processor, args=training_args)
    # Configure the processor with robot dataset stats, chat template, etc.
    configure_processor(processor, data_module["train_dataset"], training_args, logger=logger)

    model.config.use_cache = False
    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}

    log_trainable_parameters(model, logger)

    trainer_class = _import_object(recipe.trainer_class)
    trainer = trainer_class(model=model, processing_class=processor, args=training_args, **data_module)
    preview_sample(trainer, processor, data_module)

    train_or_resume(
        trainer,
        training_args,
        logger,
        resume_from_checkpoint=resume_from_checkpoint,
    )

    model.config.use_cache = True
    trainer.save_state()
    safe_save_model_for_hf_trainer(
        trainer=trainer,
        output_dir=f"{training_args.output_dir}/checkpoint-final-{trainer.state.global_step}",
    )


if __name__ == "__main__":
    train()
