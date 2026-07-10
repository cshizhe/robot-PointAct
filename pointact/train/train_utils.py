import torch
import transformers


def add_handler_to_logger(logger):
    """
    logger: accelerator MultiProcessAdapter

    Since the original logger does not have a handler, it will use a default handler
    with logging.level=WARNING. So we add a handler and setlevel to INFO
    """
    import logging

    # Add a console handler
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)

    # Optional: Add formatting
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)

    logger.logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def set_requires_grad(parameters, requires_grad):
    """Set the requires_grad attribute for the parameters."""
    for p in parameters:
        p.requires_grad = requires_grad
    

def configure_vlm(vlm, training_args, compute_dtype, device):
    # Configure the vision tower
    vision_tower = vlm.model.visual
    vision_tower.to(dtype=compute_dtype, device=device)

    vision_model_params = vlm.model.visual.parameters()
    set_requires_grad(vision_model_params, not training_args.freeze_vision_tower)

    merger_params = vlm.model.visual.merger.parameters()
    set_requires_grad(merger_params, not training_args.freeze_merger)

    # Configure the LLM
    lm_head = vlm.lm_head.parameters()
    set_requires_grad(lm_head, not training_args.freeze_lm_head)

    llm_params = vlm.model.language_model.parameters()
    set_requires_grad(llm_params, not training_args.freeze_llm)


def configure_processor(processor, dataset, training_args, logger=None):
    """Configure the processor."""
    if training_args.chat_template:
        import json

        with open(training_args.chat_template) as f:
            chat_template = json.load(f)["chat_template"]
        processor.chat_template = processor.tokenizer.chat_template = chat_template
        if logger is not None:
            logger.info("Set chat template", main_process_only=True)

    if dataset.lerobot_dataset:
        robo_config = dataset.lerobot_dataset.configuration
        robo_config["max_action_dim"] = training_args.max_action_dim
        robo_config["max_state_dim"] = training_args.max_state_dim
        robo_config["action_chunk_size"] = training_args.chunk_size
        processor.robot_config = dict(robo_config)

        if logger is not None:
            logger.info("Set qwen2.5 VL min-max pixels", main_process_only=True)
        processor.image_processor.min_pixels = training_args.image_min_pixels
        processor.image_processor.max_pixels = training_args.image_max_pixels


def smart_tokenizer_and_embedding_resize(
    processor: transformers.ProcessorMixin,
    vla: transformers.PreTrainedModel = None,
):
    """Smart tokenizer and embedding resize."""
    from pointact.constants import (
        ACTION_END_TOKEN,
        ACTION_START_TOKEN,
        DEFAULT_ACTION_TOKEN,
        DEFAULT_IMAGE_TOKEN,
        DEFAULT_STATE_TOKEN,
        DEFAULT_VIDEO_TOKEN,
        PASS_ACTION_TOKEN,
        STATE_END_TOKEN,
        STATE_START_TOKEN,
        TASK_VLA_TOKEN,
        VISION_START_TOKEN,
        THINK_START_TOKEN,
        THINK_END_TOKEN,
        POINT_START_TOKEN,
        DEFAULT_POINT_TOKEN,
        POINT_END_TOKEN,
    )

    tokenizer = processor.tokenizer
    vla_special_tokens = [
        ACTION_START_TOKEN,
        DEFAULT_ACTION_TOKEN,
        PASS_ACTION_TOKEN,
        ACTION_END_TOKEN,
        STATE_START_TOKEN,
        DEFAULT_STATE_TOKEN,
        STATE_END_TOKEN,
        TASK_VLA_TOKEN,
        POINT_START_TOKEN,
        DEFAULT_POINT_TOKEN,
        POINT_END_TOKEN,
    ]
    tokenizer_vocab = tokenizer.get_vocab()
    if THINK_START_TOKEN not in tokenizer_vocab:
        vla_special_tokens.append(THINK_START_TOKEN)
    if THINK_END_TOKEN not in tokenizer_vocab:
        vla_special_tokens.append(THINK_END_TOKEN)
    num_new_tokens = tokenizer.add_tokens(vla_special_tokens, special_tokens=True)

    # NOTE: qwen2.5 vl vocab 151936 > tokenizer 151664 + 8, we don't need to resize embeddings
    token_dict = {
        "state_token_id": DEFAULT_STATE_TOKEN,
        "action_token_start_id": ACTION_START_TOKEN,
        "action_token_id": DEFAULT_ACTION_TOKEN,
        "action_pass_id": PASS_ACTION_TOKEN,
        "vision_token_start_id": VISION_START_TOKEN,
        "image_token_id": DEFAULT_IMAGE_TOKEN,
        "video_token_id": DEFAULT_VIDEO_TOKEN,
        "point_token_id": DEFAULT_POINT_TOKEN,
    }

    for key, token in token_dict.items():
        token_id = tokenizer.convert_tokens_to_ids(token)
        if vla is not None:
            vlm = vla.vlm_backbone
            setattr(vlm.model.config, key, token_id)
            setattr(vlm.config.text_config, key, token_id)
            setattr(vla.config, key, token_id)
        setattr(processor, key, token_id)

    return num_new_tokens


def find_target_linear_names(model, num_lora_modules=-1, lora_namespan_exclude=None, verbose=True):
    """Find the target linear names for LoRA."""
    if lora_namespan_exclude is None:
        lora_namespan_exclude = []
    linear_cls = torch.nn.modules.Linear
    embedding_cls = torch.nn.modules.Embedding
    lora_module_names = []

    for name, module in model.named_modules():
        if any(ex_keyword in name for ex_keyword in lora_namespan_exclude):
            continue
        if isinstance(module, (linear_cls, embedding_cls)):
            lora_module_names.append(name)

    if num_lora_modules > 0:
        lora_module_names = lora_module_names[-num_lora_modules:]
    if verbose:
        print(f"Found {len(lora_module_names)} lora modules: {lora_module_names}")
    return lora_module_names


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa
