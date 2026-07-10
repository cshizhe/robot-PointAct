from pathlib import Path

from accelerate.logging import get_logger
from safetensors.torch import load_file

from pointact.data.dataset_pi import create_pi_data_module
from pointact.model.pi0 import (
    PI0AdvantageModel,
    PI0AdvantageProcessor,
    PI0Config,
    PI0Model,
    PI0Processor,
)
from pointact.model.pi05 import PI05Config, PI05Model, PI05Processor
from pointact.train.pipeline_config import TrainPipelineConfig
from pointact.train.script_utils import (
    has_resume_checkpoint,
    log_trainable_parameters,
    parse_training_args,
    train_or_resume,
)
from pointact.train.train_utils import (
    safe_save_model_for_hf_trainer,
    add_handler_to_logger,
)
from pointact.train.trainer import VLATrainer


logger = get_logger(__name__, log_level="INFO")
logger = add_handler_to_logger(logger)


def _pi_dtype(training_args: TrainPipelineConfig) -> str:
    if training_args.bf16:
        return "bfloat16"
    return "float32"


def build_pi0_config(training_args: TrainPipelineConfig) -> PI0Config:
    return PI0Config(
        action_chunk_size=training_args.chunk_size,
        max_action_dim=training_args.max_action_dim,
        max_state_dim=training_args.max_state_dim,
        dtype=_pi_dtype(training_args),
        gradient_checkpointing=training_args.gradient_checkpointing,
        freeze_vision_encoder=training_args.freeze_vision_tower,
        train_expert_only=training_args.pi_train_expert_only,
        advantage_condition_dropout=training_args.advantage_condition_dropout,
        advantage_cfg_scale=training_args.advantage_cfg_scale,
    )


def build_pi05_config(training_args: TrainPipelineConfig) -> PI05Config:
    return PI05Config(
        action_chunk_size=training_args.chunk_size,
        max_action_dim=training_args.max_action_dim,
        max_state_dim=training_args.max_state_dim,
        dtype=_pi_dtype(training_args),
        gradient_checkpointing=training_args.gradient_checkpointing,
        freeze_vision_encoder=training_args.freeze_vision_tower,
        train_expert_only=training_args.pi_train_expert_only,
    )


def resolve_pi_model(training_args: TrainPipelineConfig):
    model_class = (training_args.model_class or "pi05").split(".")[-1].lower()
    if model_class in {"pi0", "pi0model"}:
        return "PI0", build_pi0_config(training_args), PI0Model
    if model_class in {"pi0_advantage", "pi0advantage", "pi0advantagemodel"}:
        return "PI0Advantage", build_pi0_config(training_args), PI0AdvantageModel
    if model_class in {"pi05", "pi05model"}:
        return "PI05", build_pi05_config(training_args), PI05Model
    raise ValueError(
        "Unknown PI model_class {!r}. Expected 'pi0', 'pi0_advantage' or 'pi05'.".format(
            training_args.model_class
        )
    )


def build_pi_processor(model_name: str, training_args: TrainPipelineConfig):
    if model_name == "PI0":
        # pi0 uses mean-std normalization
        return PI0Processor(
            robot_state_action_norm_file=training_args.pi_state_action_norm_file,
        )
    if model_name == "PI0Advantage":
        return PI0AdvantageProcessor(
            robot_state_action_norm_file=training_args.pi_state_action_norm_file,
            advantage_condition_dropout=training_args.advantage_condition_dropout,
            advantage_cfg_scale=training_args.advantage_cfg_scale,
        )
    if model_name == "PI05":
        # pi0.5 uses q1-q99 min-max normalization
        return PI05Processor(
            robot_state_action_norm_file=training_args.pi_state_action_norm_file,
        )
    raise ValueError(f"Unknown PI model: {model_name}")


def load_original_safetensors(
    model,
    checkpoint_path: str | Path,
    strict: bool = False,
    model_name: str = "",
):
    def _fix_pytorch_state_dict_keys(state_dict):  # see openpi `BaseModelConfig, _fix_pytorch_state_dict_keys`
        """Fix state dict keys to match current model architecture."""
        import re

        fixed_state_dict = {}

        for key, value in state_dict.items():
            new_key = key

            # Handle layer norm structure changes: .weight -> .dense.weight + .dense.bias
            # For gemma expert layers
            if re.match(
                r"paligemma_with_expert\.gemma_expert\.model\.layers\.\d+\.(input_layernorm|post_attention_layernorm)\.weight",
                key,
            ):
                # Check if the model actually has adaRMS enabled for the expert
                expert_uses_adarms = getattr(
                    model.paligemma_with_expert.gemma_expert.config, "use_adarms", False
                )
                if expert_uses_adarms:
                    logger.warning(f"Skipping layer norm key (adaRMS mismatch): {key}")
                    continue

            if re.match(r"paligemma_with_expert\.gemma_expert\.model\.norm\.weight", key):
                # Check if the model actually has adaRMS enabled for the expert
                expert_uses_adarms = getattr(
                    model.paligemma_with_expert.gemma_expert.config, "use_adarms", False
                )
                if expert_uses_adarms:
                    logger.warning(f"Skipping norm key (adaRMS mismatch): {key}")
                    continue

            if model_name in {"PI0", "PI0Advantage"}:
                # Handle MLP naming changes for pi0
                # non-pi05 model expects action_time_mlp_*, but checkpoint might have time_mlp_*
                if key.startswith("time_mlp_in."):
                    new_key = key.replace("time_mlp_in.", "action_time_mlp_in.")
                elif key.startswith("time_mlp_out."):
                    new_key = key.replace("time_mlp_out.", "action_time_mlp_out.")
            elif model_name == "PI05":
                # Handle MLP naming changes for pi05
                # pi05 model expects time_mlp_*, but checkpoint might have action_time_mlp_*
                if key.startswith("action_time_mlp_in."):
                    new_key = key.replace("action_time_mlp_in.", "time_mlp_in.")
                elif key.startswith("action_time_mlp_out."):
                    new_key = key.replace("action_time_mlp_out.", "time_mlp_out.")
                # Also handle state_proj which shouldn't exist in pi05
                if key.startswith("state_proj."):
                    logger.warning(f"Skipping state_proj key in pi05 mode: {key}")
                    continue

            # Handle vision tower embedding layer potential differences
            if "patch_embedding" in key:
                # Some checkpoints might have this, but current model expects different structure
                logger.warning(f"Vision embedding key might need handling: {key}")

            if (
                (key == "model.paligemma_with_expert.paligemma.lm_head.weight"
                or key == "paligemma_with_expert.paligemma.lm_head.weight") and \
                "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight" not in state_dict
            ):
                fixed_state_dict[
                    "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
                ] = value.clone()

            fixed_state_dict[new_key] = value

        return fixed_state_dict
    
    state_dict = load_file(str(checkpoint_path), device="cpu")
    state_dict = _fix_pytorch_state_dict_keys(state_dict)
    
    model_state_dict = model.state_dict()

    checkpoint_keys = set(state_dict)
    model_keys = set(model_state_dict)
    missing_keys = sorted(model_keys - checkpoint_keys)
    unexpected_keys = sorted(checkpoint_keys - model_keys)
    common_keys = sorted(model_keys & checkpoint_keys)
    mismatched_keys = [
        key
        for key in common_keys
        if state_dict[key].shape != model_state_dict[key].shape
    ]
    compatible_keys = [key for key in common_keys if key not in mismatched_keys]
    compatible_state_dict = {key: state_dict[key] for key in compatible_keys}

    def print_key_summary(title: str, keys: list[str], max_items: int = 20):
        print(f"{title}: {len(keys)}")
        for key in keys[:max_items]:
            print(f"  - {key}")
        if len(keys) > max_items:
            print(f"  ... {len(keys) - max_items} more")

    model_name = model.__class__.__name__
    print(f"Loading {model_name} checkpoint: {checkpoint_path}")
    print(f"Checkpoint tensors: {len(checkpoint_keys)}")
    print(f"Model tensors: {len(model_keys)}")
    print_key_summary("Missing keys", missing_keys)
    print_key_summary("Unexpected keys", unexpected_keys)
    print_key_summary("Shape-mismatched keys", mismatched_keys)

    loaded_numel = sum(state_dict[key].numel() for key in compatible_keys)
    model_numel = sum(tensor.numel() for tensor in model_state_dict.values())
    print(
        "Compatible tensors to load: "
        f"{len(compatible_keys)} / {len(model_keys)} model tensors "
        f"({loaded_numel:,} / {model_numel:,} parameters)"
    )

    load_result = model.load_state_dict(compatible_state_dict, strict=strict)
    print_key_summary("load_state_dict missing keys", list(load_result.missing_keys))
    print_key_summary("load_state_dict unexpected keys", list(load_result.unexpected_keys))
    return load_result


def configure_processor(processor, dataset, training_args):
    if dataset.lerobot_dataset:
        robo_config = dataset.lerobot_dataset.configuration
        robo_config["max_action_dim"] = training_args.max_action_dim
        robo_config["max_state_dim"] = training_args.max_state_dim
        robo_config["action_chunk_size"] = training_args.chunk_size
        processor.robot_config = dict(robo_config)

def train():
    training_args = parse_training_args()
    resume_from_checkpoint = has_resume_checkpoint(training_args.output_dir)

    model_name, config, model_cls = resolve_pi_model(training_args)
    model = model_cls(config)
    if not resume_from_checkpoint and training_args.model_name_or_path:
        from huggingface_hub import snapshot_download
        if training_args.model_name_or_path.endswith("model.safetensors"):
            checkpoint_path = training_args.model_name_or_path
        else:
            checkpoint_dir = snapshot_download(repo_id=training_args.model_name_or_path)
            checkpoint_path = str(Path(checkpoint_dir) / "model.safetensors")
        load_original_safetensors(model, checkpoint_path, model_name)

    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}

    processor = build_pi_processor(model_name, training_args)

    data_module = create_pi_data_module(processor=processor, args=training_args)
    configure_processor(processor, data_module["train_dataset"], training_args)

    log_trainable_parameters(model, logger)

    trainer = VLATrainer(model=model, processing_class=processor, args=training_args, **data_module)

    train_or_resume(
        trainer,
        training_args,
        logger,
        resume_from_checkpoint=resume_from_checkpoint,
    )

    trainer.save_state()
    safe_save_model_for_hf_trainer(
        trainer=trainer,
        output_dir=f"{training_args.output_dir}/checkpoint-final-{trainer.state.global_step}",
    )


if __name__ == "__main__":
    train()
