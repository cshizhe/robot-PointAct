from transformers.configuration_utils import PretrainedConfig
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import (
    Qwen2_5_VLTextConfig,
    Qwen2_5_VLVisionConfig,
)

from pointact.model.action_head.flow_matching_action_head import FlowmatchingActionHeadConfig


class VLADualFlowMatchingConfig(PretrainedConfig):
    model_type = "vla_dual"
    sub_configs = {"vision_config": Qwen2_5_VLVisionConfig, "text_config": Qwen2_5_VLTextConfig}
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        image_token_id=151655,
        video_token_id=151656,
        action_chunk_size=50,
        max_action_dim=32,
        max_state_dim=64,
        num_denoise_steps=10,
        **kwargs,
    ):
        if isinstance(vision_config, dict):
            self.vision_config = self.sub_configs["vision_config"](**vision_config)
        elif vision_config is None:
            self.vision_config = self.sub_configs["vision_config"](
                hidden_size=1280,
                out_hidden_size=2048,
                tokens_per_second=2,
            )

        if isinstance(text_config, dict):
            self.text_config = self.sub_configs["text_config"](**text_config)
        elif text_config is None:
            self.text_config = self.sub_configs["text_config"](**kwargs)

        self.image_token_id = image_token_id
        self.video_token_id = video_token_id

        self.action_chunk_size = action_chunk_size
        self.max_action_dim = max_action_dim
        self.max_state_dim = max_state_dim
        self.num_denoise_steps = num_denoise_steps

        super().__init__(**kwargs)

    @property
    def action_head_config(self):
        vlm_hidden_size = self.text_config.hidden_size

        config = FlowmatchingActionHeadConfig(
            add_pos_embed=True,
            model_dtype="float32",
            diffusion_model_cfg={
                # hidden_size=32x48=1536
                "num_attention_heads": 32,
                "attention_head_dim": 48,
                "cross_attention_dim": vlm_hidden_size,
                "num_layers": 16,
                "output_dim": 1024,
                "dropout": 0.2,
                "final_dropout": True,
                "interleave_self_attention": True,
                "norm_type": "ada_norm",
                "positional_embeddings": None,
            },
            num_target_vision_tokens=32, # similar to register tokens here
            input_embedding_dim=1536,
            backbone_embedding_dim=vlm_hidden_size,
            hidden_size=1024,
            max_state_dim=self.max_state_dim,
            action_dim=self.max_action_dim,
            action_horizon=self.action_chunk_size,
            noise_beta_alpha=1.5,
            noise_beta_beta=1.0,
            noise_s=0.999,
            num_timestep_buckets=1000,
            num_inference_timesteps=self.num_denoise_steps,
            max_num_embodiments=32, # 1
            tune_projector=True,
            tune_diffusion_model=True,
            load_pretrained_det_decode_layer_path=None,
            detection_coeff=1.0,
            freeze_decode_layer=False,
            expand_batch=None,
            use_vlln=True,
            vl_self_attention_cfg={
                # hidden_size: 2048
                "num_attention_heads": 32,
                "attention_head_dim": 64,
                "dropout": 0.2,
                "final_dropout": True,
                "num_layers": 4,
                "positional_embeddings": None
            },
        )

        return config


class VLADualPointFlowMatchingConfig(VLADualFlowMatchingConfig):
    model_type = "vla_dual_point"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.ptv3_patch_size = kwargs.get("ptv3_patch_size", 128)
        self.ptv3_enc_mode = kwargs.get("ptv3_enc_mode", True)
        self.ptv3_enc_channels = kwargs.get("ptv3_enc_channels", [64, 128, 256, 384, 512])
        self.ptv3_enc_depths = kwargs.get("ptv3_enc_depths", [2, 2, 2, 6, 2])
        self.ptv3_enc_num_head = kwargs.get("ptv3_enc_num_head", [2, 4, 8, 16, 32])
        self.ptv3_dec_num_head = kwargs.get("ptv3_dec_num_head", [4, 4, 8, 16])
        self.voxel_size = 0.01
        self.point_token_id = kwargs.get("point_token_id", 151674)


# VLADualFlowMatchingConfig.register_for_auto_class()
# VLADualPointFlowMatchingConfig.register_for_auto_class()
