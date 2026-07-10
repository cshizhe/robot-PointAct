from transformers.configuration_utils import PretrainedConfig
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import (
    Qwen2_5_VLTextConfig,
    Qwen2_5_VLVisionConfig,
)

class VLAEncDec3DModelConfig(PretrainedConfig):
    model_type = "VLAEncDec3DModel"
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
        max_num_embodiments=1,
        use_robot_state=True,
        ctx_embed_size=512,
        num_denoise_steps=10,
        num_timestep_buckets=1000,
        noise_beta_alpha=1.5,
        noise_beta_beta=1.0,
        noise_s=0.999,
        flow_matching_target="x-pred",  # v-pred, x-pred
        flow_matching_loss="x-loss",    # v-loss, x-loss 
        time_embed_size=512,
        ptv3_patch_size=1024,
        ptv3_enc_mode=True,
        ptv3_enc_channels=(32, 64, 128, 256, 512),
        ptv3_enc_depths=(2, 2, 2, 6, 2),
        ptv3_enc_num_head=(2, 4, 8, 16, 32),
        ptv3_dec_channels = (64, 64, 128, 256),
        ptv3_dec_depths=(2, 2, 2, 2),
        ptv3_dec_num_head=(4, 4, 8, 16),
        ptv3_clf_head_pos_bins=100,
        ptv3_apply_point_ca=False,
        ptv3_input_channels=6,
        ptv3_backend="concerto",
        action_regression_loss="l2",
        action_head_pos_center="moe", 
        regression_head_heatmap_temp=0.1,
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
        self.max_num_embodiments = max_num_embodiments
        self.use_robot_state = use_robot_state

        self.ctx_embed_size = ctx_embed_size
        self.ptv3_patch_size = ptv3_patch_size
        self.ptv3_enc_mode = ptv3_enc_mode
        self.ptv3_enc_channels = ptv3_enc_channels
        self.ptv3_enc_depths = ptv3_enc_depths
        self.ptv3_enc_num_head = ptv3_enc_num_head
        self.ptv3_dec_channels = ptv3_dec_channels
        self.ptv3_dec_depths = ptv3_dec_depths
        self.ptv3_dec_num_head = ptv3_dec_num_head
        self.ptv3_input_channels = ptv3_input_channels
        self.ptv3_apply_point_ca = ptv3_apply_point_ca
        self.ptv3_backend = ptv3_backend

        # classification head
        self.ptv3_clf_head_pos_bins = ptv3_clf_head_pos_bins
        # choices: moe (per point prediction), zero (directly predict action from the global feature)
        self.action_head_pos_center = action_head_pos_center 

        # regression head
        self.regression_head_heatmap_temp = regression_head_heatmap_temp

        self.action_regression_loss = action_regression_loss

        # diffusion
        self.num_denoise_steps = num_denoise_steps
        self.num_timestep_buckets = num_timestep_buckets
        self.noise_beta_alpha = noise_beta_alpha
        self.noise_beta_beta = noise_beta_beta
        self.noise_s = noise_s
        self.flow_matching_target = flow_matching_target
        self.flow_matching_loss = flow_matching_loss
        self.time_embed_size = time_embed_size

        super().__init__(**kwargs)

# VLAEncDec3DModelConfig.register_for_auto_class()
