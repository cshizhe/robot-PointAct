import os
import dataclasses
import json
import numpy as np
import tyro
from PIL import Image
import time

from pointact.model.eo1.modeling_eo1 import EO1VisionFlowMatchingModel
from pointact.model.eo1.processing_eo1 import EO1VisionProcessor, EO1VisionPointProcessor
from pointact.model.eo1.modeling_eo1_point import EO1VisionPointFlowMatchingModel
from pointact.model.vla_dual.modeling_vla_dual import (
    VLADualFlowMatchingModel,
)
from pointact.model.vla_dual.modeling_vla_dual_point import VLADualPointFlowMatchingModel
from pointact.model.vla_dual.processing_vla_dual import VLADualProcessor
from pointact.model.vla_pointact.modeling_vla_pointact import (
    VLAEncDec3DClassificationModel,
    VLAEncDec3DWithActionClassificationModel,
    VLAEncDec3DRegressionModel,
    VLAEncDec3DWithActionRegressionModel
)
from pointact.model.vla_pointact.processing_vla_pointact import (
    VLAEncDec3DProcessor,
)
from pointact.model.pi0 import PI0AdvantageModel, PI0AdvantageProcessor, PI0Model, PI0Processor
from pointact.model.pi05 import PI05Model, PI05Processor

from pointact.utils.torch_utils import set_seed

from pointact.utils.server_client import PolicyServer

MODEL_MAP = {
    "EO1VisionFlowMatchingModel": (EO1VisionFlowMatchingModel, EO1VisionProcessor),
    "EO1VisionPointFlowMatchingModel": (EO1VisionPointFlowMatchingModel, EO1VisionPointProcessor),
    "VLADualFlowMatchingModel": (VLADualFlowMatchingModel, VLADualProcessor),
    "VLAEncDec3DClassificationModel": (VLAEncDec3DClassificationModel, VLAEncDec3DProcessor),
    "VLAEncDec3DWithActionClassificationModel": (VLAEncDec3DWithActionClassificationModel, VLAEncDec3DProcessor),
    "VLAEncDec3DRegressionModel": (VLAEncDec3DRegressionModel, VLAEncDec3DProcessor),
    "VLAEncDec3DWithActionRegressionModel": (VLAEncDec3DWithActionRegressionModel, VLAEncDec3DProcessor),
    "VLADualPointFlowMatchingModel": (VLADualPointFlowMatchingModel, VLAEncDec3DProcessor),
    "PI0AdvantageModel": (PI0AdvantageModel, PI0AdvantageProcessor),
    "PI0Model": (PI0Model, PI0Processor),
    "PI05Model": (PI05Model, PI05Processor),
}


@dataclasses.dataclass
class ServerArgs:
    seed: int = 7  # Random Seed (for reproducibility)
    
    pretrained_path: str = ""

    host: str = "127.0.0.1"
    port: int = 5555

    # diffusion
    num_denoise_steps: int = 10

    save_data: bool = False
    save_dir: str = "/scratch/shichen/datasets/VLA3D_exprs/server_test_data"


class Policy:
    def __init__(self, args):
        # model = (
        #     AutoModel.from_pretrained(
        #         args.pretrained_path,
        #         trust_remote_code=True,
        #         local_files_only=True,
        #         # attn_implementation="flash_attention_2",
        #     )
        #     .eval()
        #     .cuda()
        # )

        # processor = AutoProcessor.from_pretrained(
        #     args.pretrained_path, trust_remote_code=True, local_files_only=True
        # )
        
        model_config = json.load(
            open(os.path.join(args.pretrained_path, 'config.json'), 'r')
        )
        model_name = model_config['architectures'][0]
        model_class, processor_class = MODEL_MAP[model_name]

        print(model_class, processor_class, args.pretrained_path)
        self.model = model_class.from_pretrained(
            args.pretrained_path,
            device_map={"": "cuda"}
        ).eval() #.cuda()
        print(self.model)
        self.processor = processor_class.from_pretrained(args.pretrained_path)

        self.model.config.num_denoise_steps = args.num_denoise_steps

        self.save_data = args.save_data
        self.num_steps = 0
        if args.save_data:
            self.save_dir = args.save_dir
            os.makedirs(self.save_dir, exist_ok=True)
            print(f"Save data in {self.save_dir}")
    
    def get_action(self, batch, options):
        pred_rot_type = options.get("pred_rot_type", None)
        points_workspace = options.get("points_workspace", None)
        remove_arm = options.get("remove_arm", False)
        advantage_label = options.get("advantage_label", options.get("advantage", 1))
        advantage_cfg_scale = options.get("advantage_cfg_scale", None)

        for k, v in batch.items():
            if k.startswith("observation.images."):
                batch[k] = [Image.fromarray(img) for img in v]

        if self.save_data:
            obs_outfile = os.path.join(self.save_dir, f"{self.num_steps}_obs.npy")
            np.save(obs_outfile, batch)

        # start_time = time.time()
        batch_actions = self.processor.select_action(
            self.model,
            batch,
            pred_rot_type=pred_rot_type,
            points_workspace=points_workspace,
            remove_arm=remove_arm,
            advantage_label=advantage_label,
            advantage_cfg_scale=advantage_cfg_scale,
        )
        # print('cost time', time.time() - start_time)

        if self.save_data:
            action_outfile = os.path.join(self.save_dir, f"{self.num_steps}_actions.npy")
            np.save(action_outfile, batch_actions['action'])
            print(batch_actions)
            
        self.num_steps += 1
        return batch_actions
    
    def reset(self, options=None):
        self.num_steps = 0
        pass

def main(args: ServerArgs) -> None:
    # Set random seed    
    set_seed(args.seed)

    policy = Policy(args)
    server = PolicyServer.start_server(policy, args.host, args.port)

if __name__ == "__main__":
    tyro.cli(main)
