import copy
import numpy as np
from dataclasses import dataclass

from transformers import logging
import torch

from transformers.processing_utils import (
    ProcessingKwargs,
    ProcessorMixin,
    TypedDict,
    validate_typed_dict,
)

from pointact.constants import (
    ACTION_END_TOKEN,
    ACTION_START_TOKEN,
    DEFAULT_ACTION_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_POINT_TOKEN,
    DEFAULT_STATE_TOKEN,
    DEFAULT_VIDEO_TOKEN,
    PASS_ACTION_TOKEN,
    TASK_VLA_TOKEN,
)


logger = logging.get_logger(__name__)


@dataclass
class PreparedRobotBatch:
    messages: list
    states: list[torch.Tensor] | None
    repo_ids: list[str]
    points: list[torch.Tensor] | None = None

 
class RobotProcessorBase(ProcessorMixin):

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        video_processor=None,
        chat_template=None,
        robot_config=None,
        **kwargs,
    ):
        self.image_token = getattr(tokenizer, "image_token", DEFAULT_IMAGE_TOKEN)
        self.video_token = getattr(tokenizer, "video_token", DEFAULT_VIDEO_TOKEN)
        self.action_token = getattr(tokenizer, "action_token", DEFAULT_ACTION_TOKEN)
        self.state_token = getattr(tokenizer, "state_token", DEFAULT_STATE_TOKEN)

        for token_name in ["image_token", "video_token", "action_token", "state_token"]:
            setattr(self, f"{token_name}_id", getattr(tokenizer, token_name + "_id", None) or tokenizer.convert_tokens_to_ids(getattr(self, token_name)))

        self.robot_config = robot_config or {}

        super().__init__(image_processor, tokenizer, video_processor, chat_template=chat_template)
    
    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        names_from_processor = list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))
        return names_from_processor + ["second_per_grid_ts"] + ["states", "actions"]
    
    def _prepare_image_video_action_inputs(self, images, videos, text, output_kwargs):

        image_inputs = videos_inputs = {}

        if images is not None:
            image_inputs = self.image_processor(images=images, **output_kwargs["images_kwargs"])
            image_grid_thw = image_inputs["image_grid_thw"]

        if videos is not None:
            videos_inputs = self.video_processor(videos=videos, **output_kwargs["videos_kwargs"])
            video_grid_thw = videos_inputs["video_grid_thw"]

            # Get video metadata
            if not output_kwargs.get("return_metadata"):
                video_metadata = videos_inputs.pop("video_metadata")
            else:
                video_metadata = videos_inputs["video_metadata"]

            fps = [metadata.sampled_fps for metadata in video_metadata]

            if isinstance(fps, (int, float)):
                second_per_grid_ts = [self.video_processor.temporal_patch_size / fps] * len(video_grid_thw)
            elif hasattr(fps, "__len__") and len(fps) == len(video_grid_thw):
                second_per_grid_ts = [self.video_processor.temporal_patch_size / tmp for tmp in fps]
            else:
                raise ValueError(
                    f"The length of fps ({len(fps) if hasattr(fps, '__len__') else fps}) must be equal to the length of video_grid_thw ({len(video_grid_thw)}) or fps should be a single number."
                )
            videos_inputs.update({"second_per_grid_ts": second_per_grid_ts})

        if not isinstance(text, list):
            text = [text]

        text = text.copy()  # below lines change text in-place
        if images is not None:
            merge_length = self.image_processor.merge_size**2
            index = 0
            for i in range(len(text)):
                while self.image_token in text[i]:
                    num_image_tokens = image_grid_thw[index].prod() // merge_length
                    text[i] = text[i].replace(self.image_token, "<|placeholder|>" * num_image_tokens, 1)
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", self.image_token)

        if videos is not None:
            merge_length = self.video_processor.merge_size**2
            index = 0
            for i in range(len(text)):
                while self.video_token in text[i]:
                    num_video_tokens = video_grid_thw[index].prod() // merge_length
                    text[i] = text[i].replace(self.video_token, "<|placeholder|>" * num_video_tokens, 1)
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", self.video_token)

        # noise tokens
        noise_token_num = self.robot_config.get("action_chunk_size")
        # Expand the action token into multiple tokens based on the grid size
        for i in range(len(text)):
            while self.action_token in text[i]:
                text[i] = text[i].replace(self.action_token, "<|placeholder|>" * noise_token_num, 1)
            text[i] = text[i].replace("<|placeholder|>", self.action_token)

        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])

        return_mm_token_type_ids = output_kwargs["text_kwargs"].pop("return_mm_token_type_ids", None)
        if return_mm_token_type_ids:
            text_inputs["mm_token_type_ids"] = self.create_mm_token_type_ids(text_inputs["input_ids"])

        return text_inputs, image_inputs, videos_inputs
    
    def _resolve_repo_ids(self, batch: dict, batch_size: int) -> list[str]:
        repo_ids = batch.get("repo_id")
        if repo_ids is None or len(repo_ids) == 0 or repo_ids[0] is None:
            repo_ids = list(self.robot_config["state_action_norm"].keys())[0]
        
        repo_ids = [repo_ids] * batch_size if isinstance(repo_ids, str) else repo_ids
        return list(repo_ids)
    
    def _normalize_robot_state(self, state: torch.Tensor, repo_id: str) -> torch.Tensor:
        norm_info = self.robot_config["state_action_norm"][repo_id]
        if norm_info is not None:
            state_mean = np.array(norm_info["state_mean"])
            state_std = np.array(norm_info["state_std"])
            state = (state - state_mean) / state_std
        return state
    
    def _unnormalize_robot_action(self, action: torch.Tensor, repo_id: str) -> torch.Tensor:
        norm_info = self.robot_config["state_action_norm"][repo_id]
        if norm_info is not None:
            action_mean = np.array(norm_info["action_mean"])
            action_std = np.array(norm_info["action_std"])
            action = action * action_std + action_mean
        return action


class RobotPointProcessorBase(RobotProcessorBase):

    def __init__(
        self, image_processor=None, tokenizer=None, video_processor=None,
        chat_template=None, robot_config=None, **kwargs
    ):
        super().__init__(
            image_processor, tokenizer, video_processor, chat_template, robot_config, **kwargs
        )
        self.point_token = getattr(self.tokenizer, "point_token", DEFAULT_POINT_TOKEN)
        self.point_token_id = self.tokenizer.convert_tokens_to_ids(DEFAULT_POINT_TOKEN)

    @property
    def model_input_names(self):
        return super().model_input_names + ["points", "npoints_in_batch"]

    def _resolve_points_workspace(self, repo_id: str, points_workspace: dict | None) -> dict:
        if points_workspace is None:
            return self.robot_config["points_workspace"][repo_id]
        if repo_id in points_workspace and isinstance(points_workspace[repo_id], dict):
            return points_workspace[repo_id]
        return points_workspace

    def _repo_config_flag(self, key: str, repo_id: str, default: bool = False) -> bool:
        values = self.robot_config.get(key)
        if values is None:
            return default
        if isinstance(values, dict):
            return values.get(repo_id, default)
        return bool(values)

    def _filter_points_by_workspace(self, point_cloud: np.ndarray, workspace: dict) -> np.ndarray:
        point_mask = (
            (point_cloud[..., 0] > workspace['X_BBOX'][0])
            & (point_cloud[..., 0] < workspace['X_BBOX'][1])
            & (point_cloud[..., 1] > workspace['Y_BBOX'][0])
            & (point_cloud[..., 1] < workspace['Y_BBOX'][1])
            & (point_cloud[..., 2] > workspace['Z_BBOX'][0])
            & (point_cloud[..., 2] < workspace['Z_BBOX'][1])
        )
        return point_cloud[point_mask]

    def _build_camera_point_cloud(self, mini_batch: dict, camera_name: str, workspace: dict) -> np.ndarray:
        cam_point_cloud = mini_batch[f'observation.points.{camera_name}']
        if isinstance(cam_point_cloud, torch.Tensor):
            cam_point_cloud = cam_point_cloud.detach().cpu().numpy()
        cam_point_cloud = np.asarray(cam_point_cloud, dtype=np.float32)

        rgb = mini_batch[f'observation.images.{camera_name}_image']
        if isinstance(rgb, torch.Tensor):
            rgb = rgb.detach().cpu().numpy()
        rgb = np.asarray(rgb)
        if rgb.ndim == 3 and rgb.shape[0] in (1, 3) and rgb.shape[-1] not in (1, 3):
            rgb = np.moveaxis(rgb, 0, -1)
        rgb = rgb.astype(np.float32) / 255.0
        rgb = rgb * 2 - 1
        cam_point_cloud = np.concatenate([cam_point_cloud, rgb], axis=2).reshape(-1, 6)
        return self._filter_points_by_workspace(cam_point_cloud, workspace)

    @staticmethod
    def _as_numpy_point_cloud(point_cloud) -> np.ndarray:
        if isinstance(point_cloud, torch.Tensor):
            return point_cloud.detach().cpu().numpy().astype(np.float32, copy=True)
        return np.asarray(point_cloud, dtype=np.float32).copy()

    @staticmethod
    def _point_camera_names(select_video_keys: list[str]) -> list[str]:
        camera_names = []
        for key in select_video_keys:
            camera_name = key.split(".")[-1]
            if camera_name.endswith("_image"):
                camera_name = camera_name[: -len("_image")]
            camera_names.append(camera_name)
        return camera_names

    def _build_point_cloud_from_cameras(
        self,
        mini_batch: dict,
        repo_id: str,
        workspace: dict,
    ) -> np.ndarray:
        camera_names = self._point_camera_names(self.robot_config["select_video_keys"][repo_id])
        point_clouds = [
            self._build_camera_point_cloud(mini_batch, camera_name, workspace)
            for camera_name in camera_names
        ]
        return np.concatenate(point_clouds, 0)

    def _build_existing_point_cloud(self, mini_batch: dict, workspace: dict) -> np.ndarray:
        point_cloud = self._as_numpy_point_cloud(mini_batch["observation.points"])
        if point_cloud.shape[-1] < 6:
            raise ValueError("Expected observation.points to include xyz and rgb features.")
        point_cloud[:, 3:6] = point_cloud[:, 3:6] * 2 - 1
        return self._filter_points_by_workspace(point_cloud, workspace)

    @staticmethod
    def _voxel_downsample_point_cloud(point_cloud: np.ndarray, voxel_size: float = 0.01) -> np.ndarray:
        import open3d as o3d

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3])
        pcd.colors = o3d.utility.Vector3dVector(point_cloud[:, 3:6])
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        return np.concatenate([np.asarray(pcd.points), np.asarray(pcd.colors)], axis=1).astype(np.float32)

    @staticmethod
    def _subsample_point_cloud(point_cloud: np.ndarray, max_npoints: int) -> np.ndarray:
        if len(point_cloud) <= max_npoints:
            return point_cloud
        ridxs = np.random.choice(len(point_cloud), size=max_npoints, replace=False)
        return point_cloud[ridxs]

    @staticmethod
    def _remove_robot_arm_points(point_cloud: np.ndarray, mini_batch: dict) -> np.ndarray:
        import open3d as o3d

        from pointact.utils.robot_box import remove_points_inside_robot

        robot_joint_bboxes = [
            o3d.geometry.OrientedBoundingBox(bbox[0], bbox[1], bbox[2])
            for bbox in mini_batch["observation.robot_joints_bbox"]
        ]
        point_cloud, _ = remove_points_inside_robot(point_cloud, robot_joint_bboxes)
        return point_cloud

    def _prepare_point_cloud_for_sample(
        self,
        mini_batch: dict,
        repo_id: str,
        workspace: dict,
        *,
        remove_arm: bool = False,
        voxel_size: float = 0.01,
    ) -> np.ndarray:
        has_existing_point_cloud = "observation.points" in mini_batch
        if has_existing_point_cloud:
            point_cloud = self._build_existing_point_cloud(mini_batch, workspace)
        else:
            point_cloud = self._build_point_cloud_from_cameras(mini_batch, repo_id, workspace)

        point_cloud = self._voxel_downsample_point_cloud(point_cloud, voxel_size=voxel_size)
        if remove_arm and (not has_existing_point_cloud or "observation.robot_joints_bbox" in mini_batch):
            point_cloud = self._remove_robot_arm_points(point_cloud, mini_batch)
        point_cloud = self._subsample_point_cloud(point_cloud, self.robot_config["max_npoints"][repo_id])
        return point_cloud

    @staticmethod
    def _center_point_cloud_and_state(
        point_cloud: torch.Tensor,
        state: torch.Tensor | None = None,
        center_state: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        if len(point_cloud) == 0:
            raise ValueError("Point cloud is empty after workspace filtering and downsampling.")
        point_center = point_cloud[:, :3].mean(0)
        point_cloud[:, :3] = point_cloud[:, :3] - point_center
        if state is not None:
            state = state.clone()
            if center_state:
                state[:3] = state[:3] - point_center
        return point_cloud, state, point_center
