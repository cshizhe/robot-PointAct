from __future__ import annotations

import os
import json
import gzip
import logging
from collections.abc import Iterable, Mapping
from typing import Any

import mujoco
import numpy as np
import robosuite
from robocasa.utils.dataset_registry import TASK_SET_REGISTRY
from robocasa.utils.dataset_registry_utils import get_task_horizon as _get_task_horizon
from robocasa.utils.env_utils import create_env
from robosuite.controllers.composite.composite_controller import HybridMobileBase
from robosuite.utils import camera_utils
from scipy.spatial.transform import Rotation as R

from pointact.utils.depth import depth_to_point_cloud



PANDA_OMRON_CAMERA_NAMES = [
    "robot0_agentview_left",
    "robot0_agentview_right",
    "robot0_eye_in_hand",
]

PANDA_OMRON_CAMERA_MAPPING = {
    "robot0_agentview_left": "left",
    "robot0_agentview_right": "right",
    "robot0_eye_in_hand": "wrist",
}

PANDA_OMRON_STATE_KEY_TO_RAW_KEY = {
    "state.gripper_qpos": "robot0_gripper_qpos",
    "state.base_position": "robot0_base_pos",
    "state.base_rotation": "robot0_base_quat",
    "state.end_effector_position_relative": "robot0_base_to_eef_pos",
    "state.end_effector_rotation_relative": "robot0_base_to_eef_quat_site",
    "state.gripper_qvel": "robot0_gripper_qvel",
    "state.end_effector_position_absolute": "robot0_eef_pos",
    "state.end_effector_rotation_absolute": "robot0_eef_quat_site",
    "state.joint_position": "robot0_joint_pos",
    "state.joint_position_cos": "robot0_joint_pos_cos",
    "state.joint_position_sin": "robot0_joint_pos_sin",
    "state.joint_velocity": "robot0_joint_vel",
}

PANDA_OMRON_STATE_KEYS = tuple(PANDA_OMRON_STATE_KEY_TO_RAW_KEY)

PANDA_OMRON_OBSERVATION_STATE_KEYS = (
    "state.end_effector_position_relative",
    "state.end_effector_rotation_relative",
    "state.gripper_qpos",
    "state.base_position",
    "state.base_rotation",
)

PANDA_OMRON_ACTION_KEYS = (
    "action.end_effector_position",
    "action.end_effector_rotation",
    "action.gripper_close",
    "action.base_motion",
    "action.control_mode",
)

PANDA_OMRON_RAW_ROBOT_KEYS = (
    "robot0_joint_pos",         # 7D
    "robot0_joint_pos_cos",     # 7D
    "robot0_joint_pos_sin",     # 7D
    "robot0_joint_vel",         # 7D
    "robot0_joint_acc",         # 7D
    "robot0_eef_pos",           # 3D
    "robot0_eef_quat",          # 4D
    "robot0_eef_quat_site",     # 4D
    "robot0_gripper_qpos",      # 2D
    "robot0_gripper_qvel",      # 2D
    "robot0_base_pos",          # 3D
    "robot0_base_quat",         # 4D
    "robot0_base_to_eef_pos",   # 3D
    "robot0_base_to_eef_quat",  # 4D
    "robot0_base_to_eef_quat_site",  # 4D
)



def get_dummy_action() -> dict[str, np.ndarray]:
    """Return a no-op PandaOmron dict action in the registered action schema."""
    return {
        "action.gripper_close": np.array([0.0], dtype=np.float32),
        "action.end_effector_position": np.zeros(3, dtype=np.float32),
        "action.end_effector_rotation": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        "action.base_motion": np.zeros(4, dtype=np.float32),
        "action.control_mode": np.array([0.0], dtype=np.float32),
    }


def convert_flat_action_to_panda_omron_action(
    action: np.ndarray,
) -> dict[str, np.ndarray]:
    """Convert public 13D action to named PandaOmron fields.

    Public action format is shared by replay caches, LeRobot datasets, and env.step:
    [eef_pos(3), eef_quat_xyzw(4), gripper_close(1), base_motion(4), control_mode(1)].
    The robosuite OSC controller still consumes the eef rotation as axis-angle, so
    quaternion-to-axis-angle conversion happens only in unmap_panda_omron_action().
    """
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    if action.shape[0] != 13:
        raise ValueError(
            "RoboCasa365 PandaOmron actions must be 13D "
            "[eef_pos(3), eef_quat_xyzw(4), gripper_close(1), base_motion(4), control_mode(1)], "
            f"got shape {action.shape}."
        )
    return {
        "action.end_effector_position": action[0:3].copy(),
        "action.end_effector_rotation": action[3:7].copy(),
        "action.gripper_close": action[7:8].copy(),
        "action.base_motion": action[8:12].copy(),
        "action.control_mode": action[12:13].copy(),
    }


def _quat_xyzw_to_rotvec(quat: Any) -> np.ndarray:
    """Convert a public xyzw quaternion action rotation to robosuite rotvec."""
    quat = np.asarray(quat, dtype=np.float32).reshape(-1)
    if quat.shape[0] != 4:
        raise ValueError(f"Expected eef quaternion action rotation to be 4D xyzw, got {quat.shape}.")
    norm = np.linalg.norm(quat)
    if norm <= 0:
        raise ValueError("EEF quaternion action rotation has zero norm.")
    return R.from_quat(quat / norm).as_rotvec().astype(np.float32)


def unmap_panda_omron_action(action: Mapping[str, Any]) -> dict[str, np.ndarray | float]:
    """Convert public PandaOmron action keys to robosuite controller parts.

    Input rotations are xyzw quaternions. The low-level robosuite controller expects
    axis-angle / rotation-vector deltas, so this is the single conversion boundary.
    """
    gripper_close = float(np.asarray(action["action.gripper_close"]).reshape(-1)[0])
    control_mode = float(np.asarray(action["action.control_mode"]).reshape(-1)[0])
    return {
        "robot0_right_gripper": -1.0 if gripper_close < 0.5 else 1.0,
        "robot0_right": np.concatenate(
            (
                np.asarray(action["action.end_effector_position"], dtype=np.float32).reshape(-1),
                _quat_xyzw_to_rotvec(action["action.end_effector_rotation"]),
            ),
            axis=-1,
        ),
        "robot0_base": np.asarray(action["action.base_motion"], dtype=np.float32).reshape(-1)[0:3],
        "robot0_torso": np.asarray(action["action.base_motion"], dtype=np.float32).reshape(-1)[3:4],
        "robot0_base_mode": -1.0 if control_mode < 0.5 else 1.0,
    }


def _gather_robot_observations(env) -> dict[str, np.ndarray]:
    """Gather qpos-based robot observations used by the PandaOmron schema."""
    observations = {}
    for robot_id, robot in enumerate(env.robots):
        sim = robot.sim
        gripper_names = {
            robot.get_gripper_name(arm): robot.gripper[arm]
            for arm in robot.arms
        }
        for part_name, indexes in robot._ref_joints_indexes_dict.items():
            qpos_values = []
            for joint_id in indexes:
                qpos_addr = sim.model.jnt_qposadr[joint_id]
                joint_type = sim.model.jnt_type[joint_id]
                if joint_type == mujoco.mjtJoint.mjJNT_FREE:
                    qpos_size = 7
                elif joint_type == mujoco.mjtJoint.mjJNT_BALL:
                    qpos_size = 4
                else:
                    qpos_size = 1
                qpos_values = np.append(
                    qpos_values,
                    sim.data.qpos[qpos_addr : qpos_addr + qpos_size],
                )

            if part_name in gripper_names:
                qpos_values = np.asarray(qpos_values)[::-1]
            if len(qpos_values) > 0:
                observations[f"robot{robot_id}_{part_name}"] = np.asarray(
                    qpos_values,
                    dtype=np.float32,
                )
    return observations


def map_panda_omron_state(raw_obs: Mapping[str, Any]) -> dict[str, np.ndarray]:
    state = {}
    for state_key, raw_key in PANDA_OMRON_STATE_KEY_TO_RAW_KEY.items():
        if raw_key not in raw_obs:
            raise KeyError(
                f"Missing RoboCasa365 observation key {raw_key!r} for {state_key!r}. "
                f"Available keys: {sorted(raw_obs.keys())}"
            )
        state[state_key] = np.asarray(raw_obs[raw_key], dtype=np.float32)
    return state


def load_episode_initial_meta_files(ep_dir):
    # 1. episode metadata: layout / style / objects / task language
    with open(os.path.join(ep_dir, "ep_meta.json"), "r") as f:
        ep_meta = json.load(f)

    # 2. compressed MuJoCo XML
    with gzip.open(os.path.join(ep_dir, "model.xml.gz"), "rt") as f:
        model_xml = f.read()

    # 3. flattened MuJoCo states
    data = np.load(os.path.join(ep_dir, "states.npz"), allow_pickle=True)
    # print("states.npz keys:", data.files)

    if "states" in data.files:
        states = data["states"]
    else:
        # fallback: inspect print output and replace this if needed
        states = data[data.files[0]]

    return ep_meta, model_xml, states

def reset_to(env, state: Mapping[str, Any]) -> None:
    """Reset a robosuite-style env to a saved simulator state.
    See: https://github.com/ARISE-Initiative/robosuite/blob/master/robosuite/scripts/playback_demonstrations_from_hdf5.py#L67
    """
    if "model" in state:
        if state.get("ep_meta", None) is not None:
            ep_meta = state["ep_meta"]
        else:
            ep_meta = {}

        env.set_ep_meta(ep_meta)

        env.reset()
        xml = env.edit_model_xml(state["model"])

        env.reset_from_xml_string(xml)
        env.sim.reset()

    if "states" in state:
        # Restore flattened MuJoCo state: (timesteps, dim)
        env.sim.set_state_from_flattened(state["states"][0])
        env.sim.forward()

    if hasattr(env, "update_sites"):
        env.update_sites()
    if hasattr(env, "update_state"):
        env.update_state()


class RoboCasa365Env:
    """RoboCasa365 PandaOmron environment wrapper."""

    CAMERA_MAPPING = PANDA_OMRON_CAMERA_MAPPING

    def __init__(
        self,
        env_name: str,
        split: str | None = "target",
        seed: int | None = 7,
        image_resolution: int = 256,
        camera_names: list[str] | None = None,
        use_depth: bool = False,
        use_point_cloud: bool = False,
        point_cloud_in_robot_base_frame: bool = True,
        enable_render: bool = True,
        terminate_on_success: bool = False,
        obj_registries: Iterable[str] | None = None,
        **kwargs: Any,
    ):
        if split not in {None, "pretrain", "target", "all"}:
            raise ValueError(
                f'split must be one of None, "pretrain", "target", "all"; got {split!r}.'
            )
        assert env_name in TASK_SET_REGISTRY["all_tasks"], f"Unknown RoboCasa365 env {env_name!r}"

        self.env_name = env_name
        self.split = split
        self.seed = seed
        self.image_resolution = int(image_resolution)
        self.camera_names = camera_names or PANDA_OMRON_CAMERA_NAMES
        self.use_depth = use_depth or use_point_cloud
        self.use_point_cloud = use_point_cloud
        self.point_cloud_in_robot_base_frame = point_cloud_in_robot_base_frame
        self.enable_render = enable_render
        self.terminate_on_success = terminate_on_success
        self.max_episode_steps = _get_task_horizon(env_name)

        if obj_registries is not None:
            kwargs = {**kwargs, "obj_registries": tuple(obj_registries)}

        self.env = create_env(
            env_name=self.env_name,
            robots="PandaOmron",
            camera_names=list(self.camera_names),
            camera_widths=self.image_resolution,
            camera_heights=self.image_resolution,
            seed=seed,
            split=split,
            render_onscreen=False,
            **kwargs,
        )
        logging.info(
            "Created RoboCasa365 env %s split=%s seed=%s", self.env_name, split, seed,
        )

    def reset(
        self,
        initial_state: Mapping[str, Any] | None = None,
        seed: int | None = None,
        initial_state_dir: str | None = None,
        step_after_reset: bool = True,
    ):
        if seed is not None:
            self.env.rng = np.random.default_rng(seed)

        if initial_state_dir is not None:
            ep_meta, model_xml, states = load_episode_initial_meta_files(initial_state_dir)
            initial_state = {
                "ep_meta": ep_meta,
                "model": model_xml,
                "states": states,
            }
            
        if initial_state is not None:
            reset_to(self.env, initial_state)
            if not step_after_reset:
                try:
                    raw_obs = self.env._get_observations(force_update=True)
                except TypeError:
                    raw_obs = self.env._get_observations()
                obs = self.get_observation(raw_obs)
                return obs, {"success": False, "intermediate_signals": {}}
            obs, _, _, info = self.step(get_dummy_action())
            info["success"] = False
            return obs, info

        raw_obs = self.env.reset()
        obs = self.get_observation(raw_obs)
        return obs, {"success": False, "intermediate_signals": {}}

    def step(self, action: Mapping[str, Any] | np.ndarray):
        if not isinstance(action, Mapping):
            action = convert_flat_action_to_panda_omron_action(action)

        robosuite_action = self._assemble_robosuite_action(
            unmap_panda_omron_action(action)
        )
        raw_obs, _, done, info = self.env.step(robosuite_action)

        success = bool(self.env._check_success())
        reward = 1.0 if success else 0.0
        obs = self.get_observation(raw_obs)

        info = dict(info)
        info["success"] = success
        info["intermediate_signals"] = {}
        if hasattr(self.env, "_get_intermediate_signals"):
            info["intermediate_signals"] = self.env._get_intermediate_signals()

        if self.terminate_on_success and success:
            done = True
        return obs, reward, bool(done), info

    def get_observation(self, raw_obs: Mapping[str, Any]) -> dict[str, Any]:
        raw_obs = dict(raw_obs)
        # raw_obs.update(_gather_robot_observations(self.env))

        obs: dict[str, Any] = {}
        state_obs = map_panda_omron_state(raw_obs)
        obs.update(state_obs)

        obs["observation.state"] = np.concatenate(
            [state_obs[key].reshape(-1) for key in PANDA_OMRON_OBSERVATION_STATE_KEYS],
            axis=0,
        ).astype(np.float32)

        for camera_name in self.camera_names:
            mapped_name = self.CAMERA_MAPPING.get(camera_name, camera_name)
            image = self._get_image(raw_obs, camera_name)
            obs[f"observation.images.{mapped_name}_image"] = image

            if self.use_depth:
                depth = self._get_metric_depth(raw_obs, camera_name)
                intrinsic, extrinsic = self._get_camera_matrices(camera_name)

                obs[f"observation.depths.{mapped_name}"] = depth.astype(np.float32)
                obs[f"observation.camera_intrinsics.{mapped_name}"] = intrinsic.astype(np.float32)
                obs[f"observation.camera_extrinsics.{mapped_name}"] = extrinsic.astype(np.float32)

                if self.use_point_cloud:
                    points = depth_to_point_cloud(depth, intrinsic, extrinsic).astype(np.float32)
                    if self.point_cloud_in_robot_base_frame:
                        points = self.convert_points_to_robot_base_frame(
                            points,
                            raw_obs["robot0_base_pos"],
                            raw_obs["robot0_base_quat"],
                        )
                    obs[f"observation.points.{mapped_name}"] = points

        obs["task"] = self.env.get_ep_meta().get("lang", "")
        return obs

    def close(self) -> None:
        self.env.close()

    def get_points_workspace(self, obs: Mapping[str, Any] | None = None):
        return {
            "X_BBOX": [-0.8, 0.8],
            "Y_BBOX": [-0.8, 0.8],
            "Z_BBOX": [0.0, 1.0],
        }

    def convert_points_to_robot_base_frame(
        self,
        points: np.ndarray,
        robot_base_pos: np.ndarray,
        robot_base_quat: np.ndarray,
    ) -> np.ndarray:
        points_shape = points.shape
        flat_points = points.reshape(-1, 3)
        rot = R.from_quat(robot_base_quat)
        flat_points = rot.inv().apply(flat_points - robot_base_pos.reshape(1, 3))
        return flat_points.reshape(points_shape).astype(np.float32)

    def _assemble_robosuite_action(
        self,
        action_parts: dict[str, np.ndarray | float],
    ) -> np.ndarray:
        action_parts = dict(action_parts)
        env_actions = []
        for robot in self.env.robots:
            controller = robot.composite_controller
            prefix = robot.robot_model.naming_prefix
            robot_action = np.zeros(controller.action_limits[0].shape, dtype=np.float32)

            for part_name in controller.part_controllers:
                start_idx, end_idx = controller._action_split_indexes[part_name]
                robot_action[start_idx:end_idx] = action_parts.pop(f"{prefix}{part_name}")

            if isinstance(controller, HybridMobileBase):
                robot_action[-1] = action_parts.pop(f"{prefix}base_mode")

            env_actions.append(robot_action)

        if action_parts:
            raise RuntimeError(f"Unprocessed RoboCasa365 action parts: {sorted(action_parts)}")
        return np.concatenate(env_actions).astype(np.float32)

    def _get_image(self, raw_obs: Mapping[str, Any], camera_name: str) -> np.ndarray:
        key = f"{camera_name}_image"
        if not self.enable_render:
            image = np.zeros(
                (self.image_resolution, self.image_resolution, 3),
                dtype=np.uint8,
            )
        elif key in raw_obs:
            image = np.asarray(raw_obs[key], dtype=np.uint8)
            image = np.ascontiguousarray(image[::-1])
        else:
            raise KeyError(
                f"Missing image observation for camera {camera_name!r}. "
                f"Available keys: {sorted(raw_obs.keys())}"
            )
        return image

    def _get_metric_depth(self, raw_obs: Mapping[str, Any], camera_name: str) -> np.ndarray:
        key = f"{camera_name}_depth"
        if key in raw_obs:
            depth = np.asarray(raw_obs[key], dtype=np.float32)
            if depth.ndim == 3 and depth.shape[-1] == 1:
                depth = depth[:, :, 0]
            depth = np.ascontiguousarray(depth[::-1])
        else:
            _, depth = self.env.sim.render(
                width=self.image_resolution,
                height=self.image_resolution,
                camera_name=camera_name,
                depth=True,
            )
            depth = np.ascontiguousarray(depth[::-1])
        return self._to_metric_depth(depth)

    def _to_metric_depth(self, depth: np.ndarray) -> np.ndarray:
        depth = np.asarray(depth, dtype=np.float32)
        finite_depth = depth[np.isfinite(depth)]
        normalized = finite_depth.size == 0 or float(np.max(finite_depth)) <= 1.0

        depth_mask = (depth < 0) | np.isnan(depth)
        if normalized:
            depth_mask |= depth > 1
        if np.any(depth_mask):
            logging.warning("Invalid RoboCasa365 depth ratio: %.6f", float(np.mean(depth_mask)))
            depth = depth.copy()
            depth[depth_mask] = 0

        if normalized:
            depth = camera_utils.get_real_depth_map(self.env.sim, depth)
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[:, :, 0]
        return np.asarray(depth, dtype=np.float32)

    def _get_camera_matrices(self, camera_name: str):
        intrinsic = camera_utils.get_camera_intrinsic_matrix(
            self.env.sim,
            camera_name,
            self.image_resolution,
            self.image_resolution,
        )
        extrinsic = camera_utils.get_camera_extrinsic_matrix(self.env.sim, camera_name)
        return intrinsic, extrinsic

    def __getattr__(self, name: str):
        if name != "env" and "env" in self.__dict__:
            return getattr(self.__dict__["env"], name)
        raise AttributeError(name)
