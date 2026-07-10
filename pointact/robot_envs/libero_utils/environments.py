import numpy as np
from termcolor import cprint
import logging
import pathlib

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from robosuite.utils import camera_utils

from pointact.utils.depth import depth_to_point_cloud


LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]


def _get_libero_env(task, resolution, seed, use_depth=False):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
        "camera_depths": use_depth,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


class LiberoEnv:
    """LanguageTable env."""

    CAMERA_MAPPING = {
        "agentview": "front",
        "robot0_eye_in_hand": "wrist",
    }

    def __init__(
        self, 
        task_suite_name: str, 
        seed: int = 7, 
        use_depth: bool = False,
        use_point_cloud: bool = False,
        delta_action: bool = False,
        num_steps_wait: int = 10,
        image_resolution: int = 256,
    ):
        assert task_suite_name in [
            "libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"
        ]
        self.seed = seed
        self.use_depth = use_depth or use_point_cloud
        self.use_point_cloud = use_point_cloud
        self.delta_action = delta_action
        self.num_steps_wait = num_steps_wait
        self.image_resolution = image_resolution

        # Initialize LIBERO task suite
        benchmark_dict = benchmark.get_benchmark_dict()
        self.task_suite = benchmark_dict[task_suite_name]()
        self.task = None
        self.env = None
        self.initial_states = None
        self.task_description = None

        self.num_tasks_in_suite = self.task_suite.n_tasks
        logging.info(f"Task suite: {task_suite_name}, num_tasks_in_suite: {self.num_tasks_in_suite}")

    def set_task(self, task_id):
        # Get task
        self.task = self.task_suite.get_task(task_id)

        # Get default LIBERO initial states
        self.initial_states = self.task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        self.env, self.task_description = _get_libero_env(
            self.task, self.image_resolution, self.seed, use_depth=self.use_depth
        )
        logging.info(f"\nTask: {self.task_description}")

    def reset(self, initial_state=None):
        self.env.reset()
        for robot in self.env.env.robots:
            robot.controller.use_delta = self.delta_action

        if initial_state is not None:
            # Set initial states
            self.env.set_init_state(initial_state)
        
        # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
        # and we need to wait for them to fall
        for t in range(self.num_steps_wait):
            obs, reward, done, info = self.step(LIBERO_DUMMY_ACTION)
            t += 1

        return obs

    def get_robot_joints_pose_size(self, robot_id=0, collision_geoms_only=True):
        model = self.env.sim.model
        data  = self.env.sim.data

        robot = self.env.robots[robot_id]
        robot_model = robot.robot_model                   # underlying MujocoModel

        if collision_geoms_only:
            geom_names = robot_model.contact_geoms        # only collision geoms
        else:
            # You can also include visual geoms if you want a 'visual' bbox
            geom_names = getattr(robot_model, "visual_geoms", robot_model.contact_geoms)

        def geom_pose_size(model, data, geom_id):
            """
            Compute world-frame pose and size for a single MuJoCo geom.
            Works for box / capsule / sphere primitives as an approximation.
            """
            size = np.array(model.geom_size[geom_id])        # half-sizes in local frame
            pos  = np.array(data.geom_xpos[geom_id])         # world position
            mat  = np.array(data.geom_xmat[geom_id]).reshape(3, 3)  # world rotation

            concat_vector = np.concatenate([pos, mat.reshape(-1), size*2])
            return concat_vector
            # return pos, mat, size * 2

        robot_joints = []
        for name in geom_names:
            gid = model.geom_name2id(name)
            joint = geom_pose_size(model, data, gid)
            robot_joints.append(joint)
        robot_joints = np.stack(robot_joints).astype(np.float32)

        return robot_joints

    def get_observation(self, raw_obs: dict):
        """
        raw_obs keys: ['robot0_joint_pos', 'robot0_joint_pos_cos', 'robot0_joint_pos_sin', 'robot0_joint_vel', 'robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos', 'robot0_gripper_qvel', 'agentview_image', 'agentview_depth', 'robot0_eye_in_hand_image', 'robot0_eye_in_hand_depth', 'akita_black_bowl_1_pos', 'akita_black_bowl_1_quat', 'akita_black_bowl_1_to_robot0_eef_pos', 'akita_black_bowl_1_to_robot0_eef_quat', 'akita_black_bowl_2_pos', 'akita_black_bowl_2_quat', 'akita_black_bowl_2_to_robot0_eef_pos', 'akita_black_bowl_2_to_robot0_eef_quat', 'cookies_1_pos', 'cookies_1_quat', 'cookies_1_to_robot0_eef_pos', 'cookies_1_to_robot0_eef_quat', 'glazed_rim_porcelain_ramekin_1_pos', 'glazed_rim_porcelain_ramekin_1_quat', 'glazed_rim_porcelain_ramekin_1_to_robot0_eef_pos', 'glazed_rim_porcelain_ramekin_1_to_robot0_eef_quat', 'plate_1_pos', 'plate_1_quat', 'plate_1_to_robot0_eef_pos', 'plate_1_to_robot0_eef_quat', 'robot0_proprio-state', 'object-state']
        """
        obs = {}

        for camera_name in ["agentview", "robot0_eye_in_hand"]:
            mapped_cam_name = self.CAMERA_MAPPING[camera_name]
            # image observations: flip images upside down
            obs[f"observation.images.{mapped_cam_name}_image"] = np.ascontiguousarray(
                raw_obs[f"{camera_name}_image"][::-1]
            )

            if self.use_depth:
                depth = raw_obs[f"{camera_name}_depth"][::-1]
                depth_mask = (depth < 0) | (depth > 1) | np.isnan(depth)
                if np.sum(depth_mask) > 0:
                    cprint(f'invalid depth value: {np.mean(depth_mask)}', 'red')
                depth[depth_mask] = 0
                depth = camera_utils.get_real_depth_map(
                    self.env.sim, depth
                )[:, :, 0]
                depth = depth
                intrinsic = camera_utils.get_camera_intrinsic_matrix(
                    self.env.sim, camera_name, self.image_resolution, self.image_resolution
                )
                extrinsic = camera_utils.get_camera_extrinsic_matrix(
                    self.env.sim, camera_name
                )
                obs[f"observation.depths.{mapped_cam_name}"] = depth.astype(np.float32)
                obs[f"observation.camera_intrinsics.{mapped_cam_name}"] = intrinsic.astype(np.float32)
                obs[f"observation.camera_extrinsics.{mapped_cam_name}"] = extrinsic.astype(np.float32)

                if self.use_point_cloud:
                    obs[f"observation.points.{mapped_cam_name}"] = depth_to_point_cloud(
                        depth, intrinsic, extrinsic
                    ).astype(np.float32)
                
        # add instruction
        obs["task"] = self.task_description

        # robot base and joint information
        obs["robot0_base_pos"] = self.env.robots[0].base_pos.astype(np.float32)
        obs["robot0_joints_pose_size"] = self.get_robot_joints_pose_size(
            robot_id=0, collision_geoms_only=False
        )
        obs["robot0_joint_pos"] = raw_obs['robot0_joint_pos'].astype(np.float32)

        # add simple robot state: end-effector pose (xyz, quat, openness)
        obs["observation.state"] = np.concatenate(
            [raw_obs["robot0_eef_pos"], 
             raw_obs["robot0_eef_quat"],
             raw_obs["robot0_gripper_qpos"]]
        ).astype(np.float32)

        # table height
        obs["table_height"] = self._get_table_height(obs["robot0_base_pos"])

        return obs

    def step(self, action):
        raw_obs, reward, done, info = self.env.step(action)
        obs = self.get_observation(raw_obs)

        info["success"] = reward > 0
        return obs, reward, done, info
    
    def _get_table_height(self, robot_base_pos):
        robot_base_height = robot_base_pos[-1]

        # manually define the table height
        if np.isclose(robot_base_height, 0.912):
            # tasks in libero_spatial, libero_goal, and some tasks in libero_10
            table_height = robot_base_height - 0.005
        elif np.isclose(robot_base_height, 0):
            # tasks in libero_object
            table_height = robot_base_height + 0.005
        elif np.isclose(robot_base_height, 0.42):
            # some tasks in libero_10
            table_height = robot_base_height + 0.01
        else:
            raise NotImplementedError(f'undefined table height: {robot_base_height}')

        return table_height
    
    def get_points_workspace(self, obs):
        table_height = obs["table_height"]

        points_workspace = {
            'X_BBOX': [-0.5, 0.5],
            'Y_BBOX': [-0.5, 0.5],
            'Z_BBOX': [table_height, 2],
        }

        return points_workspace
