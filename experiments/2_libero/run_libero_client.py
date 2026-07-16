import os
import sys
import collections
import dataclasses
import datetime as dt
import json
import logging
import pathlib
from pathlib import Path

import imageio
import numpy as np
import tqdm
import tyro

from pointact.utils.torch_utils import set_seed
from pointact.utils.rotation import convert_rotation

from pointact.utils.server_client import PolicyClient

from pointact.robot_envs.libero_utils.environments import LiberoEnv
from pointact.robot_envs.libero_utils.robot_joints import get_robot_joint_boxes


@dataclasses.dataclass
class ClientArgs:
    seed: int = 7  # Random Seed (for reproducibility)
    task_suite_name: str = "libero_spatial"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 10  # Number of rollouts per task
    image_size: int = 256
    
    repo_id: str | None = None
    post_process_action: bool = True
    replan_steps: int = 8
    pred_rot_type: str = 'euler'    # euler, rot6d
    use_depth: bool = False
    delta_action: bool = False
    remove_arm: bool = False

    save_dir: str = ""
    save_video: bool = False
    verbose: bool = False

    host: str = "localhost"
    port: int = 5555


def setup_logging(filename=None):
    # Create root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Formatter
    # formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    formatter = logging.Formatter('%(message)s')

    # File handler
    if filename is not None:
        file_handler = logging.FileHandler(filename)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


def main(args: ClientArgs) -> None:
    assert args.pred_rot_type in ["rot6d", "euler"]

    policy_client = PolicyClient(args.host, args.port)

    is_server_running = False
    while not is_server_running:
        is_server_running = policy_client.ping()
    print(f'Server is running on host {args.host} port {args.port}')

    video_out_dir = None
    log_filename = None
    if args.save_dir:
        date_base = Path(args.save_dir, args.task_suite_name+'-'+dt.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        date_base.parent.mkdir(parents=True, exist_ok=True)
        log_filename = f"{date_base}.log"
        print(f"Save logs in {log_filename}")
        if args.save_video:
            video_out_dir = f"{date_base}+videos"
            pathlib.Path(video_out_dir).mkdir(parents=True, exist_ok=True)
    
    setup_logging(log_filename)
    logging.info(f"Arguments: {json.dumps(dataclasses.asdict(args), indent=4)}")

    # Set random seed
    set_seed(args.seed)

    # Initialize LIBERO task suite
    libero_env = LiberoEnv(
        task_suite_name=args.task_suite_name, 
        seed=args.seed, 
        use_depth=args.use_depth,
        use_point_cloud=args.use_depth,
        delta_action=args.delta_action,
        num_steps_wait=args.num_steps_wait,
        image_resolution=args.image_size,
    )

    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520 + 50  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(libero_env.num_tasks_in_suite)):
        # if task_id != 0:
        #     continue
        # else:
        #     import h5py
        #     input_h5_path = 'robot_data/libero/raw_hdf5/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5'
        #     with h5py.File(input_h5_path, 'r') as f:
        #         initial_states = []
        #         for demo in f["data"].values():
        #             orig_states = demo['states'][()][0]
        #             initial_states.append(orig_states)
        #     print(len(initial_states))

        # Get task
        libero_env.set_task(task_id)
        initial_states = libero_env.initial_states
        task_description = libero_env.task_description

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            # Set initial states
            obs = libero_env.reset(initial_state=initial_states[episode_idx])

            action_plan = collections.deque()

            # Setup
            replay_images = []

            if args.verbose:
                logging.info(f"Starting episode {task_episodes + 1}...")

            for t in range(max_steps):
                # Save preprocessed image for replay video
                replay_images.append(obs["observation.images.front_image"])

                # robot state: [xyz: 3d, quat: 4d, qpos: 2d]
                # convert rotation: quat to euler/rot6d
                rot = obs["observation.state"][3:7]
                if args.pred_rot_type == 'euler':
                    rot = convert_rotation(
                        rot, "quat", "euler", quat_order_src="xyzw", euler_order_dst="xyz"
                    )
                elif args.pred_rot_type == 'rot6d':
                    rot = convert_rotation(
                        rot, "quat", "rot6d", quat_order_src="xyzw"
                    )
                state = np.concatenate(
                    [obs["observation.state"][:3], rot, obs["observation.state"][7:]]
                )

                if not action_plan:
                    batch = {
                        "observation.images.front_image": [obs["observation.images.front_image"]],
                        "observation.images.wrist_image": [obs["observation.images.wrist_image"]],
                        "observation.state": [state],
                        "task": [obs["task"]],
                        "repo_id": [args.repo_id],
                    }
                        
                    if args.use_depth:
                        batch.update({
                            "observation.points.front": [obs["observation.points.front"]],
                            "observation.points.wrist": [obs["observation.points.wrist"]],
                        })
                        if args.remove_arm:
                            batch["observation.robot_joints_bbox"] = [get_robot_joint_boxes(
                                obs["robot0_joints_pose_size"], return_o3d=False
                            )]

                    points_workspace = libero_env.get_points_workspace(obs)
                    
                    ov_out = policy_client.get_action(
                        batch, options={
                            "pred_rot_type": args.pred_rot_type, 
                            "points_workspace": points_workspace,
                            "remove_arm": args.remove_arm,
                        }
                    )
                    action_chunk = ov_out.action[0].copy()

                    if args.post_process_action:
                        action_chunk[..., -1] = 2 * (1 - action_chunk[..., -1]) - 1

                    assert len(action_chunk) >= args.replan_steps, (
                        f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                    )
                    action_plan.extend(action_chunk[: args.replan_steps])

                action = action_plan.popleft()
                action_rot = convert_rotation(
                    action[3:7], "quat", "axisangle", quat_order_src="xyzw"
                    # action[3:7], "quat", "euler", quat_order_src="xyzw", euler_order_dst="xyz"
                )
                action = np.concatenate([action[:3], action_rot, action[7:]])
                # Execute action in environment
                obs, reward, done, info = libero_env.step(action.tolist())
                if done:
                    task_successes += 1
                    total_successes += 1
                    break

            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            if video_out_dir is not None:
                imageio.mimwrite(
                    pathlib.Path(video_out_dir) / f"{task_segment}_rollout{episode_idx}_{suffix}.mp4",
                    [np.asarray(x) for x in replay_images],
                    fps=10,
                )
            # Log current results
            if args.verbose:
                logging.info(f"Success: {done}")
                logging.info(f"# episodes completed so far: {total_episodes}")
                logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        # Log final results
        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")


if __name__ == "__main__":
    tyro.cli(main)
