import os
from typing import List
import numpy as np
import dataclasses
import json
import jsonlines
from termcolor import colored, cprint
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

import torch.multiprocessing as mp

import tyro
from pyrep.errors import ConfigurationPathError, IKError
from rlbench.backend.exceptions import InvalidActionError
from rlbench.backend.utils import task_file_to_task_class

from pointact.robot_envs.rlbench_utils.environments import (
    Mover, RLBenchEnv,
    get_robot_joints_pose_size
)
from pointact.robot_envs.rlbench_utils.eval_utils import (
    project_action_chunks_on_image, save_images_to_video,
    set_random_seed, write_to_file,
    get_rlbench_robot_workspace
)

from pointact.utils.rotation import convert_rotation
from pointact.utils.server_client import PolicyClient


CAMERA_NAMES = ("left_shoulder", "right_shoulder", "wrist", "front")


@dataclasses.dataclass
class ClientArgs:
    seed: int = 7  # Random Seed (for reproducibility)

    # rlbench
    # test_level: str = "l1"  # l1, l2, l3, l4
    taskvar: str = ""
    taskvar_file: str = ""
    num_episodes: int = 25  # Number of episodes per taskvar
    max_steps: int = 25  # Maximum number of steps per episode
    image_size: int = 256
    microstep_data_dir: str = ""
    select_cameras: List[str] = ("front", "left_shoulder", "right_shoulder", "wrist")

    no_env_reward: bool = False
    stop_with_policy: bool = False

    # model
    repo_id: str = ""  # post-process: norm and unnorm state/action
    pretrained_path: str = ""
    pred_rot_type: str = "euler" # euler, quat, rot6d
    delta_action: bool = False

    replan_steps: int = 8
    clip_within_workspace: bool = False
    action_ensemble: bool = False

    # point cloud
    remove_arm: bool = False

    # server-client
    num_workers: int = 1
    host: str = "localhost"
    port: int = 15000
    
    # output
    save_video: bool = False
    save_obs_outs: bool = False
    project_action_on_image: bool = False
    save_dir: str = ""

    verbose: bool = False


def save_obs_state(output_dir, obs_id, obs_state_dict):
    np.savez(os.path.join(output_dir, f"obs_{obs_id:06d}.npz"), **obs_state_dict)

def write_transition(output_dir, transition):
    with jsonlines.open(os.path.join(output_dir, "transitions.jsonl"), "a", flush=True) as outf:
        outf.write(transition)


def is_action_similar(prev_action, action):
    pos_delta = prev_action[:3] - action[:3]
    prev_quat = prev_action[3:7]
    curr_quat = action[3:7]
    quat_norm = float(np.dot(prev_quat, prev_quat) * np.dot(curr_quat, curr_quat))
    quat_dot = float(np.dot(prev_quat, curr_quat))
    if (
        np.dot(pos_delta, pos_delta) < 1e-3
        and quat_norm > 0.0
        and (quat_dot * quat_dot / quat_norm) > 0.999
        and (prev_action[7] > 0.5) == (action[7] > 0.5)
    ):
        return True
    return False


def producer_fn(proc_id, args, taskvar, pred_file, producer_queue):

    policy_client = PolicyClient(args.host, args.port)

    is_server_running = False
    while not is_server_running:
        is_server_running = policy_client.ping()
    print(f"Server is running on host {args.host} port {args.port}")

    set_random_seed(args.seed)

    # Initialize RLBench env
    task_str, variation_id = taskvar.split("+")
    variation_id = int(variation_id)

    env = RLBenchEnv(
        data_path=args.microstep_data_dir,
        apply_rgb=True,
        apply_depth=True,
        apply_pc=True,
        apply_mask=False,
        apply_cameras=CAMERA_NAMES,
        headless=True,
        image_size=[args.image_size, args.image_size],
        cam_rand_factor=0,
        cam_params_to_opencv=True,
        use_metric_depth=True,
    )

    env.env.launch()
    task_type = task_file_to_task_class(task_str)
    task = env.env.get_task(task_type)
    task.set_variation(variation_id)
    move = Mover(task, max_tries=10)

    if args.microstep_data_dir != "":
        episodes_dir = os.path.join(args.microstep_data_dir, task_str, f"variation{variation_id}", "episodes")
        demos = []
        if os.path.exists(str(episodes_dir)):
            episode_ids = os.listdir(episodes_dir)
            episode_ids.sort(key=lambda ep: int(ep[7:]))
            for idx, ep in enumerate(episode_ids):
                try:
                    demo = env.get_demo(task_str, variation_id, idx, load_images=False)
                    demos.append(demo)
                except Exception as e:
                    print("\tProblem to load demo_id:", idx, ep)
                    print(e)
        num_episodes = len(demos)
    else:
        demos = None
        num_episodes = args.num_episodes

    success_rate = 0
    for episode_id in tqdm(range(num_episodes)):
        replay_images = []

        if demos is None:
            instructions, obs = task.reset()
        else:
            print("Resetting to demo", episode_id)
            instructions, obs = task.reset_to_demo(demos[episode_id])  # type: ignore

        # instruction = random.choice(instructions)
        instruction = instructions[0]
        action_plan = []

        obs_state_dict = env.get_observation(obs)
        move.reset(obs_state_dict["gripper"])

        if args.save_obs_outs:
            output_dir = os.path.join(args.save_dir, "obs_outs", taskvar, f"episode_{episode_id:06d}")
            replans_dir = os.path.join(output_dir, "replans")
            os.makedirs(replans_dir, exist_ok=True)
            open(os.path.join(output_dir, "transitions.jsonl"), "w").close()
            with open(os.path.join(output_dir, "meta.json"), "w") as outf:
                json.dump(
                    {
                        "taskvar": taskvar,
                        "task": task_str,
                        "variation": variation_id,
                        "episode_id": episode_id,
                        "instruction": instruction,
                        "seed": args.seed,
                        "repo_id": args.repo_id,
                        "replan_steps": args.replan_steps,
                        "pred_rot_type": args.pred_rot_type,
                        "select_cameras": list(args.select_cameras),
                    },
                    outf,
                    indent=2,
                )
            save_obs_state(output_dir, 0, obs_state_dict)

        prev_action = None
        current_replan_file = None
        current_replan_step = None
        has_reward_accumulated = False
        for step_id in range(args.max_steps):
            state = obs_state_dict["gripper"]  
            # convert rotation: quat to euler/rot6d/axisangle
            if args.pred_rot_type in ["euler", "rot6d", "axisangle"]:
                rot = convert_rotation(
                    state[3:7], "quat", args.pred_rot_type, quat_order_src="xyzw", euler_order_dst="xyz"
                )
                state = np.concatenate([state[:3], rot, state[7:]], 0)
            
            # if not action_plan:
            if step_id % args.replan_steps == 0:
                batch = {
                    "observation.state": [state],
                    "task": [instruction],
                    "repo_id": [args.repo_id],
                }
                if args.remove_arm:
                    batch["observation.robot_joints_bbox"] = [
                        get_robot_joints_pose_size(obs_state_dict["arm_links_info"])
                    ]
                    
                if args.save_video:
                    replay_images.append([])
                for cam_name in args.select_cameras:
                    cam_idx = CAMERA_NAMES.index(cam_name)
                    rgb_image = obs_state_dict["rgb"][cam_idx]
                    batch[f"observation.images.{cam_name}_image"] = [rgb_image]
                    if args.save_video:
                        replay_images[-1].append(np.array(rgb_image))

                    # batch[f"observation.depths.{cam_name}"] = [obs_state_dict["depth"][cam_idx]]
                    # batch[f"observation.camera_extrinsics.{cam_name}"] = [obs_state_dict["camera_extrinsics"][cam_name]]
                    # batch[f"observation.camera_intrinsics.{cam_name}"] = [obs_state_dict["camera_intrinsics"][cam_name]]

                    batch[f"observation.points.{cam_name}"] = [obs_state_dict["pc"][cam_idx]]

                ov_out = policy_client.get_action(
                    batch, options={"pred_rot_type": args.pred_rot_type, "remove_arm": args.remove_arm}
                )
                action_chunk = ov_out.action[0].copy()

                # TODO: rotation delta is not simple addition, here only perform delta on position
                if args.delta_action:
                    abs_action_chunk = [state[:3]]
                    for action in action_chunk:
                        abs_action_chunk.append(abs_action_chunk[-1] + action[:3])
                    action_chunk = np.concatenate(
                        [np.array(abs_action_chunk[1:]), action_chunk[:, 3:]], 1
                    )

                assert len(action_chunk) >= args.replan_steps, (
                    f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                )
                if args.action_ensemble:
                    for i, action in enumerate(action_plan):
                        # action_plan[i] = (action + action_chunk[i]) / 2
                        # TODO: only ensemble position not rotation
                        action_plan[i][:3] = (action[:3] + action_chunk[i, :3]) / 2
                        action_plan[i][3:] = action_chunk[i, 3:]
                    action_plan.extend(action_chunk[len(action_plan):])
                else:
                    action_plan.extend(action_chunk[: args.replan_steps])

                if args.save_video:
                    if args.project_action_on_image:
                        for c, cam_name in enumerate(args.select_cameras):
                            extrinsics = obs_state_dict["camera_extrinsics"][cam_name]
                            intrinsics = obs_state_dict["camera_intrinsics"][cam_name]
                            replay_images[-1][c] = project_action_chunks_on_image(
                                replay_images[-1][c], action_chunk, extrinsics, intrinsics
                            )
                    replay_images[-1] = np.concatenate(replay_images[-1], 1)

                if args.save_obs_outs:
                    current_replan_file = os.path.join("replans", f"step_{step_id:06d}.npz")
                    np.savez(os.path.join(output_dir, current_replan_file), action_chunk=action_chunk)
                current_replan_step = step_id

            # action = action_plan.popleft()
            action = action_plan[0]
            action_plan = action_plan[1:]

            if prev_action is None:
                prev_action = action.copy()
            elif args.stop_with_policy:
                if is_action_similar(prev_action, action):
                    if args.save_obs_outs:
                        write_transition(
                            output_dir,
                            {
                                "step_id": step_id,
                                "obs_before": f"obs_{step_id:06d}.npz",
                                "obs_after": None,
                                "executed_action": None,
                                "candidate_action": np.asarray(action).tolist(),
                                "reward": reward,
                                "terminate": True,
                                "is_replan": step_id == current_replan_step,
                                "replan_file": current_replan_file,
                                "action_chunk_index": None if current_replan_step is None else step_id - current_replan_step,
                                "stop_reason": "policy",
                            },
                        )
                    break
                prev_action = action.copy()

            if args.clip_within_workspace:
                workspace = get_rlbench_robot_workspace()
                action[0] = np.clip(action[0], a_min=workspace["X_BBOX"][0], a_max=workspace["X_BBOX"][1])
                action[1] = np.clip(action[1], a_min=workspace["Y_BBOX"][0], a_max=workspace["Y_BBOX"][1])
                action[2] = np.clip(action[2], a_min=workspace["Z_BBOX"][0], a_max=workspace["Z_BBOX"][1])

            # update the observation based on the predicted action
            try:
                obs, reward, terminate, _ = move(action, verbose=False)
                # error_type = None
                obs_state_dict = env.get_observation(obs)  # type: ignore
                if args.save_obs_outs:
                    save_obs_state(output_dir, step_id + 1, obs_state_dict)
                    write_transition(
                        output_dir,
                        {
                            "step_id": step_id,
                            "obs_before": f"obs_{step_id:06d}.npz",
                            "obs_after": f"obs_{step_id + 1:06d}.npz",
                            "executed_action": np.asarray(action).tolist(),
                            "reward": float(reward),
                            "terminate": bool(terminate),
                            "is_replan": step_id == current_replan_step,
                            "replan_file": current_replan_file,
                            "action_chunk_index": None if current_replan_step is None else step_id - current_replan_step,
                        },
                    )
                if reward == 1:
                    if not has_reward_accumulated:
                        success_rate += 1 / num_episodes
                        has_reward_accumulated = True
                    if not args.no_env_reward:
                        break
                # if terminate:
                #     print("The episode has terminated!")

            except (IKError, ConfigurationPathError, InvalidActionError) as e:
                print(args.taskvar, episode_id, step_id, e)
                # error_type = str(e)
                if args.save_obs_outs:
                    write_transition(
                        output_dir,
                        {
                            "step_id": step_id,
                            "obs_before": f"obs_{step_id:06d}.npz",
                            "obs_after": None,
                            "executed_action": np.asarray(action).tolist(),
                            "reward": 0.0,
                            "terminate": True,
                            "is_replan": step_id == current_replan_step,
                            "replan_file": current_replan_file,
                            "action_chunk_index": None if current_replan_step is None else step_id - current_replan_step,
                            "error": f"{type(e).__name__}: {e}",
                        },
                    )
                reward = 0
                break

        print(
            taskvar, "Episode", episode_id, "Step", step_id+1,
            "Reward", reward, "Accumulated SR: %.2f" % (success_rate * 100), 
            "Estimated SR: %.2f" % (success_rate * num_episodes / (episode_id+1) * 100)
        )
        
        if args.save_video:
            reward_str = "success" if reward == 1 else "failure"
            video_path = os.path.join(
                args.save_dir, "videos", f"{taskvar}_episode_{episode_id:06d}_{reward_str}.mp4"
            )
            os.makedirs(os.path.dirname(video_path), exist_ok=True)
            save_images_to_video(replay_images, video_path, fps=5)


    print(f"Success Rate: {success_rate * 100:.2f}%")

    if pred_file is not None:
        write_to_file(
            pred_file,
            {
                "task": task_str, "variation": variation_id,
                "num_episodes": num_episodes, "sr": success_rate,
                "checkpoint": args.pretrained_path,
            }
        )

    env.env.shutdown()
    print(colored(f"Taskvar: {taskvar} SR: {success_rate:.2f}", "black", "on_yellow"))

    producer_queue.put(proc_id)
    

def main(args: ClientArgs) -> None:

    if os.path.exists(os.path.join(args.pretrained_path, "processor_config.json")):
        processor_config = json.load(open(os.path.join(args.pretrained_path, "processor_config.json")))
        if not args.repo_id:
            args.repo_id = list(processor_config["robot_config"]["features"].keys())[0]
            print(f"Reset repo_id as {args.repo_id}")
        select_cameras = processor_config["robot_config"]["select_video_keys"][args.repo_id]
        args.select_cameras = tuple([cam_name.split(".")[-1][:-len("_image")] for cam_name in select_cameras])
    print("select cameras", args.select_cameras)

    existed_taskvars = set()
    if args.save_dir:
        pred_file = os.path.join(args.save_dir, "results.jsonl")
        if os.path.exists(pred_file):
            with jsonlines.open(pred_file, "r") as f:
                for item in f:
                    if item["checkpoint"] == args.pretrained_path:
                        existed_taskvars.add("{}+{}".format(item["task"], item["variation"]))
            print(f"Exist #{len(existed_taskvars)} taskvars in {pred_file}.")
    else:
        pred_file = None

    if args.taskvar:
        taskvars = [args.taskvar]
    elif args.taskvar_file:
        taskvars = json.load(open(args.taskvar_file))
    else:
        raise NotImplementedError("Must define taskvar or taskvar_file")
    
    taskvars = [taskvar for taskvar in taskvars if taskvar not in existed_taskvars]
    print("Evaluate #taskvars", len(taskvars))
    if len(taskvars) == 0:
        return
    
    producer_queue = mp.Queue(args.num_workers * 2)

    producers = {}        
    i = 0
    while i < len(taskvars):
        taskvar = taskvars[i]
        if len(producers) < args.num_workers:
            print("start", i, taskvar)
            producer = mp.Process(
                target=producer_fn, 
                args=(i, args, taskvar, pred_file, producer_queue),
                name=taskvar
            )
            producer.start()
            producers[i] = producer
            i += 1
        else:
            proc_id = producer_queue.get()
            producers[proc_id].join()
            del producers[proc_id]

    for p in producers.values():
        p.join()


if __name__ == "__main__":
    tyro.cli(main)
