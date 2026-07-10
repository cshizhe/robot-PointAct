TASK_PLANS = {
    'pick_up_cup': [
        {"keysteps": (0, 2), "plan": "grasp {obj}", "obj_id": 83}, 
        {"keysteps": (2, -1), "plan": "lift {obj} up", "obj_id": 83},
        {"keysteps": (-1, ), "plan": "stop"}
    ]
}

def gen_gripper_state_cot(gripper, prefix):
    pos = ', '.join([f'{v:.2f}' for v in gripper[0]])
    rot = ', '.join([f'{v:.0f}' for v in gripper[1]])
    openness = 'open' if gripper[2] > 0.5 else 'closed'
    gripper_str = f'{prefix} position [{pos}] with rotation [{rot}] and being {openness}.\n'
    return gripper_str
    
def gen_cot(x, plan, target_gripper):
    # gripper state
    cot = gen_gripper_state_cot(x['gripper'], "The gripper is at")

    # objects
    cot += 'Objects on the table:\n'
    obj_ids = []
    for k, obj in x.items():
        if k != 'gripper':
            # if obj['front_center'] is None:
            #     continue
            cot += f'- Object {len(obj_ids)+1}: ' + obj['name'] + ', center at ['
            pos = ', '.join([f'{v:.2f}' for v in obj['pcd_center']])
            # pos = ', '.join([f'{v:.1f}' for v in obj['front_center']])
            cot += pos + ']\n'
            obj_ids.append(int(k))
    
    # add subplans
    if 'obj_id' in plan:
        obj_id = str(plan['obj_id'])
        obj_name = x[obj_id]['name']
        obj_pos = ', '.join([f'{v:.2f}' for v in x[obj_id]['pcd_center']])
        obj_str = f'{obj_name} (center at [{obj_pos}])'
        plan_str = plan['plan'].format(obj=obj_str)
    else:
        plan_str = plan['plan']

    cot = cot + f'Subplan: {plan_str}.\n'

    # add target location
    if target_gripper is not None:
        target_gripper_str = gen_gripper_state_cot(target_gripper, prefix='The robot should move to')
        cot += target_gripper_str
    
    cot = cot.strip()
    return cot

def main():
    data = {}

    taskvar = 'pick_up_cup+8'
    task_str, variation_id = taskvar.split('+')

    keystep_dir = f'../robot_data/rlbench/raw/keysteps/{taskvar}'

    with jsonlines.open(f'../robot_data/rlbench/raw/objects_meta/{taskvar}.jsonl', 'r') as f:
        for episode_data in f:
            episode_id = episode_data[0].split('_')[-1] # 'episode#'
            
            with lmdb.open(keystep_dir, readonly=True) as lmdb_env:
                with lmdb_env.begin() as txn:
                    value = txn.get(episode_id.encode('ascii'))
                    value = msgpack.unpackb(value)
                    key_frameids = value['key_frameids']

            plans = TASK_PLANS[task_str]
            
            micro2plan, micro2target = {}, {}
            for tp, plan in enumerate(plans):
                if tp < len(plans) - 1:
                    plan_keystep_start, plan_keystep_end_noinclude = plan['keysteps']
                    plan_microstep_start = key_frameids[plan_keystep_start]
                    plan_microstep_end_noinclude = key_frameids[plan_keystep_end_noinclude]
                    target_microstep = key_frameids[plans[tp+1]['keysteps'][0]]
                    target_gripper = episode_data[1][target_microstep]['gripper']
                    for t in range(plan_microstep_start, plan_microstep_end_noinclude):
                        micro2plan[t] = plan
                        micro2target[t] = target_gripper
                else:
                    plan_microstep_start = key_frameids[plan['keysteps'][0]]
                    micro2plan[plan_microstep_start] = plan
                    micro2target[plan_microstep_start] = None
            
            for t, x in enumerate(episode_data[1]):
                eid = episode_id[len('episode'):]
                data[f"{eid}-{t}"] = gen_cot(x, micro2plan[t], micro2target[t])
            
            # break