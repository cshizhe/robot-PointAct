import numpy as np

def get_rlbench_labels(task=None, table=True, robot=True, wall=True, floor=True):
    # OLD with polarnet bug
    # # 204,208: table; 199-203: wall; 246: floor; 257?
    # # 212-216, 221, 222, 225: robotic arm
    # LABELS = []
    # if table:
    #     LABELS += [204, 208]
    # if robot:
    #     LABELS += [210, 211, 212, 213, 214, 215, 216, 221, 222, 225]
    # if wall:
    #     LABELS += [199, 200, 201, 202, 203]
    # if floor:
    #     LABELS += [246, 257]
    LABELS = []
    if table:
        assert task is not None
        LABELS += [48, 51, 52]
        if task == "close_jar_peract":
            LABELS += [86]
        elif task == "close_jar":
            LABELS += [86]
        elif task == "light_bulb_in_peract":
            LABELS += [98]
        elif task == "change_channel":
            LABELS += [102]
        elif task == "empty_container":
            LABELS += [86]
        elif task == "light_bulb_in":
            LABELS += [97]
        elif task == "light_bulb_out":
            LABELS += [95]
        elif task == "open_jar":
            LABELS += [89]
        elif task == "tv_on":
            LABELS += [102]
        elif task == 'close_fridge':
            LABELS += [81]
    if floor:
        LABELS += [8, 9, 10, 70, 71]
    if robot:
        LABELS += list(range(12, 48)) + [67, 68, 69]
    if wall:
        LABELS += [53, 54, 55, 56, 57]

    undefined = 65535
    LABELS += [undefined]

    return LABELS

def generate_fg_mask(gt_sem, task=None, keep_robot=True, keep_table=False):
    bg_labels = get_rlbench_labels(
        task, table=(not keep_table), wall=True, floor=True, robot=(not keep_robot)
    )
    bg_mask = np.zeros(gt_sem.shape, dtype=bool)
    for bg_label in bg_labels:
        bg_mask = bg_mask | (gt_sem == bg_label)
    fg_mask = ~bg_mask

    return fg_mask

def compute_mask_bbox_and_center(binary_mask):
    # Ensure the mask is a numpy array
    binary_mask = np.asarray(binary_mask)
    
    # Find indices of non-zero (object) pixels
    rows, cols = np.nonzero(binary_mask)
    
    # If no object pixels are found, return None
    if len(rows) == 0 or len(cols) == 0:
        return None, None
    
    # Compute bounding box
    min_row, max_row = np.min(rows), np.max(rows)
    min_col, max_col = np.min(cols), np.max(cols)
    bbox = [min_row.item(), min_col.item(), max_row.item(), max_col.item()]
    
    # Compute center
    center_row = (min_row + max_row) / 2
    center_col = (min_col + max_col) / 2
    center = [center_row.item(), center_col.item()]

    return bbox, center

def compute_point_cloud_bbox_and_center(points):
    # Ensure points is a numpy array and has correct shape
    points = np.asarray(points)

    # Validate input
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Input must be a 2D array with shape (N, 3)")
    
    # Compute bounding box extremes
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0)
    
    # # Compute bounding box dimensions
    # bbox_dimensions = max_coords - min_coords
    bbox = np.concatenate([min_coords, max_coords], 0).tolist()
    
    # Compute center
    center = np.mean(points, axis=0).tolist()
    
    return bbox, center
