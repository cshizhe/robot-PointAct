import torch
from transformers.data.data_collator import DefaultDataCollator

from pointact.constants import IGNORE_INDEX


def pad_sequence(sequences, padding_side="right", padding_value=0):
    assert padding_side in ["right", "left"]
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    max_len = max(len(seq) for seq in sequences)
    batch_size = len(sequences)
    output = sequences[0].new_full((batch_size, max_len) + trailing_dims, padding_value)
    for i, seq in enumerate(sequences):
        length = seq.size(0)
        if padding_side == "right":
            output.data[i, :length] = seq
        else:
            output.data[i, -length:] = seq
    return output


class DataCollator(DefaultDataCollator):
    """Collate multimodal examples."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, examples):
        batch_input_ids = []
        batch_label_ids = []
        batch_pixel_values = []
        batch_pixel_video_values = []
        batch_video_thw = []
        batch_image_thw = []
        batch_second_per_grid_ts = []
        batch_points, npoints_in_batch = [], []

        batch_actions = []
        batch_states = []
        batch_action_is_pad = []

        is_labels_provided = "labels" in examples[0]
        for example in examples:
            keys = example.keys()
            batch_input_ids.append(example["input_ids"])

            if is_labels_provided:
                batch_label_ids.append(example["labels"])

            if "pixel_values_videos" in keys:
                batch_pixel_video_values.append(example["pixel_values_videos"])
                batch_video_thw.append(example["video_grid_thw"])
            elif "pixel_values" in keys:
                batch_pixel_values.append(example["pixel_values"])
                batch_image_thw.append(example["image_grid_thw"])

            if "second_per_grid_ts" in keys:
                batch_second_per_grid_ts.extend(example["second_per_grid_ts"])

            if "actions" in keys:
                batch_actions.append(example["actions"])
                batch_action_is_pad.append(example["action_is_pad"])

            if "states" in keys:
                batch_states.append(example["states"])

            if "points" in keys:
                batch_points.append(example["points"])
                npoints_in_batch.append(example["npoints_in_batch"])

        input_id_lens = torch.LongTensor([len(input_ids) for input_ids in batch_input_ids])
        input_ids = pad_sequence(batch_input_ids, padding_side="right", padding_value=self.pad_token_id)
        attention_mask = input_ids != self.pad_token_id
        data_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "input_id_lens": input_id_lens,
        }

        if is_labels_provided:
            labels = pad_sequence(batch_label_ids, padding_side="right", padding_value=IGNORE_INDEX)
            data_dict["labels"] = labels

        if len(batch_pixel_values) > 0:
            data_dict["pixel_values"] = torch.cat(batch_pixel_values, dim=0)
            data_dict["image_grid_thw"] = torch.cat(batch_image_thw, dim=0)

        if len(batch_pixel_video_values) > 0:
            data_dict["pixel_values_videos"] = torch.cat(batch_pixel_video_values, dim=0)
            data_dict["video_grid_thw"] = torch.cat(batch_video_thw, dim=0)

        if len(batch_second_per_grid_ts) > 0:
            data_dict["second_per_grid_ts"] = batch_second_per_grid_ts

        if len(batch_actions) > 0:
            data_dict["actions"] = torch.cat(batch_actions, dim=0)
            data_dict["action_is_pad"] = torch.cat(batch_action_is_pad, dim=0)

        if len(batch_states) > 0:
            data_dict["states"] = torch.cat(batch_states, dim=0)

        if len(batch_points) > 0:
            data_dict["points"] = torch.cat(batch_points, dim=0)
            data_dict["npoints_in_batch"] = torch.LongTensor(npoints_in_batch)

        return data_dict
