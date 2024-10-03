import torch
from typing import List, Dict


def fed_avg(
    global_state_dict: Dict,
    local_state_dicts: List[Dict],
    local_weights: List[float],
) -> Dict:
    if len(local_weights) == 0:
        local_weights = [1 / len(local_state_dicts)] * len(local_state_dicts)

    for key, value in global_state_dict.items():
        temp_value = torch.zeros_like(value)
        for index, state_dict in enumerate(local_state_dicts):
            if len(state_dict) == 0:
                continue
            if "num_batches_tracked" in key:
                temp_value = torch.max(temp_value, state_dict[key])
            else:
                temp_value += state_dict[key] * local_weights[index]
        global_state_dict[key] = temp_value
    return global_state_dict


