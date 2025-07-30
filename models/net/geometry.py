import torch
from torch_scatter import scatter_add


def get_distance(pos, edge_index):
    return (pos[edge_index[0]] - pos[edge_index[1]]).norm(dim=-1)


def eq_transform(score_d, pos, edge_index, edge_length):
    N = pos.size(0)
    # score_d = (score_d - score_d.mean()) / (score_d.std() + 1e-8)
    # print(f"pos:{pos},len(pos):{len(pos)}")
    # print(f"pos[edge_index]:{pos[edge_index[0]]},pos[edge_index[1]]:{pos[edge_index[1]]}")
    dd_dr = (1. / (edge_length + 1e-8)) * (pos[edge_index[0]] - pos[edge_index[1]])  # (E, 3)
    # score_pos = scatter_add(dd_dr * score_d, edge_index[0], dim=0, dim_size=N) \
    #     + scatter_add(- dd_dr * score_d, edge_index[1], dim=0, dim_size=N) # (N, 3)
    score_pos = scatter_add(dd_dr * score_d, edge_index[0], dim=0, dim_size=N)
    # print(f"score_pos:{score_pos}")
    return score_pos
