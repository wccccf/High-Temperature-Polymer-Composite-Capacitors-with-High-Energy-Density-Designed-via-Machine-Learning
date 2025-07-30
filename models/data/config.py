df62k_with_h = {
    'name': 'df62k',
    'atom_index': {1: 0, 6: 1, 7: 2, 8: 3, 9: 4, 14: 5, 15: 6, 16: 7, 17: 8, 35: 9},
    'atom_encoder': {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4, 'Si': 5, 'P': 6, 'S': 7, 'Cl': 8, 'Br': 9},
    'atom_decoder': ['H', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'Br'],
    'max_n_nodes': 80,
    'with_h': True}


def get_dataset_info(dataset_name, remove_h):
    return df62k_with_h