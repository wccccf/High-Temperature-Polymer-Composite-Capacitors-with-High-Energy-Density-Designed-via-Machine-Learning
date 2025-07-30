import torch
import torch.nn.functional as F


class CountNodesPerGraph(object):

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, data):
        data.num_nodes_per_graph = torch.LongTensor([data.num_nodes])
        return data


class GetAdj(object):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, data):
        '''
        full connected edges
        '''

        n_particles = data.num_nodes
        coors = data.pos
        rows, cols = [], []
        for i in range(n_particles):
            for j in range(i + 1, n_particles):
                rows.append(i)
                cols.append(j)
                rows.append(j)
                cols.append(i)
        # print(n_particles)
        rows = torch.LongTensor(rows).unsqueeze(0)
        cols = torch.LongTensor(cols).unsqueeze(0)
        # print(rows.size())
        adj = torch.cat([rows, cols], dim=0)
        rel_coors = coors[adj[0]] - coors[adj[1]]
        rel_dist = (rel_coors ** 2).sum(dim=-1, keepdim=True)
        data.edge_index = adj
        data.edge_type = rel_dist.squeeze(1)
        return data


class AtomFeat(object):
    def __init__(self, atom_index) -> None:
        super().__init__()
        self.atom_index = atom_index

    def __call__(self, data):
        '''
        one-hot feature
        '''
        atom_type = data.atom_type.tolist()
        # print(data.atom_type)
        atom_type = torch.tensor([self.atom_index[i] for i in atom_type])
        # print(atom_type)
        data.atom_feat = F.one_hot(atom_type, num_classes=len(self.atom_index))

        data.atom_feat_full = torch.cat([data.atom_feat, data.atom_type.unsqueeze(1)], dim=1)

        return data