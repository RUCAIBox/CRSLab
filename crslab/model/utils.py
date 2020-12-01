import torch


def edge_to_pyg_format(edge, device, type='RGCN'):
    if type == 'RGCN':
        edge_sets = torch.as_tensor(edge, dtype=torch.long, device=device)
        edge_idx = edge_sets[:, :2].t()
        edge_type = edge_sets[:, 2]
        return edge_idx, edge_type
    elif type == 'GCN':
        edge_set = [[co[0] for co in edge], [co[1] for co in edge]]
        return torch.as_tensor(edge_set, dtype=torch.long, device=device)
    else:
        raise NotImplementedError('type {} has not been implemented', type)
