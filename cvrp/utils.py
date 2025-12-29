import torch
from torch_geometric.data import Data

CAPACITY = 50
DEMAND_LOW = 1
DEMAND_HIGH = 9
DEPOT_COOR = [0.5, 0.5]

def gen_instance(n, device):
    locations = torch.rand(size=(n, 2), device=device)
    demands = torch.randint(low=DEMAND_LOW, high=DEMAND_HIGH+1, size=(n,), device=device)
    depot = torch.tensor([DEPOT_COOR], device=device)
    all_locations = torch.cat((depot, locations), dim=0)
    all_demands = torch.cat((torch.zeros((1,), device=device), demands))
    distances = gen_distance_matrix(all_locations)
    return all_demands, distances # (n+1), (n+1, n+1)

def gen_distance_matrix(tsp_coordinates):
    n_nodes = len(tsp_coordinates)
    distances = torch.norm(tsp_coordinates[:, None] - tsp_coordinates, dim=2, p=2)
    distances[torch.arange(n_nodes), torch.arange(n_nodes)] = 1e-10 # note here
    return distances

def gen_pyg_data(demands, distances, device, k_nearest=None):
    """
    Build PyG Data object.
    
    Args:
        demands: (n,) tensor
        distances: (n, n) tensor  
        device: torch device
        k_nearest: if None, builds fully connected graph (n² edges)
                   if int, builds kNN graph matching FACO's nn_list (n*k edges)
    """
    n = demands.size(0)
    
    if k_nearest is None:
        # Fully connected graph (original behavior)
        nodes = torch.arange(n, device=device)
        u = nodes.repeat(n)
        v = torch.repeat_interleave(nodes, n)
        edge_index = torch.stack((u, v))
        edge_attr = distances.reshape(((n)**2, 1))
    else:
        # kNN graph matching FACO's nn_list structure
        k = min(k_nearest, n - 1)
        
        # For each node, find k nearest neighbors (excluding self)
        # This matches build_nearest_neighbor_lists in faco.py
        u_list = []
        v_list = []
        dist_list = []
        
        for i in range(n):
            # Get distances from node i, set self-distance to inf to exclude
            dists_i = distances[i].clone()
            dists_i[i] = float('inf')
            
            # Get k nearest neighbors
            _, nn_indices = torch.topk(dists_i, k, largest=False)
            
            # Add edges i -> each neighbor
            for j in nn_indices:
                u_list.append(i)
                v_list.append(j.item())
                dist_list.append(distances[i, j].item())
        
        edge_index = torch.tensor([u_list, v_list], dtype=torch.long, device=device)
        edge_attr = torch.tensor(dist_list, dtype=torch.float32, device=device).unsqueeze(1)
    
    x = demands
    pyg_data = Data(x=x.unsqueeze(1), edge_attr=edge_attr, edge_index=edge_index)
    return pyg_data

def load_test_dataset(problem_size, device):
    test_list = []
    dataset = torch.load(f'../data/cvrp/testDataset-{problem_size}.pt', map_location=device)
    for i in range(len(dataset)):
        test_list.append((dataset[i, 0, :], dataset[i, 1:, :]))
    return test_list

if __name__ == '__main__':
    import pathlib
    pathlib.Path('../data/cvrp').mkdir(parents=False, exist_ok=True) 
    torch.manual_seed(123456)
    for n in [20, 100, 500]:
        inst_list = []
        for _ in range(100):
            demands, distances = gen_instance(n, 'cpu')
            inst = torch.cat((demands.unsqueeze(0), distances), dim=0) # (n+2, n+1)
            inst_list.append(inst)
        testDataset = torch.stack(inst_list)
        torch.save(testDataset, f'../data/cvrp/testDataset-{n}.pt')
        