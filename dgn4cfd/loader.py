from typing import List
import torch.utils.data
from torch_geometric.data import Data, Batch
from torchvision import transforms


class Collater(object):
    """A modified version of PyTorch Geometric's default collate function that corrects the indices in edge_index_{i} and idx{i}_to_idx{i+1} to account for all the nodes in the batch."""

    def __init__(
        self,
        transform: transforms.Compose = None,
    ):
        self.transform = transform

    def collate(
        self,
        batch: List[Data],
    ):
        elem = batch[0]
        # Corret the indices in edge_index_{i} and idx{i}_to_idx{i+1} to account for all the nodes in the batch
        if hasattr(elem, "edge_index_2"):
            assert hasattr(elem, "idx1_to_idx2"), "Missing idx1_to_idx2 attribute"

            num_nodes_1 = elem.num_nodes
            num_nodes_2 = elem.pos_2.size(0)
            num_edges_1 = elem.edge_index.size(1)
            for graph in batch[1:]:
                graph.edge_index_2 += num_nodes_2 - num_nodes_1
                graph.idx1_to_idx2 += num_nodes_2
                num_nodes_1 += graph.num_nodes
                num_nodes_2 += graph.pos_2.size(0)
                num_edges_1 += graph.edge_index.size(1)

            level = 3
            while hasattr(elem, f"edge_index_{level}"):
                assert hasattr(elem, f"idx{level - 1}_to_idx{level}"), (
                    f"Missing idx{level - 1}_to_idx{level} attribute"
                )

                num_nodes_1 = elem.num_nodes
                num_nodes_lr = getattr(elem, f"pos_{level}").size(0)
                num_edges_hr = getattr(elem, f"edge_index_{level - 1}").size(1)
                for graph in batch[1:]:
                    setattr(
                        graph,
                        f"edge_index_{level}",
                        getattr(graph, f"edge_index_{level}")
                        + (num_nodes_lr - num_nodes_1),
                    )
                    setattr(
                        graph,
                        f"idx{level - 1}_to_idx{level}",
                        getattr(graph, f"idx{level - 1}_to_idx{level}") + num_nodes_lr,
                    )
                    num_nodes_1 += graph.num_nodes
                    num_nodes_lr += getattr(graph, f"pos_{level}").size(0)
                    num_edges_hr += getattr(graph, f"edge_index_{level - 1}").size(1)

                level += 1

        # Create the batch
        batch = Batch.from_data_list(batch)
        # Apply transforms
        return self.transform(batch) if self.transform is not None else batch

    def __call__(self, batch):
        return self.collate(batch)


class DataLoader(torch.utils.data.DataLoader):
    """A modified version of PyTorch Geometric's DataLoader that uses the custom collate function."""

    def __init__(
        self,
        dataset: List[Data],
        batch_size: int = 1,
        shuffle: bool = False,
        transform: transforms.Compose = None,
        **kwargs,
    ):
        if "collate_fn" in kwargs:
            del kwargs["collate_fn"]
        collate_fn = Collater(transform)
        super().__init__(dataset, batch_size, shuffle, collate_fn=collate_fn, **kwargs)
