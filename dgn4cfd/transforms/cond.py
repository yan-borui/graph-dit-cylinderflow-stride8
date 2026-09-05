import torch

from ..graph import Graph


class EdgeCondFreeStreamProjection:
    def __init__(self, attr="edge_cond", replace=False) -> None:
        self.attr = attr
        self.replace = replace

    def __call__(self, graph: Graph) -> Graph:
        # Projection of U along the edge unit vector
        projection = graph.edge_attr[:, [0]] / graph.edge_attr.norm(
            dim=-1, keepdim=True
        )  # Shape: (num_edges, dim)
        # Set the graph attribute
        if (
            self.replace
            or not hasattr(graph, self.attr)
            or getattr(graph, self.attr) is None
        ):
            setattr(graph, self.attr, projection)
        else:
            setattr(
                graph,
                self.attr,
                torch.cat([getattr(graph, self.attr), projection], dim=-1),
            )
        return graph


class EdgeCondFreeStream:
    def __init__(self, attr="edge_cond", replace=False, normals="normal") -> None:
        self.attr = attr
        self.replace = replace
        self.normals = normals

    def __call__(self, graph: Graph) -> Graph:
        assert hasattr(graph, self.normals), (
            f"The normal attribute {self.normals} must be defined in the graph"
        )
        # Define the edge tangent vector
        edge_tangent_vector = graph.edge_attr / graph.edge_attr.norm(
            dim=-1, keepdim=True
        )  # Shape: (num_edges, dim)
        # Define the edge normal vector
        edge_normal_vector = getattr(graph, self.normals)[
            graph.edge_index[0]
        ]  # Shape: (num_edges, dim)
        # Define the outer product of the tangent and normal vectors
        edge_third_vector = torch.cross(edge_tangent_vector, edge_normal_vector, dim=-1)
        # Projection of U along the vectors of each edge
        u = torch.cat(
            [
                edge_tangent_vector[:, [0]],
                edge_normal_vector[:, [0]],
                edge_third_vector[:, [0]],
            ],
            dim=1,
        )
        # Set the graph attribute
        if (
            self.replace
            or not hasattr(graph, self.attr)
            or getattr(graph, self.attr) is None
        ):
            setattr(graph, self.attr, u)
        else:
            setattr(graph, self.attr, torch.cat([getattr(graph, self.attr), u], dim=-1))
        return graph
