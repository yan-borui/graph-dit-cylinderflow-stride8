import torch
from typing import Dict, Tuple

from ..graph import Graph


def scale_edges(e: torch.Tensor, r: float) -> torch.Tensor:
    r"""Linearly scale the graph edges by a factor of $1/(2r)$."""
    return e / (2 * r)


class ScaleEdgeAttr:
    r"""Transformation class for linearly scaling the graph edges by a factor of $1/(2r)$.

    Args:
        r (float): The scale factor is given by $1/(2r)$.

    Methods:
        __call__(Graph) -> Graph: Scale the graph edges.
    """

    def __init__(self, r: float) -> None:
        self.r = r

    def __call__(self, graph: Graph) -> Graph:
        graph.edge_attr /= 2 * self.r
        return graph


class ScaleAttr:
    r"""Transformation class for linearly scaling the graph attributes. Given $a$ and $b$, the scaling is given by
    $$x \leftarrow \frac{2(x-a)}{b-a} - 1.$$

    Args:
        attr (str): Attribute to scale.
        vmin (float): Minimum value of the attribute.
        vmax (float): Maximum value of the attribute.
        idx (int): Index of the attribute to scale.

    Methods:
        __call__(Graph) -> Graph: Scale the attribute of the graph.
    """

    def __init__(
        self,
        attr: str,
        vmin: float,
        vmax: float,
        idx: int = None,
    ) -> None:
        self.attr = attr
        self.idx = idx
        self.vmin = vmin
        self.vmax = vmax

    def __call__(self, graph: Graph) -> Graph:
        assert hasattr(graph, self.attr), f"Graph has no attribute {self.attr}"
        attr = getattr(graph, self.attr).clone()
        if self.idx is None:
            attr = 2 * (attr - self.vmin) / (self.vmax - self.vmin) - 1.0
        else:
            attr[:, self.idx] = (
                2 * (attr[:, self.idx] - self.vmin) / (self.vmax - self.vmin) - 1.0
            )
        setattr(graph, self.attr, attr)
        return graph


class ScaleNs:
    r"""Transformation class for linearly scaling the input and target fields for $u$, $v$ and $p$ and the $Re$ field.
    Given $a$ and $b$, the scaling is given by
    $$u \leftarrow \frac{u-c}{d},$$
    where
    $$c = \frac{a+b}{2},$$
    $$d = \frac{b-a}{2}.$$

    Args:
        scaling (dict): Dictionary of scaling factors $a$,$b$) for each field. The dictionary can contain the keys 'u', 'v', 'p' and 'Re'.
        format (str): Format of the field. Can be 'uvp' or 'uv'.

    Methods:
        __call__(Graph) -> Graph: Scale the input and target fields of the graph.
    """

    def __init__(self, scaling: Dict[str, Tuple[float, float]], format: str) -> None:
        # Check format
        assert format in ["uvp", "uv"], (
            f"Unknown format {format}, must be 'uvp' or 'uv'"
        )
        # Compute scaling factors
        self.u = (
            (
                0.5 * (scaling["u"][0] + scaling["u"][1]),
                0.5 * abs(scaling["u"][1] - scaling["u"][0]),
            )
            if "u" in scaling
            else None
        )
        self.v = (
            (
                0.5 * (scaling["v"][0] + scaling["v"][1]),
                0.5 * abs(scaling["v"][1] - scaling["v"][0]),
            )
            if "v" in scaling
            else None
        )
        self.Re = (
            (
                0.5 * (scaling["Re"][0] + scaling["Re"][1]),
                0.5 * abs(scaling["Re"][1] - scaling["Re"][0]),
            )
            if "Re" in scaling
            else None
        )
        if format == "uvp":
            self.p = (
                (
                    0.5 * (scaling["p"][0] + scaling["p"][1]),
                    0.5 * abs(scaling["p"][1] - scaling["p"][0]),
                )
                if "p" in scaling
                else None
            )
            self.num_fields = 3
        else:
            self.num_fields = 2

    def __call__(self, graph: Graph) -> Graph:
        if self.u is not None:
            if hasattr(graph, "field"):
                graph.field[:, 0 :: self.num_fields] = (
                    graph.field[:, 0 :: self.num_fields] - self.u[0]
                ) / self.u[1]
            if hasattr(graph, "target"):
                graph.target[:, 0 :: self.num_fields] = (
                    graph.target[:, 0 :: self.num_fields] - self.u[0]
                ) / self.u[1]
        if self.v is not None:
            if hasattr(graph, "field"):
                graph.field[:, 1 :: self.num_fields] = (
                    graph.field[:, 1 :: self.num_fields] - self.v[0]
                ) / self.v[1]
            if hasattr(graph, "target"):
                graph.target[:, 1 :: self.num_fields] = (
                    graph.target[:, 1 :: self.num_fields] - self.v[0]
                ) / self.v[1]
        if self.num_fields == 3 and self.p is not None:
            if hasattr(graph, "field"):
                graph.field[:, 2 :: self.num_fields] = (
                    graph.field[:, 2 :: self.num_fields] - self.p[0]
                ) / self.p[1]
            if hasattr(graph, "target"):
                graph.target[:, 2 :: self.num_fields] = (
                    graph.target[:, 2 :: self.num_fields] - self.p[0]
                ) / self.p[1]
        if self.Re is not None:
            if hasattr(graph, "glob"):
                graph.glob = (graph.glob - self.Re[0]) / self.Re[1]
        return graph
