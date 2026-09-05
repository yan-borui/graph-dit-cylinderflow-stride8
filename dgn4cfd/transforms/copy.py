from .. import Graph


class Copy:
    """Transformation class to copy a graph attribute to another attribute."""

    def __init__(
        self,
        src: str,
        dst: str,
    ) -> None:
        self.src = src
        self.dst = dst

    def __call__(
        self,
        graph: Graph,
    ) -> Graph:
        assert hasattr(graph, self.src), f"Attribute {self.src} not found in the graph"
        setattr(graph, self.dst, getattr(graph, self.src).clone())
        return graph
