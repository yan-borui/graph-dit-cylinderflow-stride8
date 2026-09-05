from ..graph import Graph


class Identity:
    def __call__(self, graph: Graph) -> Graph:
        return graph
