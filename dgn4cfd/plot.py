import torch
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from typing import Union, Optional, Tuple

try:
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk

    VTK_AVAILABLE = True
except:
    VTK_AVAILABLE = False


def triangulation(
    pos: torch.Tensor,
    bound: torch.Tensor = None,
    boundary_idx: Union[int, list[int]] = 4,
) -> tri.Triangulation:
    """Create a triangulation with a mask for the cells outside of the boundary if `bound` is provided.

    Args:
        pos (torch.Tensor): Node positions. Dim: (num_nodes, 2).
        bound (torch.Tensor, optional): Boundary mask. Dim: (num_nodes,). Defaults to None.
        boundary_idx (Union[int, List[int]], optional): Value of the boundary idx indicating the boundary. Defaults to 4.

    Returns:
        tri.Triangulation: Triangulation with mask for boundary.
    """
    pos = pos.cpu()
    # Create triangulation
    triang = tri.Triangulation(pos[:, 0], pos[:, 1])
    if bound is None:
        return triang
    else:
        bound = bound.cpu()
        # Get bound values for each triangle
        bound_on_triang_vertices = bound[triang.triangles]  # Dim: (num_triangles, 3)
        # Get triangles that are not on the boundary
        if isinstance(boundary_idx, int):  # If only one boundary index is given
            mask = (bound_on_triang_vertices == boundary_idx).all(dim=1)
        else:  # If multiple boundary indices are given
            mask = (bound_on_triang_vertices == boundary_idx[0]).all(
                dim=1
            )  # Initialize mask
            for idx in boundary_idx[1:]:  # Iterate over remaining boundary indices
                mask = mask | (bound_on_triang_vertices == idx).all(
                    dim=1
                )  # Update mask
        # Apply mask
        triang.set_mask(mask)
        return triang


def pos(
    pos: torch.Tensor,
    s: float = 0.1,
    filename: str = None,
    fontsize: int = 13,
    azim: float = None,
    dist: float = None,
    elev: float = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot node positions.

    Args:
        pos (torch.Tensor): Node positions. Dim: (num_nodes, 2) or (num_nodes, 3).
        s (float, optional): Marker size. Defaults to 0.1.
        file (Optional[str], optional): File name to save plot. Defaults to None.
        fontsize (Optional[int], optional): Font size. Defaults to 13.
        azim (float, optional): Azimuthal angle. Defaults to None. Only for 3D plots.
        dist (float, optional): Distance to plot. Defaults to None. Only for 3D plots.
        elev (float, optional): Elevation angle. Defaults to None. Only for 3D plots.

    Returns:
        None
    """
    pos = pos.to("cpu")
    fig = plt.figure()
    dim = pos.size(1)
    if dim == 2:
        ax = fig.add_subplot(111)
        plt.scatter(pos[:, 0], pos[:, 1], color="black", s=s)
        ax.set_aspect("equal")
        ax.set_xlabel("x", fontsize=fontsize)
        ax.set_ylabel("y", fontsize=fontsize)
    elif dim == 3:
        ax = fig.add_subplot(projection="3d")
        ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], s=s, color="k")
        ax.set_xlabel("x", fontsize=fontsize)
        ax.set_ylabel("y", fontsize=fontsize)
        ax.set_zlabel("z", fontsize=fontsize)
        if azim is not None:
            ax.view_init(azim=azim)
        if dist is not None:
            ax.dist = dist
        if elev is not None:
            ax.elev = elev
    if filename is not None:
        fig.savefig(filename)
    plt.show()
    return fig, ax


def pos_field(
    pos: torch.Tensor,
    u: torch.Tensor,
    s: float = 0.1,
    cmap: Optional[str] = "coolwarm",
    filename: Optional[str] = None,
    fontsize: Optional[int] = 13,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    azim: float = None,
    dist: float = None,
    elev: float = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot node positions and field values.

    Args:
        pos (torch.Tensor): Node positions. Dim: (num_nodes, 2) or (num_nodes, 3).
        u (torch.Tensor): Field values. Dim: (num_nodes,).
        s (float, optional): Marker size. Defaults to 0.1.
        cmap (Optional[str], optional): Colormap. Defaults to "coolwarm".
        file (Optional[str], optional): File name to save plot. Defaults to None.
        fontsize (Optional[int], optional): Font size. Defaults to 13.
        vmin (Optional[float], optional): Minimum value for colormap. Defaults to None.
        vmax (Optional[float], optional): Maximum value for colormap. Defaults to None.
        azim (float, optional): Azimuthal angle. Defaults to None. Only for 3D plots.
        dist (float, optional): Distance to plot. Defaults to None. Only for 3D plots.
        elev (float, optional): Elevation angle. Defaults to None. Only for 3D plots.

    Returns:

    """
    assert u.dim() == 1, "u must be a 1D tensor."  # Check dimension of u tensor is 1
    assert pos.size(0) == u.size(0), (
        "pos and u must have the same number of nodes."
    )  # Check that pos and u have the same number of nodes
    if vmin and vmax is not None:
        assert vmin < vmax, "vmin must be smaller than vmax."
    pos = pos.to("cpu")
    u = u.to("cpu")
    fig = plt.figure()
    dim = pos.size(1)
    if dim == 2:
        ax = fig.add_subplot(111)
        im = plt.scatter(
            pos[:, 0], pos[:, 1], c=u, cmap=cmap, s=s, vmin=vmin, vmax=vmax
        )
        ax.set_aspect("equal")
    elif dim == 3:
        ax = fig.add_subplot(projection="3d")
        im = ax.scatter(
            pos[:, 0],
            pos[:, 1],
            pos[:, 2],
            s=s,
            c=u,
            cmap="coolwarm",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_xlabel("x", fontsize=fontsize)
        ax.set_ylabel("y", fontsize=fontsize)
        ax.set_zlabel("z", fontsize=fontsize)
        if azim is not None:
            ax.view_init(azim=azim)
        if dist is not None:
            ax.dist = dist
        if elev is not None:
            ax.elev = elev
        # Remove background
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        # Remove grid
        ax.grid(False)
        # Remove gray panes
        ax.xaxis.pane.set_edgecolor("w")
    cax = fig.add_axes(
        [
            ax.get_position().x1 + 0.1,
            ax.get_position().y0,
            0.02,
            ax.get_position().height,
        ]
    )
    plt.colorbar(im, cax=cax)
    cax.yaxis.set_tick_params(labelsize=20)
    if filename:
        fig.savefig(filename)
    plt.show()
    return fig, ax


def field(
    pos: torch.Tensor,
    u: torch.Tensor,
    vmin: float = None,
    vmax: float = None,
    cmap: str = "coolwarm",
    filename: str = None,
    bound: torch.Tensor = None,
    boundary_idx: Union[int, list[int]] = 4,
    label: dict = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot field values.

    Args:
        pos (torch.Tensor): Node positions. Dim: (num_nodes, 2) or (num_nodes, 3).
        u (torch.Tensor): Field values. Dim: (num_nodes,).
        vmin (Optional[float], optional): Minimum value for colormap. Defaults to None.
        vmax (Optional[float], optional): Maximum value for colormap. Defaults to None.
        cmap (Optional[str], optional): Colormap. Defaults to "coolwarm".
        file (Optional[str], optional): File name to save plot. Defaults to None.
        bound (Optional[torch.Tensor], optional): Boundary mask. Dim: (num_nodes,). Defaults to None.
        boundary_idx (Optional[Union[int, List[int]]], optional): Boundary index. Defaults to None.
        label (Optional[dict], optional): Label for plot. Defaults to None.

    Returns:

    """
    # Check dimension of u tensor is 1
    assert u.dim() == 1, "u must be a 1D tensor."
    # Check that pos and u have the same number of nodes
    assert pos.size(0) == u.size(0), "pos and u must have the same number of nodes."
    # Check that vmin is smaller than vmax
    if vmin and vmax is not None:
        assert vmin < vmax, "vmin must be smaller than vmax."
    # Convert tensors to cpu
    pos = pos.to("cpu")
    u = u.to("cpu")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    triang = triangulation(pos, bound, boundary_idx=boundary_idx)
    ax.tripcolor(triang, u, vmin=vmin, vmax=vmax, cmap=cmap, shading="gouraud")
    ax.set_aspect("equal")
    ax.set_xticks([]), ax.set_yticks([])
    xmin, xmax = pos[:, 0].min().item(), pos[:, 0].max().item()
    ymin, ymax = pos[:, 1].min().item(), pos[:, 1].max().item()
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    # Add label
    if label is not None:
        ax.text(
            label["x"],
            label["y"],
            label["text"],
            fontsize=label["fontsize"],
            bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.2"),
            transform=plt.gca().transAxes,
        )
        ax.set_title(label["title"], fontsize=label["fontsize"] + 10)
    plt.show()
    if filename:
        fig.savefig(filename, bbox_inches="tight")
    return fig, ax


if VTK_AVAILABLE:

    def convert_to_vtk(
        pos: torch.Tensor,
        fields: torch.Tensor,
        componets: int = 1,
        filename: str = None,
        fieldname: str = "u",
    ) -> Union[vtk.vtkPolyData, list[vtk.vtkPolyData]]:
        """Convert node positions to vtkPolyData with field values.

        Args:
            pos (torch.Tensor): Node positions. Dim: (num_nodes, 2) or (num_nodes, 3).
            fields (torch.Tensor): Field values. Dim: (num_nodes, num_fields).
            dim (int, optional): Dimension of the field. Defaults to 1.
            filename (str, optional): File name. If provided, the vtkPolyData will be saved. Defaults to 'field'.
            fieldname (str, optional): Field name. Defaults to 'u'.

        Returns:
            Union[vtk.vtkPolyData, list[vtk.vtkPolyData]]: vtkPolyData with field values.
        """

        pos = pos.cpu()
        fields = fields.cpu()
        if fields.dim() == 1:
            fields = fields.unsqueeze(1)
        num_fields = fields.size(1) // componets
        # Create points
        points = vtk.vtkPoints()
        for point in pos:
            points.InsertNextPoint(point.tolist())
        polydata_list = []
        # Create scalars for field
        for t, field in enumerate(fields.split(componets, dim=1)):
            polydata = vtk.vtkPolyData()
            polydata.SetPoints(points)
            field = field.cpu().detach().numpy()
            if componets == 1:
                polydata.GetPointData().SetScalars(numpy_to_vtk(field))
            elif componets > 1:
                polydata.GetPointData().SetVectors(numpy_to_vtk(field))
            polydata_list.append(polydata)
            # Save polydata
            if filename is not None:
                if num_fields == 1:
                    filename_t = filename
                else:
                    if "." in filename:
                        filename_t = (
                            filename.split(".")[:-2]
                            + f"_{fieldname}_{t}"
                            + "."
                            + filename.split(".")[-1]
                        )
                    else:
                        filename_t = filename + f"_{fieldname}_{t}"
                writer = vtk.vtkPolyDataWriter()
                writer.SetFileName(filename_t)
                writer.SetInputData(polydata)
                if componets == 1:
                    writer.SetScalarsName(fieldname)
                elif componets > 1:
                    writer.SetVectorsName(fieldname)
                writer.Write()
        return polydata_list if num_fields > 1 else polydata_list[0]
