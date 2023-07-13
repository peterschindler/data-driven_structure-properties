import itertools
import os
import math

import matplotlib.pyplot as plt
import numpy as np

from matplotlib import patches
from matplotlib.patches import FancyArrowPatch, Patch, Circle, Path

from monty.serialization import loadfn
from pymatgen.core import Structure
from pymatgen import vis
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.transformations.standard_transformations import RotationTransformation
from pymatgen.core.periodic_table import Element

colors = loadfn(os.path.join(os.path.dirname(vis.__file__),
                             "ElementColorSchemes.yaml"))
color_dict = {el: [j / 256.001 for j in colors["Jmol"][el]]
              for el in colors["Jmol"].keys()}


def reorient(structure, miller_index, rotate=0.):
    # Align miller_index direction to z-direction [0,0,1]
    struct = structure.copy()
    latt = struct.lattice
    recp = latt.reciprocal_lattice_crystallographic
    normal = recp.get_cartesian_coords(miller_index)
    normal /= np.linalg.norm(normal)
    z = [0, 0, 1]
    rot_axis = np.cross(normal, z)
    # Check if normal and z are linearly dependent
    if not np.isclose(rot_axis, [0, 0, 0]).all():
        angle = np.arccos(np.clip(np.dot(normal, z), -1.0, 1.0))
        struct = RotationTransformation(rot_axis, math.degrees(angle)).apply_transformation(struct)
    # Align other axis (longest) to x-axis
    lattm = struct.lattice.matrix
    basis_lengths_xy = [lattm[0][0]**2+lattm[0][1]**2, lattm[1][0]**2+lattm[1][1]**2, lattm[2][0]**2+lattm[2][1]**2]
    max_ind = basis_lengths_xy.index(max(basis_lengths_xy))
    max_basis = list(lattm[max_ind])
    max_basis[2] = 0
    max_basis /= np.linalg.norm(max_basis)
    angle2 = np.arccos(np.clip(np.dot(max_basis, [1, 0, 0]), -1.0, 1.0))
    struct = RotationTransformation(z, math.degrees(angle2)).apply_transformation(struct)
    # Check if correct sign of rotation was applied, if not: rotate twice the angle the other direction
    if abs(struct.lattice.matrix[max_ind][1]) > 1e-5:
        struct = RotationTransformation(z, -2*math.degrees(angle2)).apply_transformation(struct)
    if rotate:
        struct = RotationTransformation(z, rotate).apply_transformation(struct)
    return struct


def repeat_uc_edge_atoms(structure, tol=1e-7):
    # Repeats atoms that touch any of the unit cell edges or corners (i.e, have a 0 or 1 fractional coordinate)
    struct = structure.copy()
    frac_coords = struct.frac_coords.tolist()
    border_species = []
    border_coords = []
    for s in struct:
        zero_ind = []
        for ind, fc in enumerate(s.frac_coords):
            if fc < tol or fc > 1-tol:
                zero_ind.append(ind)
        if len(zero_ind) > 0:
            comb = [list(c) for c in itertools.product([0, 1], repeat=len(zero_ind))]
            for c in comb:
                c = [float(k) for k in c]
                if len(c) == 3:
                    if not any(li == c for li in frac_coords + border_coords):
                        border_coords.append(c)
                        border_species.append(s.specie)
                elif len(c) == 2:
                    new_coord = [sfc for sfc in s.frac_coords]
                    new_coord[zero_ind[0]] = c[0]
                    new_coord[zero_ind[1]] = c[1]
                    if not any(li == new_coord for li in frac_coords + border_coords):
                        border_coords.append(new_coord)
                        border_species.append(s.specie)
                else:
                    new_coord = [sfc for sfc in s.frac_coords]
                    new_coord[zero_ind[0]] = c[0]
                    if not any(li == new_coord for li in frac_coords + border_coords):
                        border_coords.append(new_coord)
                        border_species.append(s.specie)

    # Fractional coords 0 and 1 may often get replaced with one another (during conversion to cart_coords),
    # hence adding/subtracting a tol value in these cases
    final_coords = [[tol if f < tol else 1-tol if f > 1-tol else f for f in fc]
                    for fc in struct.frac_coords.tolist()+border_coords]
    return Structure(struct.lattice, struct.species+border_species, final_coords)


def display_structure(structure, ax, miller_index=None, rotate=0, repeat=None, transform_to_conventional=False,
                      repeat_unitcell_edge_atoms=True, draw_unit_cell=True, draw_frame=False,
                      draw_legend=True, legend_loc='best', legend_fontsize=14, padding=5.0, scale=0.8, decay=0.0):
    """
    Function that helps visualize the struct in a 2-D matplotlib plot
    Args:
        structure (struct): struct object to be visualized
        ax (axes): matplotlib axes with which to visualize
        miller_index (list): Viewing direction (normal to Miller plane)
        rotate (float): Rotate view around the viewing direction
                          (Miller index normal stays the same, just rotation within plane normal to Miller index)
        repeat (list): number of repeating unit cells in x,y,z to visualize
        transform_to_conventional (bool): Whether to transform input structure to conventional centering
        repeat_unitcell_edge_atoms (bool): Whether to repeat atoms that lie on the edges/corners of unit cell_length
                                          (makes the visualization look cut off more smoothly)
        draw_unit_cell (bool): flag indicating whether or not to draw cell boundaries
        draw_frame (bool): Whether to draw a frame around the plot (axis on/off)
        draw_legend (bool): Whether to draw a legend labeling the atom types
        legend_loc (string): Location of legend
        legend_fontsize (int): Fontsize of legend
        padding (float): Padding of the plot around the outermost atoms
        scale (float): radius scaling for sites
        decay (float): how the alpha-value decays along the z-axis
    """
    if miller_index is None:
        miller_index = [1, 0, 0]
    if repeat is None:
        repeat = [1, 1, 1]
    struct = structure.copy()
    if transform_to_conventional:
        struct = SpacegroupAnalyzer(struct).get_conventional_standard_structure()
    if repeat_unitcell_edge_atoms:
        struct = repeat_uc_edge_atoms(struct)
    struct = reorient(struct, miller_index, rotate=rotate)
    orig_cell = struct.lattice.matrix.copy()
    if repeat:
        struct.make_supercell(repeat)
    coords = np.array(sorted(struct.cart_coords, key=lambda x: x[2]))
    sites = sorted(struct.sites, key=lambda x: x.coords[2])
    alphas = 1 - decay * (np.max(coords[:, 2]) - coords[:, 2])
    alphas = alphas.clip(min=0)
    # Draw circles at sites and stack them accordingly
    for n, coord in enumerate(coords):
        r = sites[n].specie.atomic_radius * scale
        ax.add_patch(Circle(coord[:2], r, color='w', zorder=2 * n))
        color = color_dict[sites[n].species_string]
        ax.add_patch(Circle(coord[:2], r, facecolor=color, alpha=alphas[n], edgecolor='k', lw=1, zorder=2 * n + 1))
    # Draw unit cell
    if draw_unit_cell:
        a, b, c = orig_cell[0], orig_cell[1], orig_cell[2]
        n = np.array([0, 0, 1])
        # Draw basis vectors as arrows
        proj_a, proj_b, proj_c = a - np.dot(a, n) * n, b - np.dot(b, n) * n, c - np.dot(c, n) * n
        if (proj_a[0] ** 2 + proj_a[1] ** 2) ** 0.5 > 0.5:
            verts = [[0., 0.], proj_a[:2]]
            codes = [Path.MOVETO, Path.LINETO]
            path = Path(verts, codes)
            patch = FancyArrowPatch(path=path, arrowstyle="-|>,head_length=7,head_width=4", facecolor='black', lw=2,
                                    alpha=1, zorder=500)
            ax.add_patch(patch)
        if (proj_b[0] ** 2 + proj_b[1] ** 2) ** 0.5 > 0.5:
            verts = [[0., 0.], proj_b[:2]]
            codes = [Path.MOVETO, Path.LINETO]
            path = Path(verts, codes)
            patch = FancyArrowPatch(path=path, arrowstyle="-|>,head_length=7,head_width=4", facecolor='black', lw=2,
                                    alpha=1, zorder=500)
            ax.add_patch(patch)
        if (proj_c[0] ** 2 + proj_c[1] ** 2) ** 0.5 > 0.5:
            verts = [[0., 0.], proj_c[:2]]
            codes = [Path.MOVETO, Path.LINETO]
            path = Path(verts, codes)
            patch = FancyArrowPatch(path=path, arrowstyle="-|>,head_length=7,head_width=4", facecolor='black', lw=2,
                                    alpha=1, zorder=500)
            ax.add_patch(patch)

        # Draw opposing three unit cell boundaries
        abc = a + b + c - np.dot(a + b + c, n) * n
        ab = a + b - np.dot(a + b, n) * n
        ac = a + c - np.dot(a + c, n) * n
        bc = b + c - np.dot(b + c, n) * n
        verts_top = [abc[:2], ab[:2], abc[:2], ac[:2], abc[:2], bc[:2]]
        codes_top = [Path.MOVETO, Path.LINETO, Path.MOVETO, Path.LINETO, Path.MOVETO, Path.LINETO]
        path_top = Path(verts_top, codes_top)
        patch_top = patches.PathPatch(path_top, facecolor='none', lw=2, alpha=0.5, zorder=500)
        ax.add_patch(patch_top)

        # Draw remaining surrounding unit cell boundaries
        verts_surround = [proj_a[:2], ac[:2], proj_c[:2], bc[:2], proj_b[:2], ab[:2], proj_a[:2]]
        codes_surround = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.LINETO]
        path_surround = Path(verts_surround, codes_surround)
        patch_surround = patches.PathPatch(path_surround, facecolor='none', lw=2, alpha=0.5, zorder=500)
        ax.add_patch(patch_surround)

    # Legend
    if draw_legend:
        unique_sites = list({s.species_string for s in sites})
        unique_colors = [color_dict[site] for site in unique_sites]
        handles = [Patch(facecolor=unique_colors[i], label=str(unique_sites[i]), linewidth=1,
                         edgecolor='black') for i in range(len(unique_sites))]
        ax.legend(handles=handles, frameon=False, loc=legend_loc, fontsize=legend_fontsize)

    ax.set_aspect("equal")
    x_lim = [min(coords[:, 0])-padding, max(coords[:, 0])+padding]
    y_lim = [min(coords[:, 1])-padding, max(coords[:, 1])+padding]
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    if not draw_frame:
        ax.axis('off')
    ax.set_xticks([])
    ax.set_yticks([])
    return ax


def plot_periodic_table_heatmap(elemental_data, cbar_label="", cbar_label_size=14, show_plot=False, cmap="YlOrRd",
                                alpha=0.65, cmap_range=None, blank_color="grey", value_format=None, max_row=9):
    """
    A static method that generates a heat map overlayed on a periodic table.
    Args:
         elemental_data (dict): A dictionary with the element as a key and a
            value assigned to it, e.g. surface energy and frequency, etc.
            Elements missing in the elemental_data will be grey by default
            in the final table elemental_data={"Fe": 4.2, "O": 5.0}.
         cbar_label (string): Label of the colorbar. Default is "".
         cbar_label_size (float): Font size for the colorbar label. Default is 14.
         cmap_range (tuple): Minimum and maximum value of the colormap scale.
            If None, the colormap will autotmatically scale to the range of the
            data.
         alpha: Opacity
         show_plot (bool): Whether to show the heatmap. Default is False.
         value_format (str): Formatting string to show values. If None, no value
            is shown. Example: "%.4f" shows float to four decimals.
         cmap (string): Color scheme of the heatmap. Default is 'YlOrRd'.
            Refer to the matplotlib documentation for other options.
         blank_color (string): Color assigned for the missing elements in
            elemental_data. Default is "grey".
         max_row (integer): Maximum number of rows of the periodic table to be
            shown. Default is 9, which means the periodic table heat map covers
            the first 9 rows of elements.
    """
    # Code adopted from pymatgen and changed design/esthetics

    # Convert primitive_elemental data in the form of numpy array for plotting.
    if cmap_range is not None:
        max_val = cmap_range[1]
        min_val = cmap_range[0]
    else:
        max_val = max(elemental_data.values())
        min_val = min(elemental_data.values())

    max_row = min(max_row, 9)
    if max_row > 7:
        max_row += 1
    if max_row <= 0:
        raise ValueError("The input argument 'max_row' must be positive!")

    value_table = np.empty((max_row, 18)) * np.nan
    blank_value = min_val - 0.01

    for el in Element:
        value = elemental_data.get(el.symbol, blank_value)
        if 57 <= el.Z <= 71:
            plot_row = 8
            plot_group = (el.Z - 54) % 32
        elif 89 <= el.Z <= 103:
            plot_row = 9
            plot_group = (el.Z - 54) % 32
        else:
            plot_row = el.row
            plot_group = el.group
        if plot_row > max_row:
            continue
        value_table[plot_row - 1, plot_group - 1] = value

    # Initialize the plt object
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    plt.gcf().set_size_inches(12, 6)

    # Enable alpha for colormap
    from matplotlib.colors import ListedColormap
    my_cmap_rgb = plt.get_cmap(cmap)(np.arange(256))
    for i in range(3):  # Do not include the last column!
        my_cmap_rgb[:, i] = (1 - alpha) + alpha * my_cmap_rgb[:, i]
    my_cmap = ListedColormap(my_cmap_rgb, name='my_cmap')

    # We set nan type values to masked values (ie blank spaces)
    data_mask = np.ma.masked_invalid(value_table.tolist())
    heatmap = ax.pcolor(data_mask, cmap=my_cmap, edgecolors='w', linewidths=3,
                        vmin=min_val - 0.001, vmax=max_val + 0.001)
    cbar = fig.colorbar(heatmap, aspect=40)

    cbar.outline.set_visible(False)
    cbar.set_alpha(1)
    # Grey out missing elements in input data
    cbar.cmap.set_under(blank_color, alpha=alpha)

    # Set the colorbar label and tick marks
    cbar.set_label(cbar_label, rotation=270, labelpad=25, size=cbar_label_size)
    cbar.ax.tick_params(labelsize=cbar_label_size)

    # Refine and make the table look nice
    ax.axis('off')
    ax.invert_yaxis()

    # Label each block with corresponding element and value
    for i, row in enumerate(value_table):
        for j, el in enumerate(row):
            if not np.isnan(el):
                symbol = Element.from_row_and_group(i + 1, j + 1).symbol
                z = Element.from_row_and_group(i + 1, j + 1).Z
                # ax.axhline(y=i+0.7, linewidth=1, color='white')
                plt.text(j + 0.1, i + 0.2, str(z),
                         horizontalalignment='left',
                         verticalalignment='center', fontsize=9)
                plt.text(j + 0.1, i + 0.5, symbol,
                         horizontalalignment='left',
                         verticalalignment='center', fontsize=14, weight="bold")
                if el != blank_value and value_format is not None:
                    plt.text(j + 0.5, i + 0.85, value_format % el,
                             horizontalalignment='center',
                             verticalalignment='center', fontsize=10,
                             bbox=dict(boxstyle='square,pad=0.1', facecolor='white', alpha=1, linewidth=0))

    plt.tight_layout()

    if show_plot:
        plt.show()

    return plt


if __name__ == '__main__':
    import pandas as pd
    mpids = pd.read_csv('data.csv', index_col=0)

    crys_sys = ['cubic', 'hexagonal', 'trigonal', 'tetragonal']

    elemental_data_all = {}
    elemental_data_cs = {}

    for ind in mpids.index:
        elements = mpids.loc[ind, 'chemsys'].split('-')
        for el in elements:
            if el in elemental_data_all:
                elemental_data_all[el] += 1
            else:
                elemental_data_all[el] = 1
        if mpids.loc[ind, 'symmetry.crystal_system'] in crys_sys:
            for el in elements:
                if el in elemental_data_cs:
                    elemental_data_cs[el] += 1
                else:
                    elemental_data_cs[el] = 1
    elemental_data_ratio = {x: 100 * elemental_data_cs[x] / elemental_data_all[x]
                            for x in elemental_data_all if x in elemental_data_cs}
    # print(elemental_data_ratio.keys())
    plt = plot_periodic_table_heatmap(elemental_data_ratio, cbar_label="Ratio (%)", cbar_label_size=16,
                                      show_plot=False, cmap="YlOrRd", cmap_range=(20, 70), value_format='%d')
    plt.savefig('test.png', dpi=300)
