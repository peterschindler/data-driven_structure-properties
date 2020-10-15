import itertools
import os
import math

import numpy as np
#from scipy.spatial import Delaunay

from matplotlib import patches
from matplotlib.patches import FancyArrowPatch, Patch, Circle, Path
import matplotlib.pyplot as plt

from monty.serialization import loadfn

from pymatgen import Structure, Lattice, vis
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.transformations.standard_transformations import RotationTransformation

colors = loadfn(os.path.join(os.path.dirname(vis.__file__),
                             "ElementColorSchemes.yaml"))
color_dict = {el: [j / 256.001 for j in colors["Jmol"][el]]
              for el in colors["Jmol"].keys()}

def reorient(structure, miller_index, rotate=0):
    #Align miller_index direction to z-direction [0,0,1]
    struct = structure.copy()
    latt = struct.lattice
    recp = latt.reciprocal_lattice_crystallographic
    normal = recp.get_cartesian_coords(miller_index)
    normal /= np.linalg.norm(normal)
    z = [0, 0, 1]
    rot_axis = np.cross(normal, z)
    angle = np.arccos(np.clip(np.dot(normal, z), -1.0, 1.0))
    struct = RotationTransformation(rot_axis, math.degrees(angle)).apply_transformation(struct)
    #Align other axis (longest) to x-axis
    lattm = struct.lattice.matrix
    basis_lengths_xy = [lattm[0][0]**2+lattm[0][1]**2, lattm[1][0]**2+lattm[1][1]**2, lattm[2][0]**2+lattm[2][1]**2]
    max_ind = basis_lengths_xy.index(max(basis_lengths_xy))
    max_basis = list(lattm[max_ind])
    max_basis[2] = 0
    max_basis /= np.linalg.norm(max_basis)
    angle2 = np.arccos(np.clip(np.dot(max_basis, [1, 0, 0]), -1.0, 1.0))
    struct = RotationTransformation(z, math.degrees(angle2)).apply_transformation(struct)
    #Check if correct sign of rotation was applied, if not: rotate twice the angle the other direction
    if abs(struct.lattice.matrix[max_ind][1]) > 1e-5:
        struct = RotationTransformation(z, -2*math.degrees(angle2)).apply_transformation(struct)
    if rotate:
        struct = RotationTransformation(z, rotate).apply_transformation(struct)
    return struct

def repeat_uc_edge_atoms(structure, tol=1e-7):
    #Repeats atoms that touch any of the unit cell edges or corners (i.e, have a 0 or 1 fractional coordinate)
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
            comb = [list(c) for c in itertools.product([0,1], repeat=len(zero_ind))]
            for c in comb:
                c = [float(k) for k in c]
                if len(c) == 3:
                    if not any(list == c for list in frac_coords + border_coords):
                        border_coords.append(c)
                        border_species.append(s.specie)
                elif len(c) == 2:
                    new_coord = [sfc for sfc in s.frac_coords]
                    new_coord[zero_ind[0]] = c[0]
                    new_coord[zero_ind[1]] = c[1]
                    if not any(list == new_coord for list in frac_coords + border_coords):
                        border_coords.append(new_coord)
                        border_species.append(s.specie)
                else:
                    new_coord = [sfc for sfc in s.frac_coords]
                    new_coord[zero_ind[0]] = c[0]
                    if not any(list == new_coord for list in frac_coords + border_coords):
                        border_coords.append(new_coord)
                        border_species.append(s.specie)

    #Pymatgen will replace some 0 fractional coords with 1, hence I am adding a small tolerance value to all 0 fractional coords
    final_coords = [[tol if f < tol else f for f in fc] for fc in struct.frac_coords.tolist()+border_coords]
    return Structure(struct.lattice, struct.species+border_species, final_coords)


def display_structure(structure, ax, miller_index=[1,1,0], rotate=0, repeat=[1,1,1], transform_to_conventional = True,
                      repeat_unitcell_edge_atoms=True, draw_unit_cell=True, draw_legend=True, legend_loc='best', padding=5.0, scale=0.8, decay=0.0):
    """
    Function that helps visualize the struct in a 2-D plot, for
    convenient viewing of output of AdsorbateSiteFinder.
    Args:
        struct (struct): struct object to be visualized
        ax (axes): matplotlib axes with which to visualize
        miller_index (list): Viewing direction (normal to Miller plane)
        rotate (float): Rotate view around the viewing direction
                          (Miller index normal stays the same, just rotation in plane normal to Miller index)
        repeat (list): number of repeating unit cells in x,y,z to visualize
        transform_to_conventional (bool): Whether to transform input structure to conventional centering
        repeat_unitcell_edge_atoms (bool): Whether to repeat atoms that lie on the edges/corners of unit cell_length
                                          (makes the display look cut off more smoothly)
        draw_unit_cell (bool): flag indicating whether or not to draw cell
        draw_legend (bool): Whether to draw a legend labeling the atom types
        legend_loc (string): Location of legend
        padding (float): Padding of the plot around the outermost atoms
        scale (float): radius scaling for sites
        decay (float): how the alpha-value decays along the z-axis
    """
    struct = structure.copy()
    if transform_to_conventional:
        struct = SpacegroupAnalyzer(struct).get_conventional_standard_structure()
    if repeat_unitcell_edge_atoms:
        struct = repeat_uc_edge_atoms(struct)
    struct = reorient(struct, miller_index, rotate=rotate)
    #struct.to(filename='test_reoirented.cif')
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
        ax.add_patch(Circle(coord[:2], r, facecolor=color, alpha=alphas[n],
                                    edgecolor='k', lw=1, zorder=2 * n + 1))
    # Draw unit cell
    if draw_unit_cell:
        a, b, c = orig_cell[0], orig_cell[1], orig_cell[2]
        n = np.array([0, 0, 1])
        #Draw basis vectors as arrows
        proj_a, proj_b, proj_c = a - np.dot(a, n)*n, b - np.dot(b, n)*n, c - np.dot(c, n)*n
        if (proj_a[0]**2+proj_a[1]**2)**0.5 > 0.5:
            verts = [[0., 0.], proj_a[:2]]
            codes = [Path.MOVETO, Path.LINETO]
            path = Path(verts, codes)
            patch = FancyArrowPatch(path=path, arrowstyle="-|>,head_length=7,head_width=4", facecolor='black', lw=2,
                                      alpha=1, zorder=500)#, zorder=2 * n + 2)
            ax.add_patch(patch)
        if (proj_b[0]**2+proj_b[1]**2)**0.5 > 0.5:
            verts = [[0., 0.], proj_b[:2]]
            codes = [Path.MOVETO, Path.LINETO]
            path = Path(verts, codes)
            patch = FancyArrowPatch(path=path, arrowstyle="-|>,head_length=7,head_width=4", facecolor='black', lw=2,
                                      alpha=1, zorder=500)#, zorder=2 * n + 2)
            ax.add_patch(patch)
        if (proj_c[0]**2+proj_c[1]**2)**0.5 > 0.5:
            verts = [[0., 0.], proj_c[:2]]
            codes = [Path.MOVETO, Path.LINETO]
            path = Path(verts, codes)
            patch = FancyArrowPatch(path=path, arrowstyle="-|>,head_length=7,head_width=4", facecolor='black', lw=2,
                                      alpha=1, zorder=500)#, zorder=2 * n + 2)
            ax.add_patch(patch)

        #Draw opposing three unit cell boundaries
        abc = a + b + c - np.dot(a + b + c, n)*n
        ab = a + b - np.dot(a + b, n)*n
        ac = a + c - np.dot(a + c, n)*n
        bc = b + c - np.dot(b + c, n)*n
        verts_top = [abc[:2], ab[:2], abc[:2], ac[:2], abc[:2], bc[:2]]
        codes_top = [Path.MOVETO, Path.LINETO, Path.MOVETO, Path.LINETO, Path.MOVETO, Path.LINETO]
        path_top = Path(verts_top, codes_top)
        patch_top = patches.PathPatch(path_top, facecolor='none', lw=2, alpha=0.5, zorder=500)
        ax.add_patch(patch_top)

        #Draw remaining surrounding unit cell boundaries
        verts_surround = [proj_a[:2], ac[:2], proj_c[:2], bc[:2], proj_b[:2], ab[:2], proj_a[:2]]
        codes_surround = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.LINETO]
        path_surround = Path(verts_surround, codes_surround)
        patch_surround = patches.PathPatch(path_surround, facecolor='none', lw=2, alpha=0.5, zorder=500)
        ax.add_patch(patch_surround)

    #Legend
    if draw_legend:
        unique_sites = list({s.species_string for s in sites})
        unique_colors = [color_dict[site] for site in unique_sites]
        handles = [Patch(facecolor=unique_colors[i], label=str(unique_sites[i]), linewidth=1, edgecolor='black') for i in range(len(unique_sites))]
        ax.legend(handles=handles, frameon=False, loc=legend_loc)

    ax.set_aspect("equal")
    x_lim = [min(coords[:,0])-padding, max(coords[:,0])+padding]
    y_lim = [min(coords[:,1])-padding, max(coords[:,1])+padding]
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    #ax.axis('off')
    ax.set_xticks([])
    ax.set_yticks([])
    return ax

coords = [[0, 0, 0], [0.75,0.5,0.75]]
lattice = Lattice.from_parameters(a=3.84, b=3.84, c=3.84,
                                    alpha=120, beta=90, gamma=60)
struct = Structure(lattice, ["Zn", "S"], coords)
struct.to(filename='test.cif')

from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
conv = SpacegroupAnalyzer(struct).get_conventional_standard_structure()

#coords = [[0, 0, 0], [0.75,0.5,0.75], [1, 1, 0.5], [0, 0, 0.25]]
#lattice = Lattice.from_parameters(a=3.84, b=3.84, c=3.84, alpha=120, beta=90, gamma=60)
#test = Structure(lattice, ["Zn", "S", "O", "N"], coords)
#repeat_uc_edge_atoms(test)

fig, ax = plt.subplots(tight_layout=True)
display_structure(struct, ax, miller_index=[1,0,0], scale=0.8, repeat=[2,2,2], draw_unit_cell=True, decay=0.0, transform_to_conventional=True, rotate=90)
plt.savefig('test.png', dpi = 300)
