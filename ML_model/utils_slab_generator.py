#pymatgen-2020.4.29
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
from pymatgen.core.surface import SlabGenerator, hkl_transformation
from pymatgen.core.sites import PeriodicSite
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.util.coord import in_coord_list
from pymatgen.transformations.standard_transformations import RotationTransformation
from pymatgen.core.operations import SymmOp
from functools import reduce
import itertools
import numpy as np
import math

try:
    # New Py>=3.5 import
    from math import gcd
except ImportError:
    # Deprecated import from Py3.5 onwards.
    from fractions import gcd


def get_symmetrically_distinct_miller_indices(structure, max_index):
    """
    Returns all symmetrically distinct indices below a certain max-index for
    a given structure. Analysis is based on the symmetry of the reciprocal
    lattice of the structure.
    Args:
        structure (Structure): input structure.
        max_index (int): The maximum index. For example, a max_index of 1
            means that (100), (110), and (111) are returned for the cubic
            structure. All other indices are equivalent to one of these.
    """

    r = list(range(-max_index, max_index + 1))
    r.reverse()

    # First we get a list of all hkls for conventional (including equivalent)
    conv_hkl_list = [miller for miller in itertools.product(r, r, r) if any([i != 0 for i in miller])]

    sg = SpacegroupAnalyzer(structure)
    # Get distinct hkl planes from the rhombohedral setting if trigonal
    if sg.get_crystal_system() == "trigonal":
        transf = sg.get_conventional_to_primitive_transformation_matrix()
        miller_list = [hkl_transformation(transf, hkl) for hkl in conv_hkl_list]
        prim_structure = SpacegroupAnalyzer(structure).get_primitive_standard_structure()
        symm_ops = prim_structure.lattice.get_recp_symmetry_operation()
    else:
        miller_list = conv_hkl_list
        symm_ops = structure.lattice.get_recp_symmetry_operation()

    unique_millers, unique_millers_conv = [], []

    def is_already_analyzed(miller_index):
        for op in symm_ops:
            if in_coord_list(unique_millers, op.operate(miller_index)):
                return True
        return False

    for i, miller in enumerate(miller_list):
        d = abs(reduce(gcd, miller))
        miller = tuple([int(i / d) for i in miller])
        if not is_already_analyzed(miller):
            if sg.get_crystal_system() == "trigonal":
                # Now we find the distinct primitive hkls using
                # the primitive symmetry operations and their
                # corresponding hkls in the conventional setting
                unique_millers.append(miller)
                d = abs(reduce(gcd, conv_hkl_list[i]))
                cmiller = tuple([int(i / d) for i in conv_hkl_list[i]])
                unique_millers_conv.append(cmiller)
            else:
                unique_millers.append(miller)
                unique_millers_conv.append(miller)
    return unique_millers_conv


def get_sites_id_slab(structure, index, cutoff = 5):
    #for a structure, determines how many unique elements and/or local sites there are based on neighbor_list
    #use "neighbor_list" as matrix that identifies site and then looks for unique ones.
    #Neighbor list is rounded to 3 significant figures
    #should work the same as get_symmetric_sites, ideally
    #id is only given by atoms "below" the surface atom

    id = []
    neighbor_list = structure.get_neighbors(site = structure[index], r = cutoff)
    for j in range(len(neighbor_list)):
        if(neighbor_list[j][0].coords[2] > structure[index].coords[2] + 0.01): #add 0.01 tolerance in case of rounding
            atom_number = neighbor_list[j][0].specie.number
            dist = np.round(neighbor_list[j][1], decimals = 2)
            id.append(atom_number)
            id.append(round(dist,4))
    id = np.reshape(id, [int(len(id)/2),2])
    a = id[:,0]
    b = id[:,1]
    ind = np.lexsort((b,a))
    id_sorted = [(a[i],b[i]) for i in ind]
    return(id_sorted)


def get_unique_sites_slab(structure, cutoff = 5):
    #same as get_unique_sites, but doesn't include surface sites
    #use this to get number of unique terminations for slabs
    grouped_sites = [] #list containing the sites that have already been grouped
    num_groups = 0
    min_site = min(structure.cart_coords[:,2])
    max_site = max(structure.cart_coords[:,2])
    for i in range(len(structure)):
        flat_sites = [val for sublist in grouped_sites for val in sublist] #flattened version to let next "if" statement work
        if (i not in np.array(flat_sites) and structure.cart_coords[i,2] - min_site > cutoff and max_site - structure.cart_coords[i,2] > cutoff):
            num_groups = num_groups+1
            site_id = get_sites_id_slab(structure,i,cutoff)
            original_index_list = []

            for j in range(len(structure)):
                if(site_id == get_sites_id_slab(structure,j,cutoff) and structure.cart_coords[j,2] - min_site > cutoff and max_site - structure.cart_coords[j,2] > cutoff):
                    original_index_list.append(j)
            grouped_sites.append(original_index_list)
    return(grouped_sites)


def get_unique_terminations_v2(structure, tol = 0.01, cutoff = 10):
    #utilizes get_unique_sites_slab to determine how many unique terminations a slab might have
    #termination is considered unique if there is a unique number and/or combination of grouped sites

    surface_id = []
    grouped_sites = get_unique_sites_slab(structure, cutoff)
    #min_site=min(structure.cart_coords[:,2])
    #group first and then sort by z?
    max_site = max(structure.cart_coords[:,2])
    index_grouped = []
    for i in range(len(structure)):
        if structure.cart_coords[i,2] < max_site-cutoff and i not in index_grouped:
            slab_id = []
            slab_indices = np.where(np.abs(structure.cart_coords[:,2]-structure.cart_coords[i,2]) < tol)
            #print(slab_indices[0])
            for ind in slab_indices[0]:
                if ind not in index_grouped:
                    index_grouped.append(ind)
            for slab_site in slab_indices[0]:
                for group_number in range(len(grouped_sites)):
                    if(slab_site in grouped_sites[group_number]):
                        slab_id.append(group_number)
            if(len(slab_id)>0):
                slab_id.sort()
                surface_id.append(slab_id)
        #x=[[2,3,4], [3,4,5], [2,3,4]]
        #np.unique(x, axis=0)

    try:
        unique = np.unique(surface_id, axis=0)
    except TypeError:
        unique = np.unique(surface_id)
    except ValueError:
        unique = np.unique(surface_id)

    num_termination = len(unique)
    Letters = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
    #unique=np.unique(surface_id)

    for ind,term in enumerate(surface_id):
        for uind,uterm in enumerate(unique):
            try:
                uterm = uterm.tolist()
            except:
                pass
            try:
                term = term.tolist()
            except:
                pass
            if term == uterm:
                surface_id[ind] = Letters[uind]
    termorder = ''.join(surface_id)
    while termorder[0] != 'A':
        if len(termorder) > 1:
            termorder = termorder[1:]
        else:
            termorder = 'error'
            break
    while termorder[-1] != Letters[num_termination-1]:
        if len(termorder) > 1:
            termorder = termorder[:-1]
        else:
            termorder = 'error'
            break
    termorder = termorder.replace(''.join(Letters[:num_termination]),'')

    return(termorder, num_termination)


def mirror_sym_check(slab, tol = 0.01):
    '''Centers slab and checks if the bottom half is the same as the mirrored top half (with tolerance 'tol' in Angstrom)
    Input slab is required to have a unit cell with c-direction 90 degrees to a and b basis vectors.
    Output: [sym, error] where 'sym' is the mirror symmetry (True or False) of the slab,
    'error' contains a message in case unit cell is not as required, otherwise this returns an empty string
    '''
    if round(slab.lattice.alpha,1) != 90.0 or round(slab.lattice.beta,1) != 90.0:
        sym = False
        error = "Error. c-axis of lattice is not normal to a and b axis."
    else:
        #Center slab
        midpoint = (max(slab.frac_coords[:,2]) - min(slab.frac_coords[:,2])) / 2 + min(slab.frac_coords[:,2])
        slab.translate_sites(list(range(len(slab))), [0, 0, 0.5 - midpoint])
        sym = False
        error = ""
        ftol = slab.lattice.get_fractional_coords([tol,tol,tol])#convert tolerances to fractional tolerances
        nbottom = nfound = 0
        for site in slab:
            if site.c < 0.5 - ftol[2]:
                nbottom += 1
                found = False
                for usite in slab:
                    if usite.c > 0.5 + ftol[2] and site.specie == usite.specie:
                        if abs(site.c + usite.c - 1) < ftol[2] and abs(site.a - usite.a) < ftol[0] and abs(site.b - usite.b) < ftol[1]:
                            found = True
                            nfound += 1
                if not found:
                    break
        if nbottom == nfound:
            sym = True
    return sym, error


def slab_has_mirror_sym(slab, nterm = 1, tol = 0.01):
    '''This function tests if the input slab has a mirror symmetry in the c-direction (given tolerance 'tol' in Angstrom).
    Specify 'nterm' (number of unique terminations) to ensure enough subtractions from the top/bottom of the layer are considered.
    Unit cell can be of any shape, however make sure that the layer thickness in c-direction is thick enough
    (recommendation: SlabGenerator with min_slab_size=4, in_unit_planes=True)
    Remark: A bug in pymatgen/transformations/standard_transformations.py was fixed in version pymatgen-2020.4.29

    Output: [mirror_sym, error] where 'mirror_sym' is the mirror symmetry (True or False),
    'error' may contain an error message in case any step of the unit cell rotation didn't work, otherwise contains empty string

    Procedure:
    1. Rotate unit cell such that Cartesian coordinates of a and b basis vectors is of form (a_x,0,0) and (b_x,b_y,0), respectively.
    2. Create new c-direction orthogonal to a and b.
    3. Add atoms from slab in step 1 to new unit cell of step 2.
    4. Use function mirror_sym_check to check symmetry for the initial slab, the slab with the topmost, the slab with lowermost layer missing,
    and the slab with both topmost/lowermost layers missing. If nterm > 3, additionally np.floor(nterm/2) layers are subtracted from either side and their symmetry checked.
    '''
    error = ""
    if round(slab.lattice.alpha,1) == 90.0 and round(slab.lattice.beta,1) == 90.0:
        slab_straight = Structure(slab._lattice, slab.species_and_occu, slab.frac_coords)
    else:
        R = slab.lattice.matrix
        #print(R)
        #if a base vector not parallel to x-axis in caartesian coords, rotate the cell/structure such that it is
        if abs(R[0,1]) > 0.0001 or abs(R[0,2]) > 0.0001:
            x = [1,0,0]
            rot_axis = np.cross(R[0], x)
            angle = np.arccos(np.clip(np.dot(R[0] / np.linalg.norm(R[0]), x / np.linalg.norm(x)), -1.0, 1.0))
            slab = RotationTransformation(rot_axis, math.degrees(angle)).apply_transformation(slab)
            R = slab.lattice.matrix
            #In case the wrong sign of the rotation was applied, rotate back twice the angle:
            if abs(R[0,1]) > 0.0001 or abs(R[0,2]) > 0.0001:
                slab = RotationTransformation(rot_axis, -2*math.degrees(angle)).apply_transformation(slab)
                R = slab.lattice.matrix
            if abs(R[0,1]) > 0.0001 or abs(R[0,2]) > 0.0001:
                error = "Error. Could not rotate a-axis to be parallel to x-axis."
            #print(R)
        #if b vector not lying in cartesian x-y plane, rotate to make it so (i.e. z-component of b vector = 0)
        if abs(R[1,2]) > 0.0001 and not error:
            b_x_flat = [0,0,0]
            b_x_flat[1] = R[1,1]
            b_x_flat[2] = R[1,2]
            x = [1,0,0]
            y = [0,1,0]
            angle2 = np.arccos(np.clip(np.dot(b_x_flat / np.linalg.norm(b_x_flat), y / np.linalg.norm(y)), -1.0, 1.0))#angle between y-axis and b vector projected onto y-z-plane
            slab = RotationTransformation(x,math.degrees(angle2)).apply_transformation(slab)
            R = slab.lattice.matrix
            #In case the wrong sign of the rotation was applied, rotate back twice the angle:
            if abs(R[1,2]) > 0.0001:
                slab = RotationTransformation(x,-2*math.degrees(angle2)).apply_transformation(slab)
                R = slab.lattice.matrix
            if abs(R[1,2]) > 0.0001:
                error = "Error. Could not rotate b-vector to lie within x-y-plane."
            if R[1,1] < -0.0001:
                error = "Error. Vector b faces into negative y-direction (which could cause problems)."
        #Now create new c-direction that is orthogonal to the rotated a and b vectors
        if not error:
            N = np.array(R) #new lattice vectors
            N[2] = np.cross(R[0], R[1])
            N[2] = N[2] * np.dot(R[2],N[2]) / (np.linalg.norm(N[2])*np.linalg.norm(N[2]))
            latticeN = Lattice(N) #new lattice with c perpendicular to a,b
            #Add atoms from rotated unit cell to new unit cell with orthogonal c-direction
            slab_straight = Structure(latticeN, slab.species, slab.cart_coords, coords_are_cartesian=True)
    #Check mirror symmetry for straightened slab as well as slabs that are missing either (or both) topmost, lowermost atom-layers
    if not error:
        mirror_sym = False

        #Checks mirror symmetry of slab_straight without any layers removed
        slab_orig = Structure(slab_straight._lattice, slab_straight.species_and_occu, slab_straight.frac_coords)
        msym, err = mirror_sym_check(slab_orig, tol = tol)
        if msym:
            mirror_sym = True
        if not err == '':
            error = err

        #Checks mirror symmetry of slab_straight with lowermost layer removed
        min_value=np.min(slab_orig.cart_coords[:,2])
        first_layer_size=len(np.where(np.abs(min_value - slab_orig.cart_coords[:,2]) < tol)[0])
        for i in range(first_layer_size):
            index_to_remove=np.argmin(slab_orig.frac_coords[:,2])
            slab_orig.remove_sites([index_to_remove])
        msym, err = mirror_sym_check(slab_orig, tol = tol)
        if msym:
            mirror_sym = True
        if not err == '':
            error = err

        #Checks mirror symmetry of slab_straight with topmost layer removed
        max_value=np.max(slab_straight.cart_coords[:,2])
        first_layer_size=len(np.where(np.abs(max_value - slab_straight.cart_coords[:,2]) < tol)[0])
        for i in range(first_layer_size):
            index_to_remove=np.argmax(slab_straight.frac_coords[:,2])
            slab_straight.remove_sites([index_to_remove])
        msym, err = mirror_sym_check(slab_straight, tol = tol)
        if msym:
            mirror_sym = True
        if not err == '':
            error = err

        #Checks mirror symmetry of slab_straight with lowermost and topmost layers removed
        min_value=np.min(slab_straight.cart_coords[:,2])
        first_layer_size=len(np.where(np.abs(min_value - slab_straight.cart_coords[:,2]) < tol)[0])
        for i in range(first_layer_size):
            index_to_remove=np.argmin(slab_straight.frac_coords[:,2])
            slab_straight.remove_sites([index_to_remove])
        msym, err = mirror_sym_check(slab_straight, tol = tol)
        if msym:
            mirror_sym = True
        if not err == '':
            error = err

        #If nterm is larger than 3, remove additonal layers from either side to ensure
        #that we check all relevant slabs that could yield proof that there is mirror symmetry
        if nterm > 3:
            slab_orig = Structure(slab_straight._lattice, slab_straight.species_and_occu, slab_straight.frac_coords)
            #delete more layers nterm/2 rounded down on either side from the slab_straight (which had one each side already subtracted)
            for term in range(int(np.floor(nterm/2))):
                max_value=np.max(slab_straight.cart_coords[:,2])
                first_layer_size=len(np.where(np.abs(max_value - slab_straight.cart_coords[:,2]) < tol)[0])
                for i in range(first_layer_size):
                    index_to_remove=np.argmax(slab_straight.frac_coords[:,2])
                    slab_straight.remove_sites([index_to_remove])
                msym, err = mirror_sym_check(slab_straight, tol = tol)
                if msym:
                    mirror_sym = True
                if not err == '':
                    error = err

                min_value=np.min(slab_orig.cart_coords[:,2])
                first_layer_size=len(np.where(np.abs(min_value - slab_orig.cart_coords[:,2]) < tol)[0])
                for i in range(first_layer_size):
                    index_to_remove=np.argmin(slab_orig.frac_coords[:,2])
                    slab_orig.remove_sites([index_to_remove])
                msym, err = mirror_sym_check(slab_orig, tol = tol)
                if msym:
                    mirror_sym = True
                if not err == '':
                    error = err

    else:
        #Mirror symmetry is set to False in case there was an error along the way.
        mirror_sym = False

    return mirror_sym, error


def center_single_direction(struc, i, vacuum_thickness = 10):
    '''Auxiliary function that centers structure along direction 'i' (integer) with vacuum of thickness 'vacuum_thickness' (positive float) on each side.
    Returns centered structure in vacuum.
    '''
    #Scale the unit cell to account for vacuum
    structure = struc.copy()
    cell_length = np.linalg.norm(structure.lattice.matrix[i])
    atoms_width = np.max(structure.cart_coords[:,i]) - np.min(structure.cart_coords[:,i])
    scaling_factor = (2*vacuum_thickness+atoms_width) / cell_length
    newlatt = np.array(structure.lattice.matrix) #new lattice vectors
    newlatt[i] = scaling_factor * structure.lattice.matrix[i]
    #Also scale frac_coords accordingly
    scaled_coords = []
    for site in structure:
        coords = site.frac_coords
        coords[i] = site.frac_coords[i] / scaling_factor
        scaled_coords.append(coords)
    #Create structure with vacuum in direction i
    structure = Structure(newlatt, structure.species, scaled_coords, coords_are_cartesian=False)
    #Center the structure after having added the vacuum
    midpoint = (max(structure.frac_coords[:,i]) - min(structure.frac_coords[:,i])) / 2 + min(structure.frac_coords[:,i])
    v_transl = [0,0,0]
    v_transl[i] = 0.5 - midpoint
    structure.translate_sites(list(range(len(structure))), v_transl)
    return structure


def centered_vacuum(struc, vacuum_thickness = 10, direction = 3, box = False):
    '''Function that centers structure along one, two, or all three directions with vacuum of thickness 'vacuum_thickness' on each side.
    If 'box' is set to True, then vacuum is added in all 3 directions and the enclosing box will be orthogonal (commonly used for molecules)
    Crystal 'direction' can be either an integer/string or a list containing the integers 1, 2, 3,
    or the strings 'a', 'b', 'c', or 'x', 'y', 'z'. List can be of up to length 3.
    Returns centered structure in vacuum.
    '''
    structure = struc.copy()
    #First, deal with different input parameter formats for 'direction'
    if not isinstance(direction, list):
        direction = [direction]
    try:
        direction[:] = [int(dir)-1 for dir in direction]
    except:
        for num, i in enumerate(direction):
            if i == 'a' or i == 'x':
                direction[num] = 0
            if i == 'b' or i == 'y':
                direction[num] = 1
            if i == 'c' or i == 'z':
                direction[num] = 2
    #If box is True, set directions to all 3 directions and increase vacuum that is added temporarily
    #(to avoid overflow of atoms for small vacuum_thickness and high unit cell angles)
    if box:
        direction = list(range(3))
        vac = vacuum_thickness
        if vacuum_thickness < max(structure.lattice.abc):
            vacuum_thickness = max(structure.lattice.abc)
    #Add vacuum in all given directions and centering the structure
    for i in direction:
        structure = center_single_direction(structure, i, vacuum_thickness)
    #If box is False, done, otherwise: Restore original vacuum_thickness (stored in 'vac')
    if box:
        direction = list(range(3))
        newlatt = np.zeros((3, 3),float)
        np.fill_diagonal(newlatt, [structure.lattice.a,structure.lattice.b,structure.lattice.c])
        structure = Structure(newlatt, structure.species, structure.cart_coords, coords_are_cartesian=True)
        for i in direction:
            structure = center_single_direction(structure, i, vac)
    return structure


def number_of_atomic_layers(slab, tol, direction = 3):
    #Get number of atomic layers (i.e., unique z_positions within tolerance) in direction (1, 2, or 3 corresponding to x, y, z)
    direction -= 1
    z_positions = slab.cart_coords[:,direction].tolist()
    deleteindex = []
    for indi, i in enumerate(z_positions):
        for indj, j in enumerate(z_positions):
            if indj > indi:
                if j > i:
                    if j-i < tol and indj not in deleteindex:
                        deleteindex.append(indj)
                else:
                    if i-j < tol and indj not in deleteindex:
                        deleteindex.append(indj)
    for dindex in sorted(deleteindex, reverse=True):
        del z_positions[dindex]
    return len(z_positions)


def create_min_thickness_slab(structure, index, num_termination_min, layerthickness, layer_subtractions, tol, min_thickness = 10):
    #Find final_slabthickness such that the cleaned slab is a least min_thickness thick after layer_subtractions times the average layerthickness
    #Cleaning consists of removing one layer on each side and then ensuring that the number of atomic layers is multiples of the number of unique terminations

    slabthickness = min_thickness
    final_slabthickness = 0
    while final_slabthickness < min_thickness + layerthickness * layer_subtractions:
        final_slab = SlabGenerator(structure, index, min_slab_size=slabthickness, min_vacuum_size=5, primitive=True, center_slab=True, in_unit_planes=False, max_normal_search=7).get_slab()
        final_slabthickness = np.max(final_slab.cart_coords[:,2]) - np.min(final_slab.cart_coords[:,2])
        slabthickness += 2 #increase slabthickness for potential next iteration

        if final_slabthickness > 5:#Ensure that slab is thick enough for subtractions
            #"Clean" the two outside surfaces, pymatgen makes weird terminations sometimes
            min_value = np.min(final_slab.cart_coords[:,2])
            first_layer_size = len(np.where(np.abs(min_value - final_slab.cart_coords[:,2]) < tol)[0])
            for i in range(first_layer_size):
                index_to_remove = np.argmin(final_slab.frac_coords[:,2])
                final_slab.remove_sites([index_to_remove])

            max_value = np.max(final_slab.cart_coords[:,2])
            first_layer_size = len(np.where(np.abs(max_value - final_slab.cart_coords[:,2]) < tol)[0])
            for i in range(first_layer_size):
                index_to_remove = np.argmax(final_slab.frac_coords[:,2])
                final_slab.remove_sites([index_to_remove])

            nlayers = number_of_atomic_layers(final_slab, tol)

            #Delete all layers that are not multiple of num_termination_min (e.g. make ABCABCAB -> ABCABC; important for slabs with mirror symmetry)
            n_extra_layers = nlayers % num_termination_min
            if n_extra_layers != 0 and n_extra_layers < nlayers:
                for layer in range(n_extra_layers):
                    max_value = np.max(final_slab.cart_coords[:,2])
                    first_layer_size = len(np.where(np.abs(max_value - final_slab.cart_coords[:,2]) < tol)[0])
                    for i in range(first_layer_size):
                        index_to_remove = np.argmax(final_slab.frac_coords[:,2])
                        final_slab.remove_sites([index_to_remove])

            final_slabthickness = np.max(final_slab.cart_coords[:,2]) - np.min(final_slab.cart_coords[:,2])

    while final_slabthickness - 1.3 * num_termination_min * layerthickness > min_thickness + layerthickness * layer_subtractions:
        for nterm in range(num_termination_min):
            max_value = np.max(final_slab.cart_coords[:,2])
            first_layer_size = len(np.where(np.abs(max_value - final_slab.cart_coords[:,2]) < tol)[0])
            for i in range(first_layer_size):
                index_to_remove = np.argmax(final_slab.frac_coords[:,2])
                final_slab.remove_sites([index_to_remove])

        final_slabthickness = np.max(final_slab.cart_coords[:,2]) - np.min(final_slab.cart_coords[:,2])
    return final_slab
