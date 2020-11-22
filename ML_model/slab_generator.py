#pymatgen-2020.4.29
from pymatgen.io.cif import CifParser, CifWriter
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
from pymatgen.core.surface import SlabGenerator
from pymatgen.core.sites import PeriodicSite

import sys
import os
sys.path.append(os.getcwd())
from utils_slab_generator import slab_has_mirror_sym, centered_vacuum, get_unique_terminations_v2, create_min_thickness_slab, number_of_atomic_layers, get_symmetrically_distinct_miller_indices

import numpy as np
import pandas as pd

file = 'input.csv'
mpids = pd.read_csv(file, index_col = 0)

#featurized = pd.DataFrame(columns=['mat','mpid','miller','term'])#,'slab'])

#Delete 'processed' column in file to restart featurizing from scratch
if 'processed' not in mpids:
    mpids['processed'] = False
    if os.path.exists('featurized-' + file):
        os.remove('featurized-' + file)
    if os.path.exists('errors-' + file):
        os.remove('errors-' + file)


for mat in mpids.index:
    mpid = mpids.at[mat,'material_id']
    featurized = pd.DataFrame(columns=['mat','mpid','miller','term','slab','nsites','slab_thickness','nterm','mirror'])#,'slab'])
    errors = pd.DataFrame(columns=['mat','mpid','miller','term','error'])

    if not mpids.at[mat,'processed']:
        structure = CifParser.from_string(mpids.at[mat,'cifs.conventional_standard']).get_structures(primitive = False)[0]

        miller_indices = get_symmetrically_distinct_miller_indices(structure, 1)

        for index in miller_indices:
            index_name = str(index[0]) + str(index[1]) + str(index[2])
            #print(index_name)

            tol = 0.05#tolerance for distinguishing layers in c-direction, in Angstroms
            cutoff = 7#Cutuf for considering local environment, in Angstroms

            #Make slab that is thick enough for algorithm that determines unique terminations (20A + considering the cutoff in either side)
            min_slab = SlabGenerator(structure, index, min_slab_size=2*cutoff+20, min_vacuum_size=5, primitive=True, center_slab=True, in_unit_planes=False, max_normal_search=7).get_slab()
            slab_size = max(min_slab.cart_coords[:,2]) - min(min_slab.cart_coords[:,2])
            #min_slab.to(filename=str(index_name)+'.cif')

            #This section sorts atoms from bottom to top in c-direction; required for get_unique_terminations_v2 function to assess termorder accurately
            reduced_latt = min_slab._lattice.get_lll_reduced_lattice()
            props = min_slab.site_properties
            new_sites = []
            for i, site in enumerate(min_slab):
                frac_coords = reduced_latt.get_fractional_coords(site.coords)
                site_props = {}
                for p in props:
                    site_props[p] = props[p][i]
                new_sites.append(PeriodicSite(site.species,frac_coords, reduced_latt,to_unit_cell=True))
            new_sites_species = np.array([])
            new_sites_coords = np.array([])
            for num_sites in range(len(new_sites)):
                new_sites_species = np.append(new_sites_species, new_sites[num_sites].specie)
                new_sites_coords = np.append(new_sites_coords, new_sites[num_sites].frac_coords)
            new_sites_coords = np.reshape(new_sites_coords, [len(new_sites),3])
            #sorting_indices=np.argsort(new_sites_species)#sort by species, for Vasp
            sorting_indices = np.argsort(new_sites_coords[:,2])
            new_sites_species = new_sites_species[sorting_indices]
            new_sites_coords = new_sites_coords[sorting_indices]
            min_slab = Structure(lattice=new_sites[0].lattice, species=new_sites_species, coords=new_sites_coords)

            term_order, num_termination_min = get_unique_terminations_v2(min_slab, tol = tol, cutoff = cutoff)
            #print(str(min_slab._lattice.angles[0]) +', '+str(min_slab._lattice.angles[1]))
            #If bad term_order, try again with stricter tol and cutoff
            if term_order != '':
                tol = 0.02
                cutoff = 10
                term_order, num_termination_min = get_unique_terminations_v2(min_slab, tol = tol, cutoff = cutoff)
            #print(num_termination_min)
            if num_termination_min > 1:
                msym, merr = slab_has_mirror_sym(min_slab, nterm = num_termination_min)
            else:
                msym, merr = True, ''

            #If termination order doesn't make sense or mirror symmetry could not be determined, skip this Miller index & record error
            #Attention: 'term' in the error file means num_termination_min here, not a specific termination
            if term_order != '' or merr != '':
                if term_order != '':
                    err = pd.DataFrame([[mat, mpid, index_name, num_termination_min, term_order]], columns=['mat','mpid','miller','term','error'])
                elif merr != '':
                    err = pd.DataFrame([[mat, mpid, index_name, num_termination_min, merr]], columns=['mat','mpid','miller','term','error'])
                errors = errors.append(err)
                continue

            layer_subtractions = 0
            number_of_slabs = 1
            if msym and num_termination_min > 1:
                layer_subtractions = num_termination_min - 2
                number_of_slabs = num_termination_min - 1
            else:
                layer_subtractions = 2 * num_termination_min - 2
                number_of_slabs = num_termination_min

            layerthickness = slab_size / (number_of_atomic_layers(min_slab, tol) - 1)#Average atomic layer thickness

            final_slab = create_min_thickness_slab(structure, index, num_termination_min, layerthickness, layer_subtractions, tol, 11)

            #final_slab.to(filename=str(index_name)+".cif")
            slab_name = 0
            current_results = pd.DataFrame(columns=['mat','mpid','miller','term','slab','nsites','slab_thickness','nterm','mirror'])#'f_chi','f_radius','f_packing_area',
                                                    #'f_chi2','f_radius2','f_packing_area2','f_z1_2', 'f_chi3','f_radius3','f_packing_area3','f_z1_3'])
            for i in range(number_of_slabs):
                slab_name += 1
                if(np.max(final_slab.cart_coords[:,2]) - np.min(final_slab.cart_coords[:,2]) > 10):
                    final_slab = centered_vacuum(final_slab, 7.5)#Center the slab in vacuum
                    current_results.at[slab_name,'mpid'] = mpid
                    current_results.at[slab_name,'mat'] = mat
                    current_results.at[slab_name,'miller'] = index_name
                    current_results.at[slab_name,'term'] = slab_name
                    current_results.at[slab_name,'slab'] = final_slab.as_dict()
                    #final_slab.to(filename=str(index_name)+'_'+str(slab_name)+".cif")
                    current_results.at[slab_name,'nsites'] = len(final_slab)
                    current_results.at[slab_name,'slab_thickness'] = np.max(final_slab.cart_coords[:,2]) - np.min(final_slab.cart_coords[:,2])
                    current_results.at[slab_name,'nterm'] = num_termination_min
                    current_results.at[slab_name,'mirror'] = msym

                    if not msym:
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
                else:
                    #deleteindex.append(slab_name+n)
                    err = pd.DataFrame([[mat, mpid, index_name, slab_name, 'Layer less than 10 A thick.']], columns=['mat','mpid','miller','term','error'])
                    errors = errors.append(err)

            featurized = featurized.append(current_results, ignore_index=True, sort=False)
        #After each material (all Miller indices), save files
        with open('slabs-' + file, 'a', newline='') as f:
            featurized.to_csv(f, header=f.tell()==0)
        with open('errors-' + file, 'a', newline='') as g:
            errors.to_csv(g, header=g.tell()==0)
        mpids.at[mat,'processed'] = True
        mpids.to_csv(file)
        print('Done: ' + str(mpid) + ' (' + str(int(mat)) + ' of ' + str(len(mpids.index)) + ')')
#featurized.to_csv('featurized-' + file)#, sep = ';')
if os.path.exists('slabs-' + file):
    reind = pd.read_csv('slabs-' + file, index_col=0)
    reind = reind.reset_index(drop=True)
    reind.to_csv('slabs-' + file)
#errors.to_csv('errors-' + file)
if os.path.exists('errors-' + file):
    reinderr = pd.read_csv('errors-' + file, index_col=0)
    reinderr = reinderr.reset_index(drop=True)
    reinderr.to_csv('errors-' + file)
