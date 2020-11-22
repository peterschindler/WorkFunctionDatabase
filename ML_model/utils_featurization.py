
from math import floor
import numpy as np
import math
import ast
import pandas as pd
import statistics
from scipy.stats.mstats import gmean
import time
import matplotlib.pyplot as plt
from ase.io import read, write
from ase import Atoms
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core.periodic_table import Element
from pymatgen.io.cif import CifParser
from pymatgen.core.structure import Structure

with open('atomic_features/firstionizationenergy.txt') as f:
    content = f.readlines()
fie = [float(x.strip()) for x in content]

with open('atomic_features/mendeleev.txt') as f:
    content = f.readlines()
mendeleev = [float(x.strip()) for x in content]

def featurization(slab, bottom = False, tol = 0.7):
    error = None
    if not isinstance(tol, (int, float)) or not isinstance(bottom, bool):
        error = '0.Input parameter(s) do not have correct format.'

    if not error:
        tol = float(tol)
        try:
            struc = read(slab)
        except:
            error = '1.Could not convert/handle input structure'
        if error:
            try:
                struc = AseAtomsAdaptor.get_atoms(slab)
                error = None
            except:
                error = '1.Could not convert/handle input structure'
        if error:
            try:
                slabdic = Structure.from_dict(slab)
                struc = AseAtomsAdaptor.get_atoms(slabdic)
                error = None
            except:
                error = '1.Could not convert/handle input structure'

    if not error:
        for el in struc.get_chemical_symbols():
            if el in ['He','Ne','Ar','Kr','Xe','At','Rn','Fr','Cm','Bk','Cf','Es','Fm','Md','No','Lr']:
                error = '2.Structure contains element not supported for featurization.'

    if not error:
        #struc *= (2,2,1)
        pos = struc.get_positions()
        if len(pos) > 3:
            if pos[0][2] < 0:#correct weird cif import; sometimes all coordinates are negative
                pos = pos * -1

            #Create list of indices from highest z position to lowest
            #--------------------------------------------------------
            counter = 0
            indices_list = []
            while counter < len(pos):
                #Find index for atom(s) with highest z-position
                surface = max(pos[:,2])
                highest_indices = []
                for ind, p in enumerate(pos):
                    if p[2] > surface - tol:
                        highest_indices.append(ind)
                #Once the index/indices of highest atom(s) is/are found, set that position to zero for the next while loop iteration
                #and increase counter by the number of highest found indices.
                if len(highest_indices) > 0:
                    indices_list.append(highest_indices)
                    for ind in highest_indices:
                        pos[ind]=[0, 0, 0]
                    counter = counter + len(highest_indices)
                else:
                    error = '5.Error. No highest index found. Counter = ' + str(counter)
                    break

            #Check there are at least 6 layers, given tolerance to group layers
            if len(indices_list) < 3 and not error:
                error = '4.Slab less than 3 atomic layers in z-direction, with a tolerance = ' + str(tol) + ' A.'

            pos = struc.get_positions()
            if pos[0][2] < 0:#correct weird cif import; sometimes all coordinates are negative
                pos = pos * -1

            #Check if structure is of form slab with vacuum in z-direction
            if pos[indices_list[0][0]][2] - pos[indices_list[-1][0]][2] > struc.get_cell_lengths_and_angles()[2] - 5:
                error = '6.Input structure either has no vacuum between slabs or is not oriented in z-direction'
        else:
            error = '3.Slab less than 4 atomic layers in z-direction before applying tolerance.'

    if not error:
        #Add features
        #------------
        chem = struc.get_chemical_symbols()
        cell = struc.get_cell_lengths_and_angles()

        #Refer to top or bottom surface index:
        sindex = -1 if bottom else 0
        sindex2 = -2 if bottom else 1
        sindex3 = -3 if bottom else 2

        #Feature Layer 1
        f_chi = []
        f_1_r = []
        f_fie = []
        f_mend = []

        for ind in range(len(indices_list[sindex])):
            f_chi.append(Element(chem[indices_list[sindex][ind]]).X)
            if Element(chem[indices_list[sindex][ind]]).atomic_radius_calculated:
                f_1_r.append(1 / Element(chem[indices_list[sindex][ind]]).atomic_radius_calculated)
            else:
                f_1_r.append(1 / Element(chem[indices_list[sindex][ind]]).atomic_radius)
            f_fie.append(fie[Element(chem[indices_list[sindex][ind]]).Z])
            f_mend.append(mendeleev[Element(chem[indices_list[sindex][ind]]).Z])
        f_packing_area = len(indices_list[sindex]) / (cell[0] * cell[1] * math.sin(math.radians(cell[5])))

        #Features layer 2
        f_z1_2 = abs(pos[indices_list[sindex][0]][2] - pos[indices_list[sindex2][0]][2])
        f_chi2 = []
        f_1_r2 = []
        f_fie2 = []
        f_mend2 = []

        for ind2 in range(len(indices_list[sindex2])):
            f_chi2.append(Element(chem[indices_list[sindex2][ind2]]).X)
            if Element(chem[indices_list[sindex2][ind2]]).atomic_radius_calculated:
                f_1_r2.append(1 / Element(chem[indices_list[sindex2][ind2]]).atomic_radius_calculated)
            else:
                f_1_r2.append(1 / Element(chem[indices_list[sindex2][ind2]]).atomic_radius)
            f_fie2.append(fie[Element(chem[indices_list[sindex2][ind2]]).Z])
            f_mend2.append(mendeleev[Element(chem[indices_list[sindex2][ind2]]).Z])
        f_packing_area2 = len(indices_list[sindex2]) / (cell[0] * cell[1] * math.sin(math.radians(cell[5])))

        #Features layer 3
        f_z1_3 = abs(pos[indices_list[sindex][0]][2] - pos[indices_list[sindex3][0]][2])
        f_chi3 = []
        f_1_r3 = []
        f_fie3 = []
        f_mend3 = []

        for ind3 in range(len(indices_list[sindex3])):
            f_chi3.append(Element(chem[indices_list[sindex3][ind3]]).X)
            if Element(chem[indices_list[sindex3][ind3]]).atomic_radius_calculated:
                f_1_r3.append(1 / Element(chem[indices_list[sindex3][ind3]]).atomic_radius_calculated)
            else:
                f_1_r3.append(1 / Element(chem[indices_list[sindex3][ind3]]).atomic_radius)
            f_fie3.append(fie[Element(chem[indices_list[sindex3][ind3]]).Z])
            f_mend3.append(mendeleev[Element(chem[indices_list[sindex3][ind3]]).Z])
        f_packing_area3 = len(indices_list[sindex3]) / (cell[0] * cell[1] * math.sin(math.radians(cell[5])))

        return [error, f_chi, f_chi2, f_chi3, f_1_r, f_1_r2, f_1_r3, f_fie, f_fie2, f_fie3,
            f_mend, f_mend2, f_mend3, f_z1_2, f_z1_3, f_packing_area, f_packing_area2, f_packing_area3];
    else:
        return [error, None, None, None, None, None, None, None, None, None,
                None, None, None, None, None, None, None, None];

def raw_to_final_features(raw, labels = ['f_chi', 'f_chi2', 'f_chi3', 'f_1_r', 'f_1_r2',
                                    'f_1_r3', 'f_fie', 'f_fie2', 'f_fie3', 'f_mend', 'f_mend2', 'f_mend3']):
        deleteindex = []
        for label in labels:
            if 'chi' in label:
                for mat in raw.index:
                    try:
                        list = ast.literal_eval(str(raw.at[mat, label]))
                        raw.at[mat, label] = statistics.mean(list)
                        raw.at[mat, label + '_max'] = max(list)
                        raw.at[mat, label + '_min'] = min(list)
                    except:
                        if mat not in deleteindex:
                            deleteindex.append(mat)
            if '1_r' in label:
                for mat in raw.index:
                    try:
                        list = ast.literal_eval(str(raw.at[mat, label]))
                        raw.at[mat, label] = statistics.mean(list)
                        raw.at[mat, label + '_max'] = max(list)
                        raw.at[mat, label + '_min'] = min(list)
                    except:
                        if mat not in deleteindex:
                            deleteindex.append(mat)
            if 'fie' in label:
                for mat in raw.index:
                    try:
                        list = ast.literal_eval(str(raw.at[mat, label]))
                        raw.at[mat, label] = statistics.mean(list)
                        raw.at[mat, label + '_max'] = max(list)
                        raw.at[mat, label + '_min'] = min(list)
                    except:
                        if mat not in deleteindex:
                            deleteindex.append(mat)
            if 'mend' in label:
                for mat in raw.index:
                    try:
                        list = ast.literal_eval(str(raw.at[mat, label]))
                        raw.at[mat, label] = statistics.mean(list)
                        raw.at[mat, label + '_max'] = max(list)
                        raw.at[mat, label + '_min'] = min(list)
                    except:
                        if mat not in deleteindex:
                            deleteindex.append(mat)
        print('ast errors = ' + str(len(deleteindex)))
        raw = raw.drop(deleteindex)
        raw = raw.reset_index(drop=True)
        deleteindex = []
        nbottom = 0
        for i in raw.index:
            for j in raw.index:
                if j > i and j < i + 30:
                    if (np.isclose(raw.loc[i,'f_chi':'f_mend3_min'].astype(np.double), raw.loc[j,'f_chi':'f_mend3_min'].astype(np.double))).all():
                        if i not in deleteindex:
                            if raw.at[i, 'bottom']:
                                nbottom += 1
                            deleteindex.append(i)
        print('Total deleteindex = ' + str(len(deleteindex)))
        print('Bottom deleted = ' +str(nbottom))
        raw = raw.drop(deleteindex)
        raw = raw.reset_index(drop=True)

        id = str(time.time())
        return raw, id, str(len(deleteindex))
