import numpy as np
import itertools
from functools import reduce
import os
import pandas as pd

from pymatgen.io.cif import CifParser, CifWriter
from pymatgen.core.structure import Structure

import joblib
import json

from sklearn.preprocessing import StandardScaler

import sys
sys.path.append(os.getcwd())
from utils_featurization import featurization, raw_to_final_features

file = 'slabs-input.csv'
mpids = pd.read_csv(file, index_col = 0)

mpids.loc[:,'bottom'] = False

mpids = pd.concat([mpids,pd.DataFrame(columns=['f_chi', 'f_chi2', 'f_chi3', 'f_1_r', 'f_1_r2', 'f_1_r3', 'f_fie', 'f_fie2', 'f_fie3',
    'f_mend', 'f_mend2', 'f_mend3', 'f_z1_2', 'f_z1_3', 'f_packing_area', 'f_packing_area2', 'f_packing_area3'])])

#Duplicate entries (rows) and tag 'bottom' entry of duplicate as True
mpids = mpids.iloc[np.arange(len(mpids)).repeat(2)]
mpids = mpids.reset_index(drop=True)
for mat in mpids.index:
    if mat % 2 == 0:
        mpids.loc[mat,'bottom'] = True

for mat in mpids.index:
    tol = 0.3
    slab = Structure.from_dict(eval(mpids.loc[mat,'slab']))
    bottom = True if bool(mpids.at[mat,'bottom']) else False
    features = featurization(slab, bottom = bottom, tol = tol)
    if not features[0]:
        mpids.loc[mat, 'f_chi':'f_packing_area3'] = features[1:]
    else:
        print(features[0])

#mpids.to_csv(file.split('.')[0] + '_featurized-raw_' + str(tol).replace('.','p') + '.csv')

final, id, deleted = raw_to_final_features(mpids)
print('Deleted ' + str(deleted) + ' feature duplicates.')
#final.to_csv(file.split('.')[0] + '_featurized-final_' + str(tol).replace('.','p') + '.csv')

model = joblib.load('RF_1605659041.1692553.joblib')
#Top 15 features from RFE optimized features for RF
features = ['f_chi', 'f_chi2', 'f_1_r', 'f_fie2', 'f_mend',
    'f_z1_2', 'f_z1_3', 'f_packing_area', 'f_packing_area2', 'f_packing_area3',
    'f_chi_min', 'f_1_r_min', 'f_fie_max', 'f_fie_min', 'f_mend2_min']

#Load feature scaling from training
with open('scaler.json', 'r') as f:
    scaler_json = json.load(f)
scaler_load = json.loads(scaler_json)
sc = StandardScaler()
sc.scale_ = scaler_load['scale']
sc.mean_ = scaler_load['mean']
sc.var_ = scaler_load['var']
sc.n_samples_seen_ = scaler_load['n_samples_seen']
sc.n_features_in_ = scaler_load['n_features_in']

for mat in final.index:
    X = sc.transform([final.loc[mat, features].tolist()])
    final.loc[mat, 'WF_predicted'] = model.predict(X)
final.to_csv(file.split('.')[0] + '_predicted_WFs_' + str(tol).replace('.','p') + '.csv')
