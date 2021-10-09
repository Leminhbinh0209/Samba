import os
import h5py
from tqdm import tqdm
def save_dict(data_dict, filename):
    """
    Implemetation of Two-level dictionary wrting
    """
    f = h5py.File(filename, 'w')
    grp = f.create_group('data')
    for dset_name in data_dict.keys():
        dset = grp.create_dataset(dset_name, data = data_dict[dset_name])
    f.close()
    print(f"Successfully saved to {filename}")

def read_dict(filename):
    """
    Two-level dictionary reading from h5py 
        {
            key1: {
                intance_keys[0]: data,
                intance_keys[1]: data,
                ...}
        ...}
    output:
        data-frame style dictionary
    """
    # Create an empty datadiction 
    data_dict = dict()
    f = h5py.File(filename, 'r')
    list_keys = f['data'].keys()
    for grp_name in list_keys:
        data_dict[grp_name] = f['data'][grp_name][:]
    f.close()
    return data_dict
