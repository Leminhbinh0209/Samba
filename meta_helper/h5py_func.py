import os
import h5py

def save_dict(data_dict, filename):
    f = h5py.File(filename, 'w')
    for grp_name in data_dict:
        grp = f.create_group(grp_name)
        for dset_name in data_dict[grp_name]:
            dset = grp.create_dataset(dset_name, data = data_dict[grp_name][dset_name])
    f.close()
    print(f"Successfully saved to {filename}")

def read_dict(filename):
    pass 
    # data_dict = dict()
    # f = h5py.File(filename, 'r')
    # for grp_name in data_dict:
    #     for dset_name in data_dict[grp_name]:
    #         if '_array' in dset_name:
    #             print(grp_name, dset_name, f[grp_name][dset_name][:])
    #         if '_scalar' in dset_name:
    #             print(grp_name, dset_name, f[grp_name][dset_name][()])
    # f.close()
