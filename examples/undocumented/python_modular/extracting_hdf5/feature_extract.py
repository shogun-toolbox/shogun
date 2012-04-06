#!/usr/bin/env python
import h5py

# extracting data and labels from hdf5 file to create shogun features and labels
def get_feature(filedir):
    f = h5py.File(filedir)
    root = f['data']
    data_node = root['/data/data']
    data = data_node.value
    
    label_node = root['/data/label']
    label = label_node.value
    return data, label

if __name__ == '__main__':
    data, label = get_feature('abalone.h5')
    print data
    print label
