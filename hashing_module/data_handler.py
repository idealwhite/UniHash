import h5py
import scipy.io as scio

def load_data_coco(path):
    file = h5py.File(path)
    images = file['images'][:].astype('float')#[200015,4,224,224]
    labels = file['LAll'][:]#(20015, 24)
    tags = file['YAll'][:]#(20015, 1386)KEYS
    file.close()
    return images, tags, labels
def load_data(path):
    file = h5py.File(path)
    images = file['images'][:].astype('float')#[200015,4,224,224]
    labels = file['LAll'][:]#(20015, 24)
    tags = file['YAll'][:]#(20015, 1386)KEYS
    keys = file['KEYS'][:]
    file.close()
    keys = [i.decode() for i in keys]
    return images, tags, labels,keys

def load_pretrain_model(path):
    return scio.loadmat(path)

if __name__ == '__main__':
    a = {'s': [12, 33, 44],
         's': 0.111}
    import os
    with open('result.txt', 'w') as f:
        for k, v in a.items():
            f.write(k, v)