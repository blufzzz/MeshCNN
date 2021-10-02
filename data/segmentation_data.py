import os
import torch
from data.base_dataset import BaseDataset
from util.util import is_mesh_file, pad
import numpy as np
from models.layers.mesh import Mesh
import glob

from IPython.core.debugger import set_trace

class SegmentationData(BaseDataset):

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.opt = opt
        self.device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
        self.root = opt.dataroot # ../fcd_newdataset_meshes/prepared/
        self.dir = os.path.join(opt.dataroot, opt.phase) # ../fcd_newdataset_meshes/prepared/train

        
        # self.paths = sorted(self.make_dataset(self.dir)) # mesh paths
        self.paths = list(filter(lambda x: '.obj' in x, glob.glob(os.path.join(self.dir, '*')))) # REPLACED!

        self.seg_paths = self.get_seg_files(self.paths, os.path.join(self.root, 'seg'), seg_ext='.eseg') # labels, OK!

        self.sseg_paths = self.get_seg_files(self.paths, os.path.join(self.root, 'sseg'), seg_ext='.seseg') # soft-labels, OK!
        
        self.classes, self.offset = self.get_n_segs(os.path.join(self.root, 'classes.txt'), self.seg_paths)
        self.nclasses = len(self.classes)
        self.size = len(self.paths)
        self.get_mean_std() # it will take time!
        # # modify for network later.
        opt.nclasses = self.nclasses
        opt.input_nc = self.ninput_channels

    def __getitem__(self, index):
        
        path = self.paths[index]
        mesh = Mesh(file=path, opt=self.opt, hold_history=True, export_folder=self.opt.export_folder)
        meta = {}
        meta['mesh'] = mesh
        label = read_seg(self.seg_paths[index]) - self.offset # [0,0,1,...,1,0]
        label = pad(label, self.opt.ninput_edges, val=-1, dim=0) # here!
        meta['label'] = label
        soft_label = read_sseg(self.sseg_paths[index])
        meta['soft_label'] = pad(soft_label, self.opt.ninput_edges, val=-1, dim=0)
        # get edge features
        edge_features = mesh.extract_features()
        edge_features = pad(edge_features, self.opt.ninput_edges)
        meta['edge_features'] = (edge_features - self.mean) / self.std
        
        return meta

    def __len__(self):
        return self.size

    @staticmethod
    def get_seg_files(paths, seg_dir, seg_ext='.seg'):
        segs = []
        for path in paths:
            segname = os.path.splitext(os.path.basename(path))[0] + seg_ext
            segfile = os.path.join(seg_dir, segname)
            try:
                assert(os.path.isfile(segfile))
            except:
                set_trace() # cache.seg?
            segs.append(segfile)
        return segs

    @staticmethod
    def get_n_segs(classes_file, seg_files):
        if not os.path.isfile(classes_file):
            all_segs = np.array([], dtype='float64')
            for seg in seg_files:
                seg_file = read_seg(seg) # (6287289,), unique [1,2] 
                all_segs = np.concatenate((all_segs, np.unique(seg_file))) # read_seg takes a lot of time!
            segnames = np.unique(all_segs)
            np.savetxt(classes_file, segnames, fmt='%d')
        classes = np.loadtxt(classes_file)
        offset = classes[0]
        classes = classes - offset
        return classes, offset

    @staticmethod
    def make_dataset(path): # path = ../fcd_newdataset_meshes/prepared/train
        meshes = []
        assert os.path.isdir(path), '%s is not a valid directory' % path

        for root, _, fnames in sorted(os.walk(path)):
            for fname in fnames:
                if is_mesh_file(fname):
                    path = os.path.join(root, fname)
                    meshes.append(path)

        return meshes


def read_seg(seg):
    seg_labels = np.loadtxt(open(seg, 'r'), dtype='float64')
    return seg_labels


def read_sseg(sseg_file):
    sseg_labels = read_seg(sseg_file)
    sseg_labels = np.array(sseg_labels > 0, dtype=np.int32)
    return sseg_labels