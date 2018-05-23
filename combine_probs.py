import os
import shutil

import numpy as np
import scipy.io
import scipy.misc

from utils_aich.open_files import *

base_path = '../../../databases/road_extraction';
in_path_01 = 'valid_masks_prob_70';
in_path_02 = 'valid_masks_prob_vv150';
in_path_03 = 'valid_masks_prob_s101';
out_path = 'valid_masks_combo';
img_file_suffix, gt_file_suffix = "_sat", "_mask";

def combine_probs(in_path_01, in_path_02, in_path_03, out_path,
                img_file_suffix, gt_file_suffix) :
    rm_old_mk_new_dir(out_path);
    for i, fname in enumerate(sorted(os.listdir(in_path_01))) :
        print i+1;
        if not fname.endswith('.mat') :
            continue;
        data = scipy.io.loadmat(os.path.join(in_path_01, fname))['data'];
        data_2 = scipy.io.loadmat(os.path.join(in_path_02, fname))['data'];
        #data_3 = scipy.io.loadmat(os.path.join(in_path_03, fname))['data'];
        #data = data + data_2 + data_3;
        data = data + data_2;
        data = np.argmax(data, axis=0);
        data = np.stack((data, data, data), axis=-1);
        fname_png = os.path.splitext(fname)[0];
        fname_png = fname_png.split(img_file_suffix)[0];
        fname_png = fname_png + gt_file_suffix + ".png";
        scipy.misc.imsave(os.path.join(out_path, fname_png), data);



def main() :
    combine_probs(in_path_01=os.path.join(base_path, in_path_01),
                  in_path_02=os.path.join(base_path, in_path_02),
                  in_path_03=os.path.join(base_path, in_path_03),
                  out_path=os.path.join(base_path, out_path),
                  img_file_suffix=img_file_suffix,
                  gt_file_suffix=gt_file_suffix,
                  );



if __name__ == "__main__" :
    main();
