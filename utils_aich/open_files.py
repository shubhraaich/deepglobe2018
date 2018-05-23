import os
import shutil
import pickle


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


def is_image_file(filename):
    """ source : https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
    Checks if a file is an image.
    Args:
        filename (string): path to a file
    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)

def get_file_handle(path_, r_w_a) :
    try :
        fhand = open(path_, r_w_a);
    except :
        print("Cannot open file {}".format(path_));
        exit();
    return fhand;

def save_pickle(path_, data) :
    fhand = get_file_handle(path_, 'wb+');
    pickle.dump(data, fhand);
    fhand.close();

def load_pickle(path_) :
    fhand = get_file_handle(path_, 'rb');
    data = pickle.load(fhand);
    fhand.close();
    return data;

def rm_old_mk_new_dir(dir_) :
    if os.path.isdir(dir_) :
        shutil.rmtree(dir_);
    os.mkdir(dir_);
