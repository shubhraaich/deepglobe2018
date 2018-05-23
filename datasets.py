import torch.utils.data as data

from PIL import Image
import os
from copy import deepcopy

from utils_aich.open_files import *

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']


def is_image_file(filename):
    """Checks if a file is an image.
    Args:
        filename (string): path to a file
    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)


def make_dataset(data_dir, gt_dir, do_copy=False):
    """ create a repeated list of existing images to train over ~1e5 samples in
    each epoch. """
    num_copies = 1;
    images = [];
    data_dir = os.path.expanduser(data_dir);
    gt_dir = os.path.expanduser(gt_dir);
    for fname in sorted(os.listdir(data_dir)):
        if not is_image_file(fname):
            continue;
        images.append((os.path.join(data_dir, fname), os.path.join(gt_dir, fname)));

    if do_copy :
        tmp_images = deepcopy(images); # used in list concat

        # copy list for number of additional length
        for i in xrange(1, num_copies) :
            images += tmp_images;
        del tmp_images;

    return images;


def make_dataset_test(dir):
    images = []
    dir = os.path.expanduser(dir)
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images


def pil_loader(path, is_target):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        if is_target :
            return img.convert('L');
        return img.convert('RGB');


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path, is_target):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path, is_target)
    else:
        return pil_loader(path, is_target)


class ImageFolder(data.Dataset):
    """A generic data loader where the images are arranged in this way: ::
        (data_dir/xxx.png, gt_dir/xxx.png)
        (data_dir/xxy.png, gt_dir/xxy.png)
        (data_dir/xxz.png, gt_dir/xxz.png)
        (data_dir/123.png, gt_dir/123.png)
        (data_dir/nsdf3.png, gt_dir/nsdf3.png)
        (data_dir/asd932_.png, gt_dir/asd932_.png)
    Args:
        data_dir (string): directory containing training images.
        gt_dir (string): directory containing training gt images.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
     Attributes:
        imgs (list): List of (image path, gt path) tuples
    """

    def __init__(self, data_dir, gt_dir, do_copy, transform_joint=None,
                transform=None, target_transform=None, loader=default_loader):
        imgs = make_dataset(data_dir, gt_dir, do_copy);
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + data_dir + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.data_dir = data_dir
        self.gt_dir = gt_dir
        self.imgs = imgs
        self.transform_joint = transform_joint
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path_data, path_target = self.imgs[index]
        img = self.loader(path_data, is_target=False)
        target = self.loader(path_target, is_target=True);
        if self.transform_joint is not None:
            img, target = self.transform_joint(img, target)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target;

    def __len__(self):
        return len(self.imgs)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Data Location: {}\n'.format(self.data_dir)
        fmt_str += '    GT Location: {}\n'.format(self.gt_dir)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
