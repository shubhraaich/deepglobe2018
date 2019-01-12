import os
from PIL import Image

base_path = "/media/aich/DATA/road_extraction/train";
img_path = "images";
gt_path = "masks";
img_suffix = "_sat";
gt_suffix = "_mask";
img_format = "png";


def get_raw_filename(file_name, file_suffix) :
    file_name = os.path.splitext(file_name)[0];
    file_name = file_name.split(file_suffix)[0];
    return file_name;


def change_format(path, file_suffix, img_format) :
    for i, fname in enumerate(sorted(os.listdir(path))) :
        with Image.open(os.path.join(path, fname), "r") as img :
            fname_new = get_raw_filename(fname, file_suffix) + "." + img_format;
            img.save(os.path.join(path, fname_new), img_format);
            os.remove(os.path.join(path, fname));
            print "({}) {} --> {}".format(i+1, fname, fname_new);


def rename_files(path, file_suffix, img_format) :
    for i, fname in enumerate(sorted(os.listdir(path))) :
        fname_new = get_raw_filename(fname, file_suffix) + "." + img_format;
        os.rename(os.path.join(path, fname), os.path.join(path, fname_new));
        print "({}) {} --> {}".format(i+1, fname, fname_new);


def main() :
    change_format(path=os.path.join(base_path, img_path),
                  file_suffix=img_suffix,
                  img_format=img_format);

    rename_files(path=os.path.join(base_path, gt_path),
                  file_suffix=gt_suffix,
                  img_format=img_format);


if __name__ == "__main__" :
    main();
