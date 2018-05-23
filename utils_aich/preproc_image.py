

def binarize_gray_image(im) :
    im[im<0.5*im.max()] = 0;
    im[im>=0.5*im.max()] = 1;
    return im.astype(bool);
