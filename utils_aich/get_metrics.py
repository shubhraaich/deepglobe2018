import numpy as np


def get_prec_rec(gt, seg) :
    """ returns precision and recall values between gt and seg images """
    # binarize seg and gt
    seg[seg<0.5*seg.max()] = 0;
    gt[gt<0.5*gt.max()] = 0;
    seg[seg>=0.5*seg.max()] = 1;
    gt[gt>=0.5*gt.max()] = 1;

    seg_inv, gt_inv = np.logical_not(seg), np.logical_not(gt);
    true_pos = float(np.logical_and(seg, gt).sum()); # float for division
    true_neg = np.logical_and(seg_inv, gt_inv).sum();
    false_pos = np.logical_and(seg, gt_inv).sum();
    false_neg = np.logical_and(seg_inv, gt).sum();

    prec = true_pos/(true_pos + false_pos);
    rec = true_pos/(true_pos + false_neg);
    return prec, rec;
