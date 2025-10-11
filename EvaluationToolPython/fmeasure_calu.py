import numpy as np


def fmeasure_calu(smap, gt_map, gt_size, threshold):
    """
    Calculate F-measure, Precision, Recall, Specificity, and Dice coefficient.

    This is a Python implementation of the MATLAB Fmeasure_calu function.

    Parameters:
    smap (numpy array): Saliency map
    gt_map (numpy array): Ground truth map
    gt_size (tuple): Size of ground truth
    threshold (float): Threshold value for binarization

    Returns:
    tuple: (precision, recall, specificity, dice, fmeasure)
    """
    if threshold > 1:
        threshold = 1

    # Create binary prediction map
    label3 = np.zeros(gt_size)
    label3[smap >= threshold] = 1

    # Calculate metrics
    num_rec = np.sum(label3 == 1)  # FP+TP
    num_no_rec = np.sum(label3 == 0)  # FN+TN

    # Intersection (TP)
    label_and = label3 & gt_map
    num_and = np.sum(label_and == 1)  # TP

    num_obj = np.sum(gt_map)  # TP+FN
    num_pred = np.sum(label3)  # FP+TP

    # Calculate confusion matrix elements
    fn = num_obj - num_and  # False Negatives
    fp = num_rec - num_and  # False Positives
    tn = num_no_rec - fn  # True Negatives

    if num_and == 0:
        precision = 0
        recall = 0
        fmeasure = 0
        dice = 0
        specificity = 0
    else:
        precision = num_and / num_rec if num_rec > 0 else 0
        recall = num_and / num_obj if num_obj > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        dice = 2 * num_and / (num_obj + num_pred) if (num_obj + num_pred) > 0 else 0
        fmeasure = (
            (2.0 * precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

    return precision, recall, specificity, dice, fmeasure
