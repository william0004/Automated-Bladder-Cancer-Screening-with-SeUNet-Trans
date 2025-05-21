import numpy as np


# Define metric calculation functions

def dice_score(preds, targets, num_classes=3):

    scores = []
    preds = np.argmax(preds, axis=1)  # Convert predictions to class labels
    for cls in range(num_classes):
        pred_cls = (preds == cls).astype(int)
        true_cls = (targets == cls).astype(int)
        intersection = (pred_cls * true_cls).sum()
        union = pred_cls.sum() + true_cls.sum()
        score = (2. * intersection) / (union + 1e-6)
        scores.append(score)
    return scores

def iou_score(preds, targets, num_classes=3):

    scores = []
    preds = np.argmax(preds, axis=1)  # Convert predictions to class labels
    for cls in range(num_classes):
        pred_cls = (preds == cls).astype(int)
        true_cls = (targets == cls).astype(int)
        intersection = (pred_cls & true_cls).sum()
        union = (pred_cls | true_cls).sum()
        score = intersection / (union + 1e-6)
        scores.append(score)
    return scores

def nc_ratio_error(preds, targets, nucleus_cls=2, cyto_cls=1):

    errors = []
    preds = np.argmax(preds, axis=1)  # Convert predictions to class labels
    for pred_mask, true_mask in zip(preds, targets):
        pred_nucleus_area = (pred_mask == nucleus_cls).sum()
        pred_cyto_area = (pred_mask == cyto_cls).sum()
        true_nucleus_area = (true_mask == nucleus_cls).sum()
        true_cyto_area = (true_mask == cyto_cls).sum()

        if pred_cyto_area == 0 or true_cyto_area == 0:
            continue

        pred_nc_ratio = pred_nucleus_area / pred_cyto_area
        true_nc_ratio = true_nucleus_area / true_cyto_area
        errors.append(abs(pred_nc_ratio - true_nc_ratio))
    return np.mean(errors) if errors else np.nan

def accuracy(preds, targets):

    preds = np.argmax(preds, axis=1)  # Convert predictions to class labels
    correct = (preds == targets).sum()
    total = np.prod(targets.shape)
    return correct / total