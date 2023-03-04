import torch
from torchmetrics.classification import BinaryAccuracy, BinaryJaccardIndex, BinaryPrecision, BinaryRecall, BinaryF1Score

""" Dice loss function """
def dc_loss(pred, target):
    smooth = 1.

    predf = pred.view(-1)
    targetf = target.view(-1)
    intersection = (predf * targetf).sum()

    return 1 - ((2. * intersection + smooth) /
                (predf.sum() + targetf.sum() + smooth))


""" IoU loss (Jaccard loss) """
def jac_loss(pred, target):
    smooth = 1.

    predf = pred.view(-1)
    targetf = target.view(-1)
    intersection = (predf * targetf).abs().sum()
    sum_ = torch.sum(predf.abs() + targetf.abs())

    jac = (intersection + smooth) / (sum_ - intersection + smooth)

    return (1 - jac) * smooth


""" Jaccard index """
def binary_jac(maskf, predf):
    m = BinaryJaccardIndex()
    m.update(maskf, predf)

    return m.compute()

""" Accuracy = (tp + tn) / (tp + tn + fp + tn) """
def binary_acc(maskf, predf):
    m = BinaryAccuracy()
    m.update(maskf, predf)

    return m.compute()

""" Precision = tp / (tp + fp) """
def binary_prec(maskf, predf):
    m = BinaryPrecision()
    m.update(maskf, predf)

    return m.compute()

""" Recall = tp / (tp + fn) """
def binary_rec(maskf, predf):
    m = BinaryRecall()
    m.update(maskf, predf)

    return m.compute()

""" F1-score = 2 * ((P * R) / (P + R)) """
def binary_f1s(maskf, predf):
    m = BinaryF1Score()
    m.update(maskf, predf)

    return m.compute()
