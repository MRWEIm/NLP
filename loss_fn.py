import torch.nn as nn
import torch
def loss_func(course_pred, course_true, fine_pred, fine_true):
    course_weights = torch.tensor([1.0, 2.0, 2.0, 2.0], device=course_pred.device)
    course_loss = nn.CrossEntropyLoss(weight=course_weights)(course_pred, course_true)

    fine_pred[:, [11, -1]] = 0
    # fine_weights = torch.tensor([5.0, 1.0, 2.0, 1.0, 3.0, 3.0, 4.0, 4.0, 4.0, 3.5, 2.0, 0.0, 2.5, 2.5, 5.0, 0.0], device=fine_pred.device)
    fine_loss = nn.CrossEntropyLoss()(fine_pred, fine_true)
    return course_loss + fine_loss
