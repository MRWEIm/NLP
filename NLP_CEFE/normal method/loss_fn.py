import torch.nn as nn
import torch
def loss_func(course_pred, course_true, fine_pred, fine_true, pos_weight=2.0):
    course_weights = torch.tensor([1.0, 2.0, 2.0, 2.0], device=course_pred.device)
    # course_loss = nn.CrossEntropyLoss(weight=course_weights)(course_pred, course_true)
    course_loss = weighted_bce_loss(course_pred, course_true, pos_weight)

    fine_pred[:, [11, -1]] = 0
    # fine_loss = nn.CrossEntropyLoss()(fine_pred, fine_true)
    fine_loss = weighted_bce_loss(fine_pred, fine_true, pos_weight)
    # return course_loss + fine_loss
    return [course_loss, fine_loss]


def weighted_bce_loss(pred, target, pos_weight=6.0):
    weights = torch.ones_like(target)
    weights[target == 1] = pos_weight

    loss_fn = nn.BCELoss(weight=weights)
    loss = loss_fn(pred, target)
    return loss


class AutomaticWeightedLoss(nn.Module):
    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        # 创建一个长度为 num 的张量，元素初始化为 1，且支持梯度更新
        params = torch.ones(num, requires_grad=True)  
        # 将张量封装为 PyTorch 的 Parameter，使其成为模型可学习参数
        self.params = torch.nn.Parameter(params)

    def forward(self, x):
        loss_sum = 0  
        # 遍历输入的损失（x 是包含各任务损失的可变参数）
        for i, loss in enumerate(x):  
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)  
        return loss_sum  