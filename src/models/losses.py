import math
import torch
import torch.nn as nn
from configs.base import Config
from torch.nn.functional import linear, normalize


class BaseLoss(nn.Module):
    def set_lambda(self, value: float):
        assert value >= 0.0 and value <= 1.0
        self.lambda_ = value


class CrossEntropyLoss(BaseLoss):
    def __init__(self, cfg: Config):
        super(CrossEntropyLoss, self).__init__()
        self.cel = nn.CrossEntropyLoss(reduction=cfg.loss_reduction)

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        return self.cel(inputs[0], targets)


class CombinedMarginLoss(BaseLoss):
    def __init__(self, cfg: Config):
        """Combined margin loss for SphereFace, CosFace, ArcFace"""
        super(CombinedMarginLoss, self).__init__()
        self.in_features = cfg.embedding_linear_out
        self.out_features = cfg.num_classes
        self.s = cfg.margin_loss_scale  # s (float): scale factor
        self.m1 = cfg.margin_loss_m1  # m1 (float): margin for SphereFace
        self.m2 = (
            cfg.margin_loss_m2
        )  # m2 (float): margin for ArcFace, m1 must be 1.0 and m2 must be 0.0
        self.m3 = (
            cfg.margin_loss_m3
        )  # m3 (float): margin for CosFace, m1 must be 1.0 and m3 must be 0.0

        #self.weight = torch.nn.Parameter(
        #    torch.normal(0, 0.01, (self.out_features, self.in_features))
        #)

        # For ArcFace
        self.cos_m = math.cos(self.m2)
        self.sin_m = math.sin(self.m2)
        self.theta = math.cos(math.pi - self.m2)
        self.sinmm = math.sin(math.pi - self.m2) * self.m2
        self.easy_margin = False

        # CrossEntropyLoss
        self.ce_loss = nn.CrossEntropyLoss(reduction=cfg.loss_reduction)

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        embbedings = inputs[1]
        weight = inputs[-1] 
        norm_embeddings = normalize(embbedings)
        norm_weight_activated = normalize(weight)
        logits = linear(norm_embeddings, norm_weight_activated)
        logits = logits.clamp(-1, 1)

        index_positive = torch.where(targets != -1)[0]
        target_logit = logits[index_positive, targets[index_positive].view(-1)]

        if self.m1 == 1.0 and self.m3 == 0.0:
            with torch.no_grad():
                target_logit.arccos_()
                logits.arccos_()
                final_target_logit = target_logit + self.m2
                logits[index_positive, targets[index_positive].view(-1)] = (
                    final_target_logit
                )
                logits.cos_()
            logits = logits * self.s

        elif self.m3 > 0:
            final_target_logit = target_logit - self.m3
            logits[index_positive, targets[index_positive].view(-1)] = (
                final_target_logit
            )
            logits = logits * self.s
        else:
            raise ValueError("Unsupported margin values.")

        loss = self.ce_loss(logits, targets)
        return loss





class CrossEntropyLoss_CombinedMarginLoss(BaseLoss):
    def __init__(self, cfg: Config):
        super(CrossEntropyLoss_CombinedMarginLoss, self).__init__()
        self.cml_loss = CombinedMarginLoss(cfg)
        self.ce_loss = CrossEntropyLoss(cfg)
        self.lambda_ = cfg.loss_lambda

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        ce_loss = self.ce_loss(inputs, targets)
        cml_loss = self.cml_loss(inputs, targets)
        total_loss = self.lambda_ * ce_loss + (1 - self.lambda_) * cml_loss
        return total_loss






