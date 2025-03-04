import torch
from torch import nn
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from utils import box_ops as box
from loss.loss import HungarianMatcher, SetCriterion

from models.backbone.backbone_resnet import build_resnet
from models.head import DynamicHead


def build_backbone():
    # 临时
    backbone, conv5 = build_resnet(name="resnet50", pretrained=True)
    return backbone


def build_neck():
    pass


def build_roi_head():
    # 临时
    return DynamicHead(cfg=None, roi_input_shape=None)


class Sparsercnn(nn.Module):
    def __init__(self, cfg=None):
        super(Sparsercnn, self).__init__()
        # base information
        self.num_proposals = 100  # #cfg.MODEL.SparseRCNN.ALPHA
        self.hidden_dim = 256  # #cfg.MODEL.SparseRCNN.ALPHA
        self.num_head = 6  # #cfg.MODEL.SparseRCNN.ALPHA
        self.num_classes = 1  # #cfg.MODEL.SparseRCNN.ALPHA
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
        # Build Backbone
        self.backbone = build_backbone()
        self.neck = build_neck()

        # Build Proposals
        self.init_proposal_features = nn.Embedding(self.num_proposals, self.hidden_dim)
        self.init_proposal_boxes = nn.Embedding(self.num_proposals, 4)
        nn.init.constant_(self.init_proposal_boxes.weight[:, :2], 0.5)
        nn.init.constant_(self.init_proposal_boxes.weight[:, 2:], 1.0)

        # Build Roi Head
        self.roi_head = build_roi_head()

        # Loss parameters
        class_weight = 1  # cfg.MODEL.LOSS.CLASS_WEIGHTS
        giou_weight = 1  # cfg.MODEL.LOSS.GIOU_WEIGHTS
        l1_weight = 1  # cfg.MODEL.LOSS.L1_WEIGHTS
        reid_weight = 1  # cfg.MODEL.LOSS.REID_WEIGHTS
        no_object_weight = 1  # cfg.MODEL.LOSS.NO_OBJECT_WEIGHT
        self.deep_supervision = True  # cfg.MODEL.LOSS.DEEP_SUPERVISION
        self.use_focal = True  # cfg.MODEL.LOSS.USE_FOCAL

        # Build Criterion
        matcher = HungarianMatcher(cfg=cfg,
                                   cost_class=class_weight,
                                   cost_bbox=l1_weight,
                                   cost_giou=giou_weight,
                                   use_focal=self.use_focal
                                   )
        weight_dict = {'loss_ce': class_weight,
                       'loss_bbox': l1_weight,
                       'loss_giou': giou_weight,
                       'loss_reid': reid_weight}

        if self.deep_supervision:
            aux_weight_dict = {}
            for i in range(self.num_head - 1):
                aux_weight_dict.update({k + f'head_{i}': v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)
        losses = ['labels', 'boxes']

        self.criterion = SetCriterion(cfg=cfg,
                                      num_classes=self.num_classes,
                                      matcher=matcher,
                                      weight_dict=weight_dict,
                                      eos_coef=no_object_weight,
                                      losses=losses,
                                      use_focal=self.use_focal)

        # Build transform
        transform = GeneralizedRCNNTransform(
            min_size=900,  # cfg.INPUT.MIN_SIZE,
            max_size=1500,  # cfg.INPUT.MAX_SIZE,
            image_mean=[0.485, 0.456, 0.406],
            image_std=[0.229, 0.224, 0.225],
        )
        self.transform = transform

    def inference(self, images, targets=None, query_img_as_gallery=False):
        pass

    def forward(self, images, targets=None, query_img_as_gallery=False):
        if not self.training:
            return self.inference(images, targets, query_img_as_gallery)
        # preprocess image
        images, targets, images_whwh = self.preprocess_image(images, targets)
        # Feature Extraction
        features = self.extract_feat(images.tensors)
        # print(images.image_sizes) [(99, 177), (99, 177)]
        # prepare proposals
        proposal_boxes = self.init_proposal_boxes.weight.clone()
        proposal_feats = self.init_proposal_features.weight
        proposal_boxes = box.box_cxcywh_to_xyxy(proposal_boxes)
        proposal_boxes = proposal_boxes[None] * images_whwh[:, None, :]
        # prediction

        outputs_class, outputs_coord, outputs_reid = self.roi_head(features, proposal_boxes,
                                                                   proposal_feats,images.image_sizes)
        output = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1],
                  'pred_reid': outputs_reid[-1],'aux_outputs': []}

        for a, b, c in zip(outputs_class[:-1], outputs_coord[:-1], outputs_reid[:-1]):
            output['aux_outputs'].append({'pred_logits': a, 'pred_boxes': b, 'pred_reid': c})
        loss_dict = self.criterion(output, targets)
        weight_dict = self.criterion.weight_dict
        for k in loss_dict.keys():
            if k in weight_dict:
                loss_dict[k] *= weight_dict[k]
        return loss_dict

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.neck is not None:
            x = self.neck(x)
        return x

    def preprocess_image(self, images, targets):
        images, targets = self.transform(images, targets)
        images_whwh = [torch.tensor([hw[1], hw[0], hw[1], hw[0]],
                                    dtype=torch.float32, device=self.device)
                       for hw in images.image_sizes]
        targets = self.prepare_targets(targets, images_whwh)
        images_whwh = torch.stack(images_whwh)
        return images, targets, images_whwh

    def prepare_targets(self, targets, images_whwh):
        new_targets = []
        for targets_per_image, image_size_xyxy in zip(targets, images_whwh):
            target = {}
            gt_classes = targets_per_image['labels']
            gt_boxes = targets_per_image['boxes'] / image_size_xyxy
            # gt_boxes = box.box_xyxy_to_cxcywh(gt_boxes) # cxcywh
            target["labels"] = gt_classes.to(self.device)
            # target["boxes"] = gt_boxes.to(self.device)  # cxcywh
            target["boxes_xyxy"] = targets_per_image['boxes'].to(self.device)
            target["image_size_xyxy"] = image_size_xyxy.to(self.device)
            image_size_xyxy_tgt = image_size_xyxy.unsqueeze(0).repeat(len(gt_boxes), 1)
            target["image_size_xyxy_tgt"] = image_size_xyxy_tgt.to(self.device)
            new_targets.append(target)

        return new_targets
