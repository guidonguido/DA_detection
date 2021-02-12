from ssd.modeling.anchors.prior_box import PriorBox
from .target_transform import SSDTargetTransform
from .transforms import *


def build_transforms(cfg, is_train=True):
    if is_train:
        if cfg.SOLVER.VARIATION > 0:
            print( "[+] Using AdaIN Variation with probability ", cfg.SOLVER.VARIATION)
            transform = [
                MyCustomTransform(cfg.SOLVER.VARIATION, cfg.SOLVER.ADAIN_STATE, cfg.SOLVER.STYLE_PATH),
                ConvertFromInts(),
                PhotometricDistort(),
                Expand(cfg.INPUT.PIXEL_MEAN),
                RandomSampleCrop(),
                RandomMirror(),
                ToPercentCoords(),
                Resize(cfg.INPUT.IMAGE_SIZE),
                SubtractMeans(cfg.INPUT.PIXEL_MEAN),
                ToTensor(),
            ]
        else:
            transform = [
                ConvertFromInts(),
                PhotometricDistort(),
                Expand(cfg.INPUT.PIXEL_MEAN),
                RandomSampleCrop(),
                RandomMirror(),
                ToPercentCoords(),
                Resize(cfg.INPUT.IMAGE_SIZE),
                SubtractMeans(cfg.INPUT.PIXEL_MEAN),
                ToTensor(),
            ]
    else:
        transform = [
            Resize(cfg.INPUT.IMAGE_SIZE),
            SubtractMeans(cfg.INPUT.PIXEL_MEAN),
            ToTensor()
        ]
    transform = Compose(transform)
    return transform


def build_target_transform(cfg):
    transform = SSDTargetTransform(PriorBox(cfg)(),
                                   cfg.MODEL.CENTER_VARIANCE,
                                   cfg.MODEL.SIZE_VARIANCE,
                                   cfg.MODEL.THRESHOLD)
    return transform
