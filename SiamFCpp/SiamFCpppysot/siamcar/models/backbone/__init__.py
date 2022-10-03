# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from SiamFCpp.SiamFCpppysot.siamcar.models.backbone.alexnet import alexnetlegacy, alexnet
from SiamFCpp.SiamFCpppysot.siamcar.models.backbone.mobile_v2 import mobilenetv2
from SiamFCpp.SiamFCpppysot.siamcar.models.backbone.resnet_atrous import resnet18, resnet34, resnet50
from SiamFCpp.SiamFCpppysot.siamcar.models.backbone.convnext import convnext_base, convnext_tiny, convnext_xlarge, convnext_large, convnext_small

BACKBONES = {
              'alexnetlegacy': alexnetlegacy,
              'mobilenetv2': mobilenetv2,
              'resnet18': resnet18,
              'resnet34': resnet34,
              'resnet50': resnet50,
              'alexnet': alexnet,
              'convnexttiny': convnext_tiny,
              'convnextsmall': convnext_small,
              'convnextbase': convnext_base
            }


def get_backbone(name, **kwargs):
    return BACKBONES[name](**kwargs)
