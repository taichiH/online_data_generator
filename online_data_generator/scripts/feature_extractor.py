import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import models, transforms, utils
import numpy as np
import copy

class FeatureExtractor():
    def __init__(self, model, use_cuda=True, padding=True, threshold=0.6):
        self.model = copy.deepcopy(model)
        self.model = self.model.eval()
        self.use_cuda = use_cuda
        self.feature_maps = []
        self.threshold = threshold

        if self.use_cuda:
            self.model = self.model.cuda()

        self.index = []
        self.f = []
        self.stride = []
        for i, module in enumerate(self.model.children()):
            if isinstance(module, nn.Conv2d):
                self.index.append(i)
                self.f.append(module.kernel_size[0])
                self.stride.append(module.stride[0])
            if isinstance(module, nn.MaxPool2d):
                if padding:
                    module.padding = 1
                self.index.append(i)
                self.f.append(module.kernel_size)
                self.stride.append(module.stride)

        self.rf = np.array(self.calc_rf(self.f, self.stride))

    def save_template_feature_map(self, module, input, output):
        self.template_feature_map = output.detach()

    def save_image_feature_map(self, module, input, output):
        self.image_feature_map = output.detach()

    def calc_rf(self, f, stride):
        rf = []
        for i in range(len(f)):
            if i == 0:
                rf.append(3)
            else:
                rf.append(rf[i-1] + (f[i]-1)*self.product(stride[:i]))
        return rf

    def product(self, lis):
        if len(lis) == 0:
            return 0
        else:
            res = 1
            for x in lis:
                res *= x
            return res

    def calc_extraction_layer(self, template, k=3):
        l = np.sum(self.rf <= min(list(template.size()[-2:]))) - 1
        return max(l - k, 1)

    def calc_NCC(self, F, M):
        c, h_f, w_f = F.shape[-3:]
        tmp = np.zeros((c, M.shape[-2] - h_f, M.shape[-1] - w_f, h_f, w_f))
        for i in range(M.shape[-2] - h_f):
            for j in range(M.shape[-1] - w_f):
                M_tilde = M[:, :, i:i+h_f, j:j+w_f][:, None, None, :, :]
                tmp[:, i, j, :, :] = M_tilde / np.linalg.norm(M_tilde)
        NCC = np.sum(tmp*F.reshape(F.shape[-3], 1, 1, F.shape[-2], F.shape[-1]), axis=(0, 3, 4))
        return NCC

    def nms(self, dets, scores, thresh):
        x1 = dets[:, 0, 0]
        y1 = dets[:, 0, 1]
        x2 = dets[:, 1, 0]
        y2 = dets[:, 1, 1]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]
        return keep


    def extract(self, template, image, layer, use_cython=False):
        self.NCC = self.calc_NCC(
            self.template_feature_map.numpy(), self.image_feature_map.numpy())

        threshold = float(self.threshold) * np.max(self.NCC)
        match_indices = np.array(np.where(self.NCC > threshold)).T

        # print("detected boxes: {}".format(len(match_indices)))

        boxes = []
        centers = []
        scores = []
        for max_index in match_indices:
            i_star, j_star = max_index
            NCC_part = self.NCC[i_star-1:i_star+2, j_star-2:j_star+2]

            x_center = (j_star + self.template_feature_map.size()[-1]/2) * image.shape[-1] // self.image_feature_map.size()[-1]
            y_center = (i_star + self.template_feature_map.size()[-2]/2) * image.shape[-2] // self.image_feature_map.size()[-2]

            x1_0 = x_center - template.size()[-1]/2
            x2_0 = x_center + template.size()[-1]/2
            y1_0 = y_center - template.size()[-2]/2
            y2_0 = y_center + template.size()[-2]/2

            stride_product = self.product(self.stride[:layer])

            x1 = np.sum(
                NCC_part * (x1_0 + np.array([-2, -1, 0, 1]) * stride_product)[None, :]) / np.sum(NCC_part)
            x2 = np.sum(
                NCC_part * (x2_0 + np.array([-2, -1, 0, 1]) * stride_product)[None, :]) / np.sum(NCC_part)
            y1 = np.sum(
                NCC_part * (y1_0 + np.array([-1, 0, 1]) * stride_product)[:, None]) / np.sum(NCC_part)
            y2 = np.sum(
                NCC_part * (y2_0 + np.array([-1, 0, 1]) * stride_product)[:, None]) / np.sum(NCC_part)

            x1 = int(round(x1))
            x2 = int(round(x2))
            y1 = int(round(y1))
            y2 = int(round(y2))
            x_center = int(round(x_center))
            y_center = int(round(y_center))

            boxes.append([(x1, y1), (x2, y2)])
            centers.append((x_center, y_center))
            scores.append(np.sum(NCC_part))

        return boxes, centers, scores, match_indices
