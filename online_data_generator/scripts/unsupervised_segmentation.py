#from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import cv2
import sys
import numpy as np
from skimage import segmentation # error numpy 1.15.0
import torch.nn.init

from online_data_generator_msgs.srv import Segmentation, SegmentationResponse

class CNN(nn.Module):
    def __init__(
            self, input_dim, n_channel=3, kernel_size=3, stride=1, padding=1):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, n_channel, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(args.nChannel)
        self.conv2 = []
        self.bn2 = []
        for i in range(args.nConv-1):
            self.conv2.append( nn.Conv2d(args.nChannel, args.nChannel, kernel_size=3, stride=1, padding=1 ) )
            self.bn2.append( nn.BatchNorm2d(args.nChannel) )
        self.conv3 = nn.Conv2d(args.nChannel, args.nChannel, kernel_size=1, stride=1, padding=0 )
        self.bn3 = nn.BatchNorm2d(args.nChannel)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu( x )
        x = self.bn1(x)
        for i in range(args.nConv-1):
            x = self.conv2[i](x)
            x = F.relu( x )
            x = self.bn2[i](x)
        x = self.conv3(x)
        x = self.bn3(x)
        return x

class UnsupervisedSegmentation():

    def __init__(self):
        self.use_cuda = torch.cuda.is_available()
        self.set_args()
        rospy.Service('~set_params', Segmentation, self.service_callback)

    def set_args(self, req):
        self.n_channel = req.channel
        self.max_iter = req.max_iter
        self.min_labels = req.min_labels
        self.lr = req.lr
        self.n_conv = req.conv
        self.num_superpixels = req.superpixels
        self.compactness = req.compactness
        self.visualize = req.visualize

    def service_callback(self, req):
        self.set_args(req)
        br = cv_bridge.CvBridge()
        img = br.imgmsg_to_cv2(req.input_image, desired_encoding='bgr8')

        data = torch.from_numpy(
            np.array([img.transpose( (2, 0, 1) ).astype('float32')/255.]) )
        if self.use_cuda:
            data = data.cuda()
        data = Variable(data)

        labels = segmentation.slic(
            img, compactness = self.compactness, n_segments = self.num_superpixels)
        labels = labels.reshape(img.shape[0] * img.shape[1])
        u_labels = np.unique(labels)
        l_inds = []
        for i in range(len(u_labels)):
            l_inds.append(np.where(labels == u_labels[i])[0])

        # train
        model = CNN(data.size(1))

        if self.use_cuda:
            model.cuda()
            for i in range(self.n_conv - 1):
                model.conv2[i].cuda()
                model.bn2[i].cuda()

        model.train()
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr = self.lr, momentum=0.9)
        label_colours = np.random.randint(255,size=(100,3))
        for batch_idx in range(self.max_iter):
            print('current: %d, end: %d' % (batch_idx, self.max_iter))
            # forwarding
            optimizer.zero_grad()
            output = model( data )[ 0 ]
            output = output.permute( 1, 2, 0 ).contiguous().view( -1, self.n_channel )
            ignore, target = torch.max( output, 1 )
            im_target = target.data.cpu().numpy()
            nLabels = len(np.unique(im_target))

            if self.visualize:
                im_target_rgb = np.array([label_colours[ c % 100 ] for c in im_target])
                im_target_rgb = im_target_rgb.reshape( img.shape ).astype( np.uint8 )
                cv2.imshow( "output", im_target_rgb )
                cv2.waitKey(10)

            for i in range(len(l_inds)):
                labels_per_sp = im_target[ l_inds[ i ] ]
                u_labels_per_sp = np.unique( labels_per_sp )
                hist = np.zeros( len(u_labels_per_sp) )
                for j in range(len(hist)):
                    hist[ j ] = len( np.where( labels_per_sp == u_labels_per_sp[ j ] )[ 0 ] )
                im_target[ l_inds[ i ] ] = u_labels_per_sp[ np.argmax( hist ) ]
            target = torch.from_numpy( im_target )
            if self.use_cuda:
                target = target.cuda()
            target = Variable(target)
            loss = loss_fn(output, target)

            loss.backward()
            optimizer.step()

            # print (batch_idx, '/', self..max_iter, ':', nLabels, loss.data[0])
            if nLabels <= self.min_labels:
                print ("nLabels", nLabels, "reached minLabels", self.min_labels, ".")
                break

        output = model(data)[0]
        output = output.permute(1, 2, 0).contiguous().view(-1, self.n_channel)
        ignore, target = torch.max(output, 1)
        im_target = target.data.cpu().numpy()
        im_target_rgb = np.array([label_colours[ c % 100 ] for c in im_target])
        im_target_rgb = im_target_rgb.reshape(img.shape).astype(np.uint8)

        res = SegmentationResponse()
        res.output_image = br.cv2_to_imgmsg(im_target_rgb, encoding='bgr8')
        res.status = True
        return res

if __name__=='__main__':
    rospy.init_node('unsupervised_segmentation')
    node = UnsupervisedSegmentation()
    rospy.spin()
