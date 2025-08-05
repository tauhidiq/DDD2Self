import numpy as np
import os
import glob
from matplotlib import cm
from matplotlib import pyplot as plt
import cv2
import util
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import utils, models
import torch.nn.functional as F
import torchvision.transforms as T
from sklearn.model_selection import train_test_split
from PIL import Image
from tqdm import trange
from time import sleep
from scipy.io import loadmat
import torchvision.datasets as dset
from torch.utils.data import sampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.distributed as dist
from partialconv2d import PartialConv2d
from model import self2self
import matplotlib.pyplot as plt
from array import *
from partialconv2d import PartialConv2d
from model import self2self
import glob
from skimage.draw import disk, ellipse, polygon
from torch.autograd import Variable

def image_loader(image, device, p1, p2):
    """load image, returns cuda tensor"""
    loader = T.Compose([T.RandomHorizontalFlip(torch.round(torch.tensor(p1))),
                        T.RandomVerticalFlip(torch.round(torch.tensor(p2))),
                        T.ToTensor()
                        ])
    image = Image.fromarray(image.astype(np.uint8))
    image = loader(image).float()
    image = torch.tensor(image)
    image = image.unsqueeze(0)
    return image.to(device)
if __name__ == "__main__":
    USE_GPU = True
    dtype = torch.float32
    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print('using device:', device)
    model = self2self(3)
    # Path to the directory
    path = "/"
    # or
    #path = './'
    # Extract the list of filenames
    files = glob.glob(path + '*', recursive=False)
    folder_list = []
    # Loop to print the filenames
    for filename in files:
        folder_list.append(filename)
    print(len(folder_list))
    image_list = []
    image_folder = []
    for filepath in glob.iglob("/*.png"):
        image_list.append(filepath)
    print(len(image_list))
    z=0

    for z in range(len(image_list)):
        img=np.array(Image.open(image_list[z]))
        #img = np.stack((img1,) * 3, axis=-1)
        print("Start new image running")
        print(img.shape)
        learning_rate = 1e-4
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        w, h, c = img.shape
        rate = 0.5
        NPred = 100
        x = torch.Tensor(img)
        slice_avg = torch.tensor([1, 3, w, h]).to(device)
        i = []
        l = []

        for itr in range(100000):
            input_reshape = x.view(x.size()[0], -1)
            stdev = torch.std(input_reshape, 0, True).data.numpy()
            # each variable is kept with probability of stored in probs
            probs = np.zeros(input_reshape.size()[1])
            for j in range(len(probs)):
                probs[j] = np.random.normal(loc=1 - rate, scale=stdev[j], size=1)
                if probs[j] > 1.0:
                    probs[j] = 1.0
                if probs[j] < 0.0:
                    probs[j] = 0.0
            probs = np.tile(probs, (input_reshape.size()[0], 1))
            mask = Variable(torch.bernoulli(torch.Tensor(probs)), requires_grad=True)
            mask = mask.view(x.size())
            convrt = torch.tensor(mask, requires_grad=True)
            convrt = convrt.detach().numpy()
            # img_input = img*mask
            img_input = img
            # y = img - img_input
            y = y = (1-mask.detach().numpy())*img
            p1 = np.random.uniform(size=1)
            p2 = np.random.uniform(size=1)
            img_input_tensor = image_loader(img, device, p1, p2)
            y = image_loader(y, device, p1, p2)
            mask = np.expand_dims(np.transpose(convrt, [2, 0, 1]), 0)
            mask = torch.tensor(mask).to(device, dtype=torch.float32)
            # print(mask)
            model.train()
            img_input_tensor = img_input_tensor * mask
            output = model(img_input_tensor, mask)

            loader = T.Compose([T.RandomHorizontalFlip(torch.round(torch.tensor(p1))),
                                T.RandomVerticalFlip(torch.round(torch.tensor(p2)))])
            if itr == 0:
                slice_avg = loader(output)
            else:
                slice_avg = slice_avg * 0.99 + loader(output) * 0.01
            # output = model(torch.mul(img_input_tensor,mask))
            # LossFunc = nn.MSELoss(reduction='sum')
            # loss = LossFunc(output*(mask), y*(mask))/torch.sum(mask)
            # loss = torch.sum(abs(output - y))
            loss = torch.sum(abs(output - y) * (1 - mask)) / torch.sum(1 - mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(torch.max(output), torch.max(y))
            print("iteration %d, loss = %.4f" % (itr + 1, loss.item() * 100))
            li = loss.item() * 100
            i.append(itr + 1)
            l.append(li)
            # break
            if (itr + 1) % 1000== 0:
                model.eval()
                img_array = []
                sum_preds = np.zeros((img.shape[0], img.shape[1], img.shape[2]))
                for j in range(NPred):
                    input_reshape = x.view(x.size()[0], -1)
                    stdev = torch.std(input_reshape, 0, True).data.numpy()
                    # each variable is kept with probability of stored in probs
                    probs = np.zeros(input_reshape.size()[1])
                    for j in range(len(probs)):
                        probs[j] = np.random.normal(loc=1 - rate, scale=stdev[j], size=1)
                        if probs[j] > 1.0:
                            probs[j] = 1.0
                        if probs[j] < 0.0:
                            probs[j] = 0.0
                    probs = np.tile(probs, (input_reshape.size()[0], 1))
                    mask = Variable(torch.bernoulli(torch.Tensor(probs)), requires_grad=True)
                    mask = mask.view(x.size())
                    convrt = torch.tensor(mask, requires_grad=True)
                    convrt = convrt.detach().numpy()

                    img_input = img * convrt
                    img_input_tensor = image_loader(img_input, device, 0.1, 0.1)
                    mask = np.expand_dims(np.transpose(convrt, [2, 0, 1]), 0)
                    mask = torch.tensor(mask).to(device, dtype=torch.float32)

                    output_test = model(img_input_tensor, mask)
                    sum_preds[:, :, :] += np.transpose(output_test.detach().cpu().numpy(), [2, 3, 1, 0])[:, :, :, 0]
                    img_array.append(np.transpose(output_test.detach().cpu().numpy(), [2, 3, 1, 0])[:, :, :, 0])
                if z == z:
                   k=z
                   print("k= "+str(k)+" image saving done")
                   # calculate avg
                   average = np.squeeze(np.uint8(np.clip(np.average(img_array, axis=0), 0, 1) * 255))
                   write_img = Image.fromarray(average)
                   write_img.save(folder_list[z]+"/avg-" + str(itr + 1) + ".png")
                   # calculate median
                   med = np.squeeze(np.uint8(np.clip(np.median(img_array, axis=0), 0, 1) * 255))
                   write_img2 = Image.fromarray(med)
                   write_img2.save(folder_list[z]+"/med-" + str(itr + 1) + ".png")

                   # calculate Max
                   # mode_preds = np.squeeze(np.uint8(np.clip(stats.mode(img_array, axis=None), 0, 1) * 255))
                   # mode_preds = np.squeeze(np.uint8(np.clip((3*(np.median(img_array, axis=0))-(2*((sum_preds-np.min(sum_preds)) / (np.max(sum_preds)-np.min(sum_preds)))), 0, 1) * 255))
                   #max_preds = np.squeeze(np.uint8(np.clip(np.max(img_array, axis=0), 0, 1) * 255))  # 3*med - 2 * average
                   #write_img1 = Image.fromarray(max_preds)
                   #write_img1.save(folder_list[z]+"/max-" + str(itr + 1) + ".png")
                   #path = 'folder {}'.format(k)
                   #write_img.save(os.makedirs('folder {}'.format(k)) + "/Self2self_avg-" + str(itr + 1) + ".png")"""
                k=k+z
    z+1

