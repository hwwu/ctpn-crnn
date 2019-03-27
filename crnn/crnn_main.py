from __future__ import print_function
import argparse
import random
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import numpy as np
from warpctc_pytorch import CTCLoss
import os
import utils
import dataset
import models.crnn as crnn
import re
import params
import time

parser = argparse.ArgumentParser()
parser.add_argument('--trainroot', required=True, help='path to dataset')
parser.add_argument('--valroot', required=True, help='path to dataset')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--model_path', required=True, help='enables cuda')

opt = parser.parse_args()
print(opt)


# custom weights initialization called on crnn
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def val(net, dataset, criterion, max_iter=100):
    print('Start val')
    for p in crnn.parameters():
        p.requires_grad = False
    net.eval()
    data_loader = torch.utils.data.DataLoader(
        dataset, shuffle=True, batch_size=params.batchSize, num_workers=int(params.workers))
    val_iter = iter(data_loader)
    i = 0
    n_correct = 0
    loss_avg = utils.averager()

    max_iter = min(max_iter, len(data_loader))
    for i in range(max_iter):
        data = val_iter.next()
        i += 1
        cpu_images, cpu_texts = data
        batch_size = cpu_images.size(0)
        utils.loadData(image, cpu_images)
        t, l = converter.encode(cpu_texts)
        utils.loadData(text, t)
        utils.loadData(length, l)
        preds = crnn(image)
        # print('-----preds-----')
        # print(preds)
        preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
        # print('-----preds_size-----')
        # print(preds_size)
        cost = criterion(preds, text, preds_size, length) / batch_size
        loss_avg.add(cost)
        _, preds = preds.max(2)
        # print('-----preds.max(2)-----')
        # print(preds)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        # print('-----preds.transpose(1, 0)-----')
        # print(preds)
        sim_preds = converter.decode(preds.data, preds_size.data, raw=False)

        list_1 = []
        for m in cpu_texts:
            list_1.append(m.decode('utf-8', 'strict'))

        # if (i - 1) % 10 == 0:
        # print('-----sim_preds-----list_1-----')
        # print(sim_preds, list_1)
        for pred, target in zip(sim_preds, list_1):
            if pred == target:
                n_correct += 1
            else:
                print('%-20s, gt: %-20s' % (pred, target))

    # raw_preds = converter.decode(preds.data, preds_size.data, raw=True)[:params.n_test_disp]
    # for raw_pred, pred, gt in zip(raw_preds, sim_preds, list_1):
    #     print('%-20s => %-20s, gt: %-20s' % (raw_pred, pred, gt))

    print(n_correct)
    print(max_iter * params.batchSize)
    accuracy = n_correct / float(max_iter * params.batchSize)
    print('Test loss: %f, accuray: %f' % (loss_avg.val(), accuracy))


def trainBatch(net, criterion, optimizer, train_iter):
    data = train_iter.next()
    cpu_images, cpu_texts = data
    # print('----cpu_images-----')
    # print(cpu_images.shape)
    batch_size = cpu_images.size(0)
    utils.loadData(image, cpu_images)
    t, l = converter.encode(cpu_texts)
    utils.loadData(text, t)
    utils.loadData(length, l)
    # print('----image-----')
    # print(image.shape)
    preds = crnn(image)
    preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
    cost = criterion(preds, text, preds_size, length) / batch_size
    crnn.zero_grad()
    cost.backward()
    optimizer.step()
    return cost


def training(start):
    for total_steps in range(start, params.niter):
        train_iter = iter(train_loader)
        i = 0
        print(len(train_loader))
        while i < len(train_loader):
            for p in crnn.parameters():
                p.requires_grad = True
            crnn.train()
            cost = trainBatch(crnn, criterion, optimizer, train_iter)
            loss_avg.add(cost)
            i += 1
            if i % params.displayInterval == 0:
                print('[%d/%d][%d/%d] Loss: %f' %
                      (total_steps, params.niter, i, len(train_loader), loss_avg.val()))
                loss_avg.reset()
            if i % params.valInterval == 0:
                val(crnn, test_dataset, criterion)
        if (total_steps + 1) % params.saveInterval == 0:
            # if i % params.valInterval == 0:
            print('save model ..........')
            ti = time.strftime('%Y-%m-%d', time.localtime(time.time()))
            torch.save(crnn.state_dict(),
                       '{0}/crnn_Rec_done_{1}_{2}.pth'.format(params.experiment, total_steps, ti))
            print('save model done')


if __name__ == '__main__':

    manualSeed = random.randint(1, 10000)  # fix seed
    random.seed(manualSeed)
    np.random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    cudnn.benchmark = True

    # store model path
    if not os.path.exists('./expr'):
        os.mkdir('./expr')

    # read train set
    train_dataset = dataset.lmdbDataset(root=opt.trainroot)
    assert train_dataset
    if not params.random_sample:
        sampler = dataset.randomSequentialSampler(train_dataset, params.batchSize)
    else:
        sampler = None

    # images will be resize to 32*160
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=params.batchSize,
        shuffle=True, sampler=sampler,
        num_workers=int(params.workers),
        collate_fn=dataset.alignCollate(imgH=params.imgH, imgW=params.imgW, keep_ratio=params.keep_ratio))

    # read test set
    # images will be resize to 32*160
    test_dataset = dataset.lmdbDataset(
        root=opt.valroot, transform=dataset.resizeNormalize((280, 32)))
    # root=opt.valroot, transform=dataset.resizeNormalize((560, 64)))

    nclass = len(params.alphabet) + 1
    nc = 1

    converter = utils.strLabelConverter(params.alphabet)
    criterion = CTCLoss()

    # cnn and rnn
    image = torch.FloatTensor(params.batchSize, 3, params.imgH, params.imgH)
    text = torch.IntTensor(params.batchSize * 5)
    length = torch.IntTensor(params.batchSize)

    crnn = crnn.CRNN(params.imgH, nc, nclass, params.nh)
    if opt.cuda:
        crnn.cuda()
        image = image.cuda()
        criterion = criterion.cuda()

    crnn.apply(weights_init)
    # if params.crnn != '':
    #     print('loading pretrained model from %s' % params.crnn)
    #     crnn.load_state_dict(torch.load(params.crnn))
    start = 0
    if opt.model_path != '':
        print('loading pretrained model from %s' % opt.model_path)
        crnn.load_state_dict(torch.load(opt.model_path))
        start = int(str(opt.model_path).split('_')[3]) + 1

    image = Variable(image)
    text = Variable(text)
    length = Variable(length)

    # loss averager
    loss_avg = utils.averager()

    # setup optimizer
    if params.adam:
        optimizer = optim.Adam(crnn.parameters(), lr=params.lr,
                               betas=(params.beta1, 0.999))
    elif params.adadelta:
        optimizer = optim.Adadelta(crnn.parameters(), lr=params.lr)
    else:
        optimizer = optim.RMSprop(crnn.parameters(), lr=params.lr)

    training(start)
