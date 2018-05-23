# modified from sources:
# (1) https://github.com/pytorch/examples/blob/
# 42e5b996718797e45c46a25c55b031e6768f8440/imagenet/main.py
import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import numpy as np

import model_defs

import datasets
import transforms

from utils_aich.open_files import *

cudnn.benchmark = True;

parser = argparse.ArgumentParser(description='PyTorch DeepGlobe2018 Road Extraction')
parser.add_argument('--train', default=1, type=int, metavar='N',
                    help='train(1) or test(0)');
parser.add_argument('--data', type=str, metavar='DIR',
                    help='path to training data')
parser.add_argument('--gt', type=str, metavar='DIR',
                    help='path to training gt')
parser.add_argument('--out', type=str, metavar='DIR',
                    help='output directory for test data');
parser.add_argument('--arch', '-a', type=str, metavar='ARCH', default='',
                    help='model architecture to be used');
parser.add_argument('--optim', type=str, metavar='OPTIMIZER', default='sgd',
                    help='optimization algorithm for training');
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)');
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)');
parser.add_argument('--start-epoch', default=1, type=int, metavar='N',
                    help='starting epoch number (useful on restarts)');
parser.add_argument('--end-epoch', default=100, type=int, metavar='N',
                    help='end epoch number');
parser.add_argument('--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate');
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum');
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)');
parser.add_argument('--save-interval', default=1, type=int,
                    metavar='N', help='epoch interval to be saved');
parser.add_argument('--print-freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--load-epoch', default=0, type=int,
                    help='epoch to be loaded for test');
parser.add_argument('--test-stride', default=10, type=int,
                    help='stride used to generate the test samples');
parser.add_argument('--test-batch-size', default=10, type=int,
                    help='batch size in test mode');


def add_dropout2d(model) :
    for m in model.module.children() :
        break;
    model = m;
    tmp = list(model.children());
    # add spatial dropout
    dropout_ind = [];
    count = 0;
    for i in range(len(tmp)) :
        if tmp[i].__class__.__name__ == "Sequential" :
            dropout_ind.append(i+1+count);
            count += 1;
    dropout_ind.pop(); # dont add after last block
    for i in dropout_ind :
        tmp.insert(i, nn.Dropout2d(p=0.5));

    # model
    model = nn.Sequential(*tmp);
    model = nn.DataParallel(model);
    return model;


def add_dropout2d_vgg(model) :
    tmp = list(model.module.model.children());
    # add spatial dropout
    dropout_ind = [];
    count = 0;
    for i in range(len(tmp)) :
        if tmp[i].__class__.__name__ == "MaxPool2d" :
            dropout_ind.append(i+1+count);
            count += 1;
#    dropout_ind.pop(); # dont add after last block
    for i in dropout_ind :
        tmp.insert(i, nn.Dropout2d(p=0.5));

    # model
    model = nn.Sequential(*tmp);
    model = nn.DataParallel(model);
    return model;

def conv_nxn_with_init(in_channels, out_channels, kernel_size, stride, padding, bias):
    """nxn convolution with initialization"""
    layer_ = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                     kernel_size=kernel_size, stride=stride, padding=padding,
                     bias=bias);
    nn.init.xavier_normal(layer_.weight, gain=1.0);
    if bias :
            nn.init.constant(layer_.bias, 0.0);
    return layer_;


def add_conv_layers_vgg(model) :
    tmp = list(model.module.children());
    tmp.append(conv_nxn_with_init(in_channels=2,
                                       out_channels=2,
                                       kernel_size=15, stride=1,
                                       padding=7, bias=False));

    # model
    model = nn.Sequential(*tmp);
    model = nn.DataParallel(model);
    return model;


def main():
    SIZE_IMG = 1024; # 224
    global args
    args = parser.parse_args()

    # create model
    if args.arch.startswith('resnet50') :
        model = model_defs.resnet50_oneway(num_classes=2);
    elif args.arch.startswith('vgg16_bn') :
        model = model_defs.vgg16_bn_oneway(num_classes=2, pretrained=True);
    elif args.arch.startswith('segnet') :
        model = model_defs.segnet(num_classes=2, pretrained=True);

    model = nn.DataParallel(model.model); # dirty trick
    #model = nn.DataParallel(model); # dirty trick


    # open log file
    if args.train == 1 :
        log_dir = 'logs';
        log_name = args.arch + '_new.csv';
        if not os.path.isdir(log_dir) :
            os.mkdir(log_dir);
        log_handle = get_file_handle(os.path.join(log_dir, log_name), 'wb+');
        log_handle.write('Epoch, LearningRate, Momentum, WeightDecay,' + \
                        'Loss, Precision, Recall, Accuracy(IoU), FgWeight, BgWeight\n');
        log_handle.close();

    # check model directory
    model_dir = 'models';
    if not os.path.isdir(model_dir) :
        os.mkdir(model_dir);

    # resume learning based on cmdline arguments
    if ((args.start_epoch > 1) and (args.train==1)) :
        load_epoch = args.start_epoch - 1;
    elif (args.train==0) :
        load_epoch = args.load_epoch;
    else :
        load_epoch = 0;

    if load_epoch > 0 :
        print("=> loading checkpoint for epoch = '{}'"
                        .format(load_epoch));
        checkpoint_name = args.arch + '_ep_' + str(load_epoch) + '.pth.tar';
        checkpoint = torch.load(os.path.join(model_dir, checkpoint_name));
        model.load_state_dict(checkpoint['state_dict']);

    # if args.arch.startswith('resnet50') :
    #     model = add_dropout2d(model);
    # if args.arch.startswith('vgg16_bn') :
    #     model = add_dropout2d_vgg(model);


    #model = add_conv_layers_vgg(model);

    model.cuda(); # transfer to cuda

    print(model);

    mean = load_pickle('./mean');
    std = load_pickle('./std');

    if args.train == 1 :

        train_data_dir, train_gt_dir = args.data, args.gt;
        train_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(train_data_dir, train_gt_dir,
                transform_joint=transforms.Compose_Joint([
                    transforms.RandomCrop(SIZE_IMG),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(), ]),
                transform=transforms.Compose([
                    transforms.ColorJitter(0.3, 0.3, 0.3, 0),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std),]),
                target_transform=transforms.Compose([
                    transforms.ToTensorTarget(),]),
                do_copy=True,
            ),
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True);

        weights = torch.from_numpy(np.array([1.,1.2])).float();
        criterion = nn.CrossEntropyLoss(weights).cuda();

        if args.optim == 'adam' :
            optimizer = torch.optim.Adam(model.parameters(),
                                    lr=args.learning_rate,
                                    weight_decay=args.weight_decay);
        elif args.optim == 'sgd' :
            optimizer = torch.optim.SGD(model.parameters(),
                        lr=args.learning_rate,
                        momentum=args.momentum,
                        weight_decay=args.weight_decay);

        for epoch in range(args.start_epoch, args.end_epoch+1):

            # train for one epoch
            stats_epoch = train(train_loader, model, criterion, optimizer, epoch);

            model_name = args.arch + '_ep_' + str(epoch) + '.pth.tar';
            # get current parameters of optimizer
            for param_group in optimizer.param_groups :
                cur_lr = param_group['lr'];
                cur_wd = param_group['weight_decay'];
                if param_group.has_key('momentum') :
                    cur_momentum = param_group['momentum'];
                else :
                    cur_momentum = 'n/a';
                break; # constant parameters throughout the network

            if epoch % args.save_interval == 0 :
                state = {
                    'epoch': epoch,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'learning_rate': cur_lr,
                    'moemntum': cur_momentum,
                    'weight_decay': cur_wd,
                    'fg_weight': weights[1],
                    'bg_weight': weights[0],
                };

                torch.save(state, os.path.join(model_dir, model_name));

            # write logs using logHandle
            log_handle = get_file_handle(os.path.join(log_dir, log_name), 'ab');
            log_handle.write(str(epoch) + ',' +
                            str(cur_lr) + ',' +
                            str(cur_momentum) + ',' +
                            str(cur_wd) + ',' +
                            str(stats_epoch['loss']) + ',' +
                            str(stats_epoch['prec']) + ',' +
                            str(stats_epoch['recall']) + ',' +
                            str(stats_epoch['acc']) + ',' +
                            str(weights[1]) + ',' +
                            str(weights[0]) + '\n');

            log_handle.close();

#            adjust_learning_rate(optimizer, epoch, 10); # adjust learning rate

    elif args.train == 0 : # test
        testdir = args.data;
        outdir = args.out;
#        stride = args.test_stride;
#        test_batch_size = args.test_batch_size;

        test_transformer = transforms.Compose([
#            transforms.RandomHorizontalFlip(),
#            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),]);
#        test(testdir, outdir, test_transformer, model, load_epoch, stride, SIZE_IMG);
#        test_batch_form(testdir, outdir, test_transformer, model, load_epoch,
#                                stride, SIZE_IMG, test_batch_size);
        test_full_res(testdir, outdir, test_transformer, model, load_epoch, SIZE_IMG);


# ----------------------------------------------------------------------- #

def train(train_loader, model, criterion, optimizer, epoch):
    global args
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    prec = AverageMeter()
    recall = AverageMeter()
    acc = AverageMeter()

    # switch to train mode
    model.train()

    time_start = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - time_start);

        target = target.long().cuda(async=True);
        assert(target.eq(0).sum()+target.eq(1).sum() == target.numel());

        input_var = torch.autograd.Variable(input).cuda();
        target_var = torch.autograd.Variable(target);

        # compute output
        output = model(input_var)

        prec_batch, recall_batch, acc_batch = get_prec_recall_iou(output.data, target);
        # reshape output and target
        # source: https://github.com/delta-onera/segnet_pytorch/blob/master/train.py
        output = output.view(output.size(0),output.size(1), -1);
        output = output.transpose(1,2).contiguous();
        output = output.view(-1,output.size(2));
        target_var = target_var.view(-1);

        loss = criterion(output, target_var);

        # measure accuracy and record loss
        losses.update(loss.data[0], input.size(0))
        prec.update(prec_batch, input.size(0))
        recall.update(recall_batch, input.size(0))
        acc.update(acc_batch, input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - time_start);
        time_start = time.time();

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Precision {prec.val:.3f} ({prec.avg:.3f})\t'
                  'Recall {recall.val:.3f} ({recall.avg:.3f})\t'
                  'IoU {acc.val:.3f} ({acc.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses,
                   prec=prec, recall=recall, acc=acc));

    return {'loss':losses.avg, 'prec':prec.avg, 'recall':recall.avg, 'acc':acc.avg};


def regen_dataset(model, in_data_dir, in_gt_dir, out_data_dir, out_gt_dir,
                  sample_per_file, size_input, transform_joint, transform_img,
                  transform_target, transform_reverse, test_batch_size,
                  max_test_per_file, prec_th, rec_th) :
    import scipy.misc;
    gpu_time = AverageMeter();
    data_time = AverageMeter();
    write_time = AverageMeter();

    file_list = datasets.make_dataset(in_data_dir, in_gt_dir, do_copy=False);

    rm_old_mk_new_dir(out_data_dir);
    rm_old_mk_new_dir(out_gt_dir);

    model.eval(); # switch to evaluate mode
    model.cuda();

    time_start = time.time();
    count = 0; # count processed files

    for count_files, file_path in enumerate(file_list) :
        sample_saved, sample_tested = 0, 0;
        img = datasets.default_loader(file_path[0], is_target=False);
        gt = datasets.default_loader(file_path[1], is_target=True);

        while sample_tested < max_test_per_file :
            input_, target = None, None;
            input_, target = transform_joint(img, gt);
            input_, target = transform_img(input_), transform_target(target);
            input_, target = input_.unsqueeze(0), target.unsqueeze(0);
            for b in range(test_batch_size) :
                tmp_input, tmp_target = transform_joint(img, gt);
                tmp_input, tmp_target = transform_img(tmp_input), transform_target(tmp_target);
                tmp_input = tmp_input.unsqueeze(0);
                input_ = torch.cat((input_, tmp_input), dim=0);
                target = torch.cat((target, tmp_target), dim=0);

            sample_tested += test_batch_size;

            input_var = torch.autograd.Variable(input_, volatile=True);
            target = target.long();

            data_time.update(time.time() - time_start); # data loading time

            time_start = time.time(); # time reset
            # compute output
            output = model(input_var)
            gpu_time.update(time.time() - time_start); # computation time
            time_start = time.time(); # time reset

            prec_batch, recall_batch = get_prec_recall_batch(output.data.cpu(), target);

            ind_to_save = (prec.lt(prec_th) * recall.lt(rec_th)).eq(1); # binary
            for i in range(ind_to_save.size(0)) :
                if ind_to_save[i] == 0 :
                    continue;

                out_img = transform_reverse(input_[i,:,:,:]).numpy();
                out_img = np.around(out_img * 255).astype(np.uint8);
                out_img = np.transpose(out_img, (1,2,0));
                out_gt = (target[i,:,:].numpy() * 255).astype(np.uint8);

                count += 1;
                fname = str(count) + ".png";
                scipy.misc.imsave(os.path.join(out_data_dir, fname), out_img);
                scipy.misc.imsave(os.path.join(out_gt_dir, fname), out_gt);



            write_time.update(time.time() - time_start); # data loading time
            time_start = time.time(); # time reset

            print('File (Tested): [{0}({1})]\t'
                  'Time {gpu_time.val:.3f} ({gpu_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Write {write_time.val:.3f} ({write_time.avg:.3f})'.format(
                   count_files+1, sample_tested, gpu_time=gpu_time,
                   data_time=data_time, write_time=write_time));



def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - time_start);
        time_start = time.time();

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg


def test(testdir, outdir, test_transformer, model, load_epoch, stride, size_input) :
    import scipy.misc;
    gpu_time = AverageMeter();
    data_time = AverageMeter();
    write_time = AverageMeter();

    file_list = datasets.make_dataset_test(testdir);

    if os.path.isdir(outdir) :
        shutil.rmtree(outdir);
    os.mkdir(outdir);

    model.eval(); # switch to evaluate mode
    model.cuda();

    time_start = time.time();
    count = 0; # count processed files

    for file_path in file_list :
        file_ = datasets.default_loader(file_path, is_target=False);
        input_img = test_transformer(file_).unsqueeze(0);
        w,h = file_.size;
        range_w = range(0, w-size_input, stride);
        range_w.append(w-size_input);
        range_h = range(0, h-size_input, stride);
        range_h.append(h-size_input);
        input_ = None;
        for wi in range_w :
            for hi in range_h :
                if input_ is None :
                    input_ = input_img[:, :, hi:hi+size_input, wi:wi+size_input];
                else :
                    input_ = torch.cat((input_, input_img[:, :, hi:hi+size_input, wi:wi+size_input]), dim=0);

        input_var = torch.autograd.Variable(input_, volatile=True);

        data_time.update(time.time() - time_start); # data loading time

        time_start = time.time(); # time reset
        output = model(input_var);
        gpu_time.update(time.time() - time_start); # computation time
        time_start = time.time(); # time reset

#        # softmax computation
        output = output.data.cpu();
#        max_output, _ = output.max(1);
#        output.sub_(max_output.unsqueeze(1)).exp_().div_(output.sum(1).unsqueeze(1));

        # load binary probability matrix and get max indices
        out_prob = torch.zeros(2, h, w);
        count_patch = 0;
        for wi in range_w :
            for hi in range_h :
                out_prob[:, hi:hi+size_input, wi:wi+size_input] += output[count_patch, :, :, :];
                count_patch += 1;
        _, out_max_ind = out_prob.max(0); # get ensemble prob max
        # save binary image
        out_img = out_max_ind.numpy();
        scipy.misc.imsave(os.path.join(outdir, os.path.basename(file_path)), out_img);

        write_time.update(time.time() - time_start); # data loading time
        time_start = time.time(); # time reset

        count += 1;
        print('File: [{0}]\t'
              'Time {gpu_time.val:.3f} ({gpu_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Write {write_time.val:.3f} ({write_time.avg:.3f})'.format(
               count, gpu_time=gpu_time, data_time=data_time,
               write_time=write_time));


def test_batch_form(testdir, outdir, test_transformer, model, load_epoch, stride, size_input, test_batch_size) :
    import scipy.misc;
    import scipy.io;
    img_file_suffix, gt_file_suffix = "_sat", "_mask";

    gpu_time = AverageMeter();
    data_time = AverageMeter();
    write_time = AverageMeter();

    file_list = datasets.make_dataset_test(testdir);

    outdir_prob = outdir + "_prob";
    rm_old_mk_new_dir(outdir);
    rm_old_mk_new_dir(outdir_prob);

    model.eval(); # switch to evaluate mode
    model.cuda();

    time_start = time.time();
    count = 0; # count processed files

    pad_ = [0] * 4; # pad_l, pad_r, pad_t, pad_b
    for file_path in file_list :
        file_ = datasets.default_loader(file_path, is_target=False);
        input_img = test_transformer(file_).unsqueeze(0);
        w,h = input_img.size(3), input_img.size(2);
        if w < size_input :
            pad_[0] = int(round((size_input - w)/2));
            pad_[1] = size_input - w - pad_[0];
            w = size_input;
        if h < size_input :
            pad_[2] = int(round((size_input - h)/2));
            pad_[3] = size_input - h - pad_[2];
            h = size_input;

        if any(pad_) :
            mod_pad = nn.ConstantPad2d(padding=pad_, value=0.);
            input_img = mod_pad(input_img).data;

        range_w = range(0, w-size_input, stride);
        range_w.append(w-size_input);
        range_h = range(0, h-size_input, stride);
        range_h.append(h-size_input);
        wh_list = [];
        for wi in range_w :
            for hi in range_h :
                wh_list.append((wi, hi));

        # load binary probability matrix and get max indices
        out_prob = torch.zeros(2, h, w);
        input_ = None;
        for i, (wi, hi) in enumerate(wh_list) :
            if input_ is None :
                input_ = input_img[:, :, hi:hi+size_input, wi:wi+size_input];
            else :
                input_ = torch.cat((input_, input_img[:, :, hi:hi+size_input, wi:wi+size_input]), dim=0);

            if ((i+1)%test_batch_size==0) or (i+1==len(wh_list)): # batch full or list ended
                input_var = torch.autograd.Variable(input_, volatile=True);

                data_time.update(time.time() - time_start); # data loading time

                time_start = time.time(); # time reset
                output = model(input_var);
                gpu_time.update(time.time() - time_start); # computation time
                time_start = time.time(); # time reset

                output = output.data.cpu();

                i_st = i - input_.size(0) + 1;
                for count_patch, (wi, hi) in enumerate(wh_list[i_st:i+1]) :
                    out_prob[:, hi:hi+size_input, wi:wi+size_input] += output[count_patch, :, :, :];

                input_ = None;

        # -------------------------------------------------------------------- #
        _, out_max_ind = out_prob.max(0); # get ensemble prob max
        # save binary image
        out_img = out_max_ind.numpy();
        out_img = np.stack((out_img, out_img, out_img), axis=-1);

        if any(pad_) :
            out_prob = out_prob[:, pad_[2]:out_prob.shape[1]-pad_[3],
                                   pad_[0]:out_prob.shape[2]-pad_[1]];
            out_img = out_img[pad_[2]:out_img.shape[0]-pad_[3],
                                   pad_[0]:out_img.shape[1]-pad_[1]];
            pad_ = [0] * 4;

        file_name = os.path.splitext(os.path.basename(file_path))[0];
        file_name = file_name.split(img_file_suffix)[0];
        file_name = file_name + gt_file_suffix + ".png";
        scipy.misc.imsave(os.path.join(outdir, file_name), out_img);
        file_path_mat = os.path.splitext(os.path.basename(file_path))[0] + ".mat";
        scipy.io.savemat(os.path.join(outdir_prob, file_path_mat),
                                                    {'data': out_prob.numpy()});

        write_time.update(time.time() - time_start); # data loading time
        time_start = time.time(); # time reset

        count += 1;
        print('File: [{0}]\t'
              'Time {gpu_time.val:.3f} ({gpu_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Write {write_time.val:.3f} ({write_time.avg:.3f})'.format(
               count, gpu_time=gpu_time, data_time=data_time,
               write_time=write_time));
# ---------------------------------------------------------------------------- #


def test_full_res(testdir, outdir, test_transformer, model, load_epoch, size_input) :
    import scipy.misc;
    import scipy.io;
    img_file_suffix, gt_file_suffix = "_sat", "_mask";

    gpu_time = AverageMeter();
    data_time = AverageMeter();
    write_time = AverageMeter();

    file_list = datasets.make_dataset_test(testdir);

    outdir_prob = outdir + "_prob";
    rm_old_mk_new_dir(outdir);
    rm_old_mk_new_dir(outdir_prob);

    model.eval(); # switch to evaluate mode
    model.cuda();

    time_start = time.time();
    count = 0; # count processed files

    pad_ = [0] * 4; # pad_l, pad_r, pad_t, pad_b
    for count, file_path in enumerate(file_list) :
        file_ = datasets.default_loader(file_path, is_target=False);
        input_img = test_transformer(file_).unsqueeze(0);
        w,h = input_img.size(3), input_img.size(2);
        if w < size_input :
            pad_[0] = int(round((size_input - w)/2));
            pad_[1] = size_input - w - pad_[0];
            w = size_input;
        if h < size_input :
            pad_[2] = int(round((size_input - h)/2));
            pad_[3] = size_input - h - pad_[2];
            h = size_input;

        if any(pad_) :
            mod_pad = nn.ConstantPad2d(padding=pad_, value=0.);
            input_img = mod_pad(input_img).data;

        # load binary probability matrix and get max indices
        input_ = input_img;
        input_var = torch.autograd.Variable(input_, volatile=True);

        data_time.update(time.time() - time_start); # data loading time

        time_start = time.time(); # time reset
        output = model(input_var);
        gpu_time.update(time.time() - time_start); # computation time
        time_start = time.time(); # time reset

        out_prob = output.data.cpu().squeeze();
        # -------------------------------------------------------------------- #
        _, out_max_ind = out_prob.max(0); # get ensemble prob max
        # save binary image
        out_img = out_max_ind.numpy();
        out_img = np.stack((out_img, out_img, out_img), axis=-1);

        if any(pad_) :
            out_prob = out_prob[:, pad_[2]:out_prob.shape[1]-pad_[3],
                                   pad_[0]:out_prob.shape[2]-pad_[1]];
            out_img = out_img[pad_[2]:out_img.shape[0]-pad_[3],
                                   pad_[0]:out_img.shape[1]-pad_[1]];
            pad_ = [0] * 4;

        file_name = os.path.splitext(os.path.basename(file_path))[0];
        file_name = file_name.split(img_file_suffix)[0];
        file_name = file_name + gt_file_suffix + ".png";
        scipy.misc.imsave(os.path.join(outdir, file_name), out_img);
        file_path_mat = os.path.splitext(os.path.basename(file_path))[0] + ".mat";
        scipy.io.savemat(os.path.join(outdir_prob, file_path_mat),
                                                    {'data': out_prob.numpy()});

        write_time.update(time.time() - time_start); # data loading time
        time_start = time.time(); # time reset

        print('File: [{0}]\t'
              'Time {gpu_time.val:.3f} ({gpu_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Write {write_time.val:.3f} ({write_time.avg:.3f})'.format(
               count+1, gpu_time=gpu_time, data_time=data_time,
               write_time=write_time));




# ----------------------------------------------------------------------- #
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, epoch_interval):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""

#    lr = args.learning_rate * (0.1 ** (epoch // epoch_interval))
    if epoch % epoch_interval == 0 :
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1;


def get_prec_recall_acc(output, target):
    """Computes precision and recall"""
    _, max_inds = output.max(1);
    tmp = max_inds.sub(target);
    false_neg = float(tmp.eq(-1).sum());
    false_pos = float(tmp.eq(1).sum());
    tmp = max_inds.add(target);
    true_neg = float(tmp.eq(0).sum());
    true_pos = float(tmp.eq(2).sum());

    prec = true_pos/(true_pos + false_pos + 1e-6);
    recall = true_pos/(true_pos + false_neg + 1e-6);
    acc = (true_pos + true_neg)/target.numel();

#    if prec==0 and recall==0 : # chance is that TP=0 in the input
#        prec, recall = 1.0, 1.0;

    return prec, recall, acc;


def get_prec_recall_iou(output, target):
    """Computes precision and recall"""
    _, max_inds = output.max(1);
    tmp = max_inds.sub(target);
    false_neg = float(tmp.eq(-1).sum());
    false_pos = float(tmp.eq(1).sum());
    tmp = max_inds.add(target);
    true_neg = float(tmp.eq(0).sum());
    true_pos = float(tmp.eq(2).sum());

    prec = true_pos/(true_pos + false_pos + 1e-6);
    recall = true_pos/(true_pos + false_neg + 1e-6);
    iou = true_pos/(true_pos + false_pos + false_neg);

    return prec, recall, iou;


def get_prec_recall_batch(output, target):
    """Computes precision and recall"""
    _, max_inds = output.max(1);
    tmp = max_inds.sub(target);
    tmp = tmp.view(tmp.size(0), -1);
    false_neg = tmp.eq(-1).sum(1).float(); # preserves batch dim
    false_pos = tmp.eq(1).sum(1).float();
    tmp = max_inds.add(target);
    tmp = tmp.view(tmp.size(0), -1);
    true_neg = tmp.eq(0).sum(1).float();
    true_pos = tmp.eq(2).sum(1).float();

    prec = true_pos/(true_pos + false_pos + 1e-6);
    recall = true_pos/(true_pos + false_neg + 1e-6);

    return prec, recall;


if __name__ == '__main__':
    main();
