from models import PSNet as PSNet

import argparse
import time
import csv

import numpy as np
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import custom_transforms
from utils import tensor2array
from loss_functions import compute_errors_test
from sequence_folderse import SequenceFolder

import os
from path import Path
from scipy.misc import imsave
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description='Structure from Motion Learner training on KITTI and CityScapes Dataset',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--sequence-length', type=int, metavar='N', help='sequence length for training', default=2)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('--pretrained-dps', dest='pretrained_dps', default=None, metavar='PATH',
                    help='path to pre-trained dpsnet model')
parser.add_argument('--seed', default=0, type=int, help='seed for random functions, and network initialization')
parser.add_argument('--output-dir', default='result', type=str, help='Output directory for saving predictions in a big 3D numpy file')
parser.add_argument('--nlabel', type=int ,default=64, help='number of label')
parser.add_argument('--mindepth', type=float ,default=0.5, help='minimum depth')
parser.add_argument('--maxdepth', type=float ,default=10, help='maximum depth')
parser.add_argument('--output-print', default=True, action='store_true', help='print output depth')
parser.add_argument('--print-freq', default=1, type=int,
                    metavar='N', help='print frequency')

def main():
    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    normalize = custom_transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                            std=[0.5, 0.5, 0.5])
    valid_transform = custom_transforms.Compose([custom_transforms.ArrayToTensor(), normalize])
    val_set = SequenceFolder(
        args.data,
        transform=valid_transform,
        seed=args.seed,
        sequence_length=args.sequence_length
    )

    ##print("------------")
    ##print(val_set)
    print('{} samples found in {} valid scenes'.format(len(val_set), len(val_set.scenes)))
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    dpsnet = PSNet(args.nlabel, args.mindepth).cuda()
    weights = torch.load(args.pretrained_dps)
    dpsnet.load_state_dict(weights['state_dict'])
    dpsnet.eval()

    output_dir= Path(args.output_dir)
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    errors = np.zeros((2, 8, int(np.ceil(len(val_loader)/args.print_freq))), np.float32)
    print("****errors-shape*****")
    print(errors.shape)
    out_depth_max=0.0;tgt_depth_max=10000.0;out_depth_min=10000.0;tgt_depth_min=0.0
    with torch.no_grad():
        for ii, (tgt_img, ref_imgs, ref_poses, intrinsics, intrinsics_inv, tgt_depth, scale_) in enumerate(val_loader):
            if ii % args.print_freq == 0:
                i = int(ii / args.print_freq)
                tgt_img_var = Variable(tgt_img.cuda())
                ref_imgs_var = [Variable(img.cuda()) for img in ref_imgs]
                ref_poses_var = [Variable(pose.cuda()) for pose in ref_poses]
                intrinsics_var = Variable(intrinsics.cuda())
                intrinsics_inv_var = Variable(intrinsics_inv.cuda())
                tgt_depth_var = Variable(tgt_depth.cuda())
                scale = scale_.numpy()[0]

                # compute output
                pose = torch.cat(ref_poses_var,1)
                start = time.time()
                output_depth = dpsnet(tgt_img_var, ref_imgs_var, pose, intrinsics_var, intrinsics_inv_var)
                
                ##print("out-min,out-max,shape,scale,gt-min,gt-max,shape")
                print(output_depth.shape, scale,tgt_depth.shape)
                ##out_depth_max=max(out_depth_max,float(output_depth.max))
                ##tgt_depth_max=max(tgt_depth_max,float(tgt_depth.max))
                ##print("type")
                ##print(type(output_depth))
                ##print("type")
                output_depth=output_depth.cpu()
                output_depth1=output_depth.numpy()
                tgt_depth=tgt_depth.cpu()
                tgt_depth1=tgt_depth.numpy() 
                ##print(type(output_depth1))
                print("out_depth-min-max")
                print(np.min(output_depth1),np.max(output_depth1))
                print("tgt_depth-min-max")
                ##tgt_depth1=tgt_depth.numpy()
                print(np.min(tgt_depth1), np.max(tgt_depth1))
                out_depth_max=max(out_depth_max,np.max(output_depth1))
                tgt_depth_max=min(tgt_depth_max,np.max(tgt_depth1))
                out_depth_min=min(out_depth_min,np.min(output_depth1))
                tgt_depth_min=max(tgt_depth_min,np.min(tgt_depth1))
                
                elps = time.time() - start
                mask = (tgt_depth <= args.maxdepth) & (tgt_depth >= args.mindepth) & (tgt_depth == tgt_depth)

                tgt_disp = args.mindepth*args.nlabel/tgt_depth
                output_disp = args.mindepth*args.nlabel/output_depth


                output_disp_ = torch.squeeze(output_disp.data.cpu(),1)
                output_depth_ = torch.squeeze(output_depth.data.cpu(),1)
                
                np.save('./gt_result/'+'%08d'%i,tgt_depth)
                np.save('./output_depth_result/'+'%08d'%i,output_depth_)
                fil=open('./scale_result/'+'%08d.txt'%i,'a')
                fil.write(str(scale))
                fil.close()
                errors[0,:,i] = compute_errors_test(tgt_depth[mask]/scale, output_depth_[mask]/scale)
                ##print(tgt_depth[mask]/scale, output_depth_[mask]/scale)
                
                print("-----------------")
                print("erros[0,:,i]")
                print(errors[0,:,i])
                errors[1,:,i] = compute_errors_test(tgt_disp[mask]/scale, output_disp_[mask]/scale)

                print('Elapsed Time {} Abs Error {:.4f}'.format(elps, errors[0,0,i]))

                if args.output_print:
                    output_disp_n = (output_disp_).numpy()[0]
                    np.save(output_dir/'{:04d}{}'.format(i,'.npy'), output_disp_n)
                    disp = (255*tensor2array(torch.from_numpy(output_disp_n), max_value=args.nlabel, colormap='bone')).astype(np.uint8)
                    imsave(output_dir/'{:04d}_disp{}'.format(i,'.png'), disp)
                
                if args.output_print:
                    output_depth_n = (output_depth_).numpy()[0]
                    np.save(output_dir/'{:08d}{}'.format(i,'.npy'), output_depth_n)
                    depth = (255*tensor2array(torch.from_numpy(output_depth_n), max_value=args.nlabel, colormap='bone')).astype(np.uint8)
                    imsave(output_dir/'{:08d}_disp{}'.format(i,'.png'), depth)
                    
    mean_errors = errors.mean(2)
    print("out-min,out-max,gt-min,gt-max")  
    print(out_depth_min,out_depth_max,tgt_depth_min,tgt_depth_max)
    print("************")
    print("mean_errors")
    print(mean_errors)
    print("mean_errors[0]")
    print(mean_errors[0])
    error_names = ['abs_rel','abs_diff','sq_rel','rms','log_rms','a1','a2','a3']
    print("{}".format(args.output_dir))
    print("Depth Results : ")
    print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format(*error_names))
    print("{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(*mean_errors[0]))

    print("Disparity Results : ")
    print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format(*error_names))
    print("{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(*mean_errors[1]))

    np.savetxt(output_dir/'errors.csv', mean_errors, fmt='%1.4f', delimiter=',')


if __name__ == '__main__':
    main()
