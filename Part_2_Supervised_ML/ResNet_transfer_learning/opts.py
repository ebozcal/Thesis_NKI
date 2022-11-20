import argparse
from pathlib import Path


def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path',
                        default=None,
                        type=Path,
                        help='Root directory path')
    
    parser.add_argument(
        '--n_classes',
        default=3,
        type=int,
        help=
        'Number of classes (activitynet: 200, kinetics: 400 or 600, ucf101: 101, hmdb51: 51)'
    )
    parser.add_argument('--n_pretrain_classes',
                        default=3,
                        type=int,
                        help=('Number of classes of pretraining task.'
                              'When using --pretrain_path, this must be set.'))
    parser.add_argument('--pretrain_path',
                        default="/processing/ertugrul/Part_2/TL_ResNet_pytorch/resnet_50.pth",
                        #default="/processing/ertugrul/Part_2/TL_ResNet_pytorch/resnet_10.pth",
                        type=Path,
                        help='Pretrained model path (.pth).')
    parser.add_argument(
        '--ft_begin_module',
        default='',
        type=str,
        help=('Module name of beginning of fine-tuning'
              '(conv1, layer1, fc, denseblock1, classifier, ...).'
              'The default means all layers are fine-tuned.'))
    parser.add_argument('--sample_size',
                        default=196,
                        type=int,
                        help='Height and width of inputs')

    parser.add_argument('--learning_rate',
                        default=0.1,
                        type=float,
                        help=('Initial learning rate'
                              '(divided by 10 while training by lr scheduler)'))
    parser.add_argument(
        '--batchnorm_sync',
        action='store_true',
        help='If true, SyncBatchNorm is used instead of BatchNorm.')
    parser.add_argument('--no_cuda',
                        action='store_true',
                        help='If true, cuda is not used.')
  
    parser.add_argument('--checkpoint',
                        default=10,
                        type=int,
                        help='Trained model is saved at every this epochs.')
    parser.add_argument(
        '--model',
        default='resnet',
        type=str,
        help=
        '(resnet | resnet2p1d | preresnet | wideresnet | resnext | densenet | ')
    parser.add_argument('--model_depth',
                        default=50,
                        type=int,
                        help='Depth of resnet (10 | 18 | 34 | 50 | 101)')
    parser.add_argument('--conv1_t_size',
                        default=7,
                        type=int,
                        help='Kernel size in t dim of conv1.')
    parser.add_argument('--conv1_t_stride',
                        default=1,
                        type=int,
                        help='Stride in t dim of conv1.')
    parser.add_argument('--no_max_pool',
                        action='store_true',
                        help='If true, the max pooling after conv1 is removed.')
    parser.add_argument('--resnet_shortcut',
                        default='B',
                        type=str,
                        help='Shortcut type of resnet (A | B)')
    parser.add_argument(
        '--resnet_widen_factor',
        default=1.0,
        type=float,
        help='The number of feature maps of resnet is multiplied by this value')
    parser.add_argument('--manual_seed',
                        default=1,
                        type=int,
                        help='Manually set random seed')
    parser.add_argument(
        '--distributed',
        action='store_true',
        help='Use multi-processing distributed training to launch '
        'N processes per node, which has N GPUs.')
    parser.add_argument('--dist_url',
                        default='tcp://127.0.0.1:23456',
                        type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--world_size',
                        default=-1,
                        type=int,
                        help='number of nodes for distributed training')

    args = parser.parse_args()

    return args
