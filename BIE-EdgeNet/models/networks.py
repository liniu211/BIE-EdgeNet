import torch
import torch.nn as nn
from torch.nn import init
from torch.optim import lr_scheduler

from models.BIE_EdgeNet import BIE_EdgeNet
from models.BIE_EdgeNet_DINO import BIE_EdgeNet_DINO
from models.ChangeDINO.BIE_ChangeDINO import BIE_ChangeDINO


###############################################################################
# Helper Functions
###############################################################################

def get_scheduler(optimizer, args):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        args (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more information.
    """
    if args.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - epoch / float(args.max_epochs + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif args.lr_policy == 'step':
        step_size = args.max_epochs//3
        # args.lr_decay_iters
        scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', args.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02, pretrained_backbone_name='dino'):
    """
    迁移学习专用初始化：只初始化新增层，跳过DINO预训练主干
    pretrained_backbone_name: 预训练主干的层名前缀（根据你的模型实际命名调整）
    """
    def init_func(m):
        classname = m.__class__.__name__
        # 1. 跳过预训练主干层（根据你模型中DINO主干的层名前缀判断，比如'dino_backbone'、'vit'等）
        # 先获取当前层的完整名称（需要给模型层命名，或通过parent判断）
        # 简化方案：如果层属于DINO主干（可通过层名包含特定关键词判断），直接跳过
        layer_name = ''
        for name, module in net.named_modules():
            if module is m:
                layer_name = name
                break
        # 跳过DINO预训练主干的层（关键词根据你的模型实际调整，比如'dino'、'vit'、'backbone'）
        if pretrained_backbone_name in layer_name.lower():
            return

        # 2. 只对需要初始化的层（卷积、线性、转置卷积）进行处理
        if any(cls in classname for cls in ['Conv', 'Linear', 'ConvTranspose']):
            if hasattr(m, 'weight') and m.weight is not None:
                if init_type == 'normal':
                    init.normal_(m.weight.detach(), 0.0, init_gain)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight.detach(), gain=init_gain)
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.detach(), a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.detach(), gain=init_gain)
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.detach(), 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        if len(gpu_ids) > 1:
            net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(args, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create the BIE-EdgeNet family change detection network.

    Parameters:
        args (option class) -- stores the experiment flags; args.net_G selects
                               one of: BIE_EdgeNet | BIE_EdgeNet_DINO | BIE_ChangeDINO
    Return the initialized network.
    """
    if args.net_G == 'BIE_EdgeNet':
        net = BIE_EdgeNet(img_size=args.img_size, input_nc=3, output_nc=2, embed_dim=args.embed_dim, num_classes=args.n_class)

    elif args.net_G == 'BIE_EdgeNet_DINO':
        net = BIE_EdgeNet_DINO(img_size=args.img_size, input_nc=3, output_nc=2, embed_dim=args.embed_dim, num_classes=args.n_class)

    elif args.net_G == 'BIE_ChangeDINO':
        net = BIE_ChangeDINO(
            backbone="mobilenetv2",
            # backbone="resnet34",
            fpn_name="fpn",
            fpn_channels=128,
            deform_groups=4,
            gamma_mode="SE",
            beta_mode="contextgatedconv",
            n_layers=[1, 1, 1, 1],
            extract_ids=[5, 11, 17, 23],
        )

    else:
        raise NotImplementedError('net_G [%s] is not recognized' % args.net_G)

    net = init_net(net, init_type, init_gain, gpu_ids)
    return net
