"""Run every released baseline entry point on four images before queuing trials (needs a GPU)."""

import copy
import sys
import tempfile

import numpy as np
import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10

import classification as c


def main():
    sys.argv = ['preflight']
    args = c.arg_parser.parse_args()
    args.unlearn_epochs = 1
    args.print_freq = 10000
    args.mask_ratio = 0.5
    base = c.utils.model_dict['resnet18'](num_classes=10).cuda()
    base.normalize = c.utils.NormalizeByChannelMeanStd(
        mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.2435, 0.2616]
    ).cuda()
    ds = CIFAR10(c.DATA, train=True, transform=transforms.ToTensor())
    ds.data = ds.data[:4]
    ds.targets = np.array(ds.targets[:4])
    data = {k: c.loader(copy.deepcopy(ds), True) for k in ('forget', 'retain', 'test', 'val')}
    args.save_dir = tempfile.mkdtemp()
    for name, entry in c.RELEASED.items():
        model = copy.deepcopy(base)
        args.unlearn = entry
        mask = c.saliency_mask(model, ds, 0.5) if name == 'SalUn' else None
        c.unlearn.get_unlearn_method(entry)(data, model, torch.nn.CrossEntropyLoss(), args, mask)
        if name == 'BE':
            c.drop_reject_class(model)
        assert all(torch.isfinite(p).all() for p in model.parameters()), name
        assert model(torch.zeros(1, 3, 32, 32).cuda()).shape[1] == 10, name
        print('PASS', name, flush=True)


if __name__ == '__main__':
    main()
