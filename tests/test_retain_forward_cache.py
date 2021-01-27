from torchvision.models import resnet18
from torchvision.transforms import Compose, Resize, ToTensor
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader
from tacklebox.hooks.model_tracking import ModuleTracker
import torch


def print_out_with_grad(module, grad_in, grad_out, inputs, outputs):
    print('%s output-gradient pairs: ' % module.name, end='')
    for out, grad in zip(outputs, grad_out):
        print(out.dtype, type(grad), end=' ')
    print('')


if __name__ == '__main__':
    model = resnet18()
    model.cuda()

    dataset = CIFAR100('../../data/', train=True, transform=Compose([Resize((224, 224)), ToTensor()]))
    loader = DataLoader(dataset)

    x, y = next(iter(loader))
    x = x.to(0)

    from tacklebox.hook_management import HookManager

    hookmngr = HookManager()

    hookmngr.register_backward_hook(print_out_with_grad, conv=model.conv1, retain_forward_cache=True)

    with hookmngr.hook_all_context():
        model(x).sum().backward()
