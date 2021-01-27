from torchvision.models import resnet18
from torchvision.transforms import Compose, Resize, ToTensor
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader
from tacklebox.hooks.model_tracking import ModuleTracker
import torch


def pre_hook(module, inputs):
    print('setting %s input to zero' % module.name)
    input, = inputs
    return (input - input,)


def forward_hook(module, inputs, outputs):
    print('%s input shape: %s, output shape: %s' % (module.name, repr(inputs[0].shape), repr(outputs[0].shape)))


def backward_hook(module, grad_in, grad_out):
    print('in %s backward hook' % module.name)


if __name__ == '__main__':
    model = resnet18()
    model.cuda()

    dataset = CIFAR100('../../data/', train=True, transform=Compose([Resize((224, 224)), ToTensor()]))
    loader = DataLoader(dataset)

    x, y = next(iter(loader))
    x = x.to(0)

    from tacklebox.hook_management import HookManager
    hookmngr = HookManager()

    hookmngr.register_forward_hook(forward_hook, conv=model.conv1, activate=False)
    hookmngr.register_forward_pre_hook(pre_hook, model.conv1, activate=False, hook_fn_name='pre_hook')
    hookmngr.register_backward_hook(backward_hook, model.conv1)

    hookmngr.remove_module_by_name('conv')

    with hookmngr.hook_all_context():
        model(x).sum().backward()

    hookmngr.register_forward_hook(forward_hook, conv=model.conv1, activate=False)
    hookmngr.register_forward_pre_hook(pre_hook, model.conv1, activate=False)
    hookmngr.register_backward_hook(backward_hook, model.conv1)
    hookmngr.register_forward_hook(forward_hook, pool=model.avgpool)

    hookmngr.remove_hook_function(forward_hook)
    hookmngr.remove_hook_by_name('pre_hook[conv]')

    with hookmngr.hook_all_context():
        model(x).sum().backward()

    hookmngr.register_forward_hook(forward_hook, conv=model.conv1, activate=False)
    hookmngr.register_forward_pre_hook(pre_hook, model.conv1, activate=False)
    hookmngr.register_forward_hook(forward_hook, pool=model.avgpool)

    hookmngr.remove_hook_by_name('pre_hook[conv]')
    hookmngr.remove_hook_function(pre_hook)

    with hookmngr.hook_all_context():
        model(x).sum().backward()

    pass
