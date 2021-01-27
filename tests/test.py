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
    hookmngr.register_forward_pre_hook(pre_hook, model.conv1, activate=False)
    hookmngr.register_backward_hook(backward_hook, model.conv1)

    model(x).sum().backward()
    hookmngr.deactivate_module_hooks(model.conv1)

    tracker = ModuleTracker(model.layer1[0].conv1, pooling=model.avgpool)
    with tracker.track():
        model(x).sum().backward()
        raw = tracker.gather()

    # try using category in HookManager context
    hookmngr = tracker.hook_manager
    with hookmngr.hook_all_context(category='forward_hook'):
        model(x).sum().backward()
        raw2 = tracker.gather()

    with hookmngr.hook_all_context(category='backward_hook'):
        model(x).sum().backward()
        raw3 = tracker.gather()

    pass
