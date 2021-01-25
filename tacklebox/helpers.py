import json
import torch
from tqdm.auto import tqdm


def find_network_modules_by_name(network, module_names):
    """
    Searches the the result of network.named_modules() for the modules specified in module_names and returns them
    :param network: the network to search
    :param module_names: List[String] containing the module names to search for
    :return: List[torch.nn.Module] of modules specified in module_names
    """
    assert hasattr(network, 'named_modules'), 'Network %s has no attribute named_modules' % repr(network)
    assert len(list(network.named_modules())) > 0, 'Network %s has no modules in it' % repr(network)
    ret_modules = []
    module_names = set(module_names)
    all_found = False
    for name, module in network.named_modules():
        if name in module_names:
            ret_modules += [module]
            module_names.discard(name)
            if len(module_names) == 0:
                all_found = True
                break
    assert all_found, 'Could not find the following modules in the passed network: %s' % \
                      ', '.join(module_names)
    return ret_modules


def get_named_modules_from_network(network, include_bn=False):
    """
    Returns all modules in network.named_modules() that have a 'weight' attribute as a Dict indexed by module name
    :param network: the network to search
    :param include_bn: if True, include BatchNorm layers in the returned Dict
    :return: Dict[String, torch.nn.Module] containing modules indexed by module name
    """
    assert hasattr(network, 'named_modules'), 'Network %s has no attribute named_modules' % repr(network)
    assert len(list(network.named_modules())) > 0, 'Network %s has no modules in it' % repr(network)
    ret_modules = {}
    for name, module in network.named_modules():
        if not hasattr(module, 'weight'):
            continue
        if type(module) == torch.nn.BatchNorm2d and not include_bn:
            continue
        ret_modules[name] = module

    return ret_modules


def set_torchvision_network_module(network, module_name, new_module):
    """
    Overwrite a particular module in a torchvision model. Assumes modules are stored in member variables or
    sequential objects that may be referenced by index.
        Eg. network['layer4.1.conv2'] => network.layer4[1].conv2
    :param network: the network to overwrite in
    :param module_name: name of the module to replace
    :param new_module: the torch.nn.Module to replace the existing module with
    :return: the network passed in after module replacement
    """
    ids = module_name.split('.')
    parent_module = network
    for id_ in ids[:-1]:
        if id_.isnumeric():
            parent_module = parent_module[int(id_)]
        else:
            parent_module = getattr(parent_module, id_)
    if ids[-1].isnumeric():
        parent_module[int(ids[-1])] = new_module
    else:
        setattr(parent_module, ids[-1], new_module)

    return network


def data_pass(loader, network, device=0, backward_fn=None, early_stop=None,
              pre_backward_hooks=[], pre_zero_grad_hooks=[]):
    """
    Perform a forward-backward pass over all data batches in the passed DataLoader
    :param loader: torch.utils.data.DataLoader to use
    :param network: the network to pass data batches through
    :param device: the device that network is on
    :param backward_fn: if specified, the result of backward_fn(network(x), y) will be backpropagated for each batch.
                        Otherwise, no backward pass will be conducted.
    :param early_stop: if specified, execution will stop after the provided number of batches have
                       completed. Otherwise, all batches will be processed.
    :param pre_backward_hooks: List[function] of functions to be executed before backward() is called
    :param pre_zero_grad_hooks: List[function] of functions to be executed before zero_grad() is called
                                only if backward_fn is set and backward() is called
    """
    context = CustomContext()
    backward = True
    if backward_fn is None:
        backward = False
        context = torch.no_grad()

    with context:
        for itr, (i, x, y) in enumerate(tqdm(loader)):
            if early_stop and itr >= early_stop:
                return
            x, y = x.to(device), y.to(device)
            out = network(x)

            # call hook methods
            for fn in pre_backward_hooks:
                fn()

            if backward:
                backward_out = backward_fn(out, y)
                backward_out.backward()

                # call hook methods
                for fn in pre_zero_grad_hooks:
                    fn()

                network.zero_grad()


def flatten_activations(activations):
    # if outputs are conv featuremaps aggregate over spatial dims and sample dim
    if len(activations.shape) == 4:
        return activations.transpose(1, 0).flatten(start_dim=1, end_dim=3)
    # if outputs are vectors only need to aggregate over sample dim
    elif len(activations.shape) == 2:
        return activations.transpose(1, 0)
    else:
        raise TypeError('Output type unknown for Tensor: \n%s' % repr(activations))


class CustomContext:

    def __init__(self, enter_fns=[], exit_fns=[], handle_exc_vars=False):
        if not handle_exc_vars:
            new_exit_fns = []
            for fn in exit_fns:
                new_exit_fns += [self.wrap_exit_fn(fn)]
            for fn_ in new_exit_fns:
                fn_()
            exit_fns = new_exit_fns
        self.enter_fns = enter_fns
        self.exit_fns = exit_fns

    def __enter__(self):
        for fn in self.enter_fns:
            fn()

    def __exit__(self, *exc_vars):
        for fn in self.exit_fns:
            fn(*exc_vars)

    def __add__(self, other):
        return self.merge_context(other)

    @staticmethod
    def wrap_exit_fn(fn):
        def new_fn(*exc_vars):
            return fn()
        return new_fn

    def merge_context(self, context, inplace: bool = False):
        assert hasattr(context, '__enter__') and hasattr(context, '__exit__'), \
            "context object '%s' is missing '__enter__' or '__exit__' methods"
        # if context is CustomContext object, concatenate enter and exit functions directly
        if type(context) == __class__:
            enter_fns, exit_fns = context.enter_fns, context.exit_fns
        else:
            enter_fns, exit_fns = [context.__enter__], [context.__exit__]

        if inplace:
            self.enter_fns += enter_fns
            self.exit_fns += exit_fns
            return self
        else:
            return __class__(enter_fns=self.enter_fns + enter_fns,
                             exit_fns=self.exit_fns + exit_fns,
                             handle_exc_vars=True)


class Protocol:

    DEFAULT_PROTOCOL = None

    def __init__(self, **overwrite_protocol):
        self._set_proto_dict()

    def __iter__(self):
        return iter(self.proto_dict)

    def __getitem__(self, item):
        return self.proto_dict[item]

    def __setitem__(self, key, value):
        self.proto_dict[key] = value

    def __getattr__(self, item):
        return self.proto_dict[item]

    def __setattr__(self, key, value):
        if hasattr(self, 'proto_dict') and key in self.proto_dict:
            self.proto_dict[key] = value
        else:
            self.__dict__[key] = value

    def _set_proto_dict(self):
        if self.DEFAULT_PROTOCOL is None:
            raise TypeError("cannot create object of abstract class 'Protocol'")
        self.__dict__['proto_dict'] = dict(self.DEFAULT_PROTOCOL)

    def _add_from_json(self, json_file):
        protocol_dict = json.load(open(json_file, 'r'))
        for protocol, value in protocol_dict.items():
            self.proto_dict[protocol] = value

    def _add_from_namespace(self, namespace):
        for protocol, value in namespace.__dict__.items():
            if protocol in self.proto_dict:
                self.proto_dict[protocol] = value

    def overwrite_protocol(self, **overwrite_protocol):
        for protocol, value in overwrite_protocol.items():
            assert protocol in self.proto_dict, "cannot overwrite protocol '%s'. '%s' is not contained" \
                                                " in proto_dict." % (protocol, protocol)
            self.proto_dict[protocol] = value

    def keys(self):
        return self.proto_dict.keys()

    def values(self):
        return self.proto_dict.values()

    def items(self):
        return self.proto_dict.items()

