from copy import copy
from inspect import getfullargspec
from .helpers import CustomContext
from torch import Tensor
import torch
from torch.nn import Module


def increment_name(name):
    """
    Increments the number appended to the name and returns the resulting string
    :param name: the name to be incremented
    :return: the name with its numeric suffix incremented
    """
    if '_' in name and name.split('_')[-1].isnumeric():
        count = int(name.split('_')[-1]) + 1
        return '_'.join(name.split('_')[:-1]) + '_%d' % count
    return name + '_1'


def detach_hook(module, input, output):
    return output.detach()


def validate_hook_category(fn):
    valid = HookFunction.HOOK_TYPES + ['all']
    def wrapped_fn(*args, category='all', **kwargs):
        assert category in valid, '%s is not a valid hook category.' \
                                  ' Choices are: %s' % (category, ', '.join(valid))
        return fn(*args, category=category, **kwargs)

    return wrapped_fn


def _clone_cache(cache):
    ret = []
    for val in cache:
        if type(val) == torch.Tensor:
            ret += [val.clone()]
        else:
            ret += [val]
    return ret


class HookFunction:

    HOOK_TYPES = ['forward_hook', 'backward_hook', 'forward_pre_hook']

    def __init__(self, hook_fn, hook_type, name=None, modules=None, pass_by_pos=True, retain_forward_cache=False):
        assert hook_type in self.HOOK_TYPES, 'invalid hook type: %s' % hook_type
        self.pass_by_pos = pass_by_pos
        self.retain_forward_cache = retain_forward_cache
        if retain_forward_cache:
            assert hook_type == 'backward_hook', 'retain_forward_cache option is exclusively for backward hooks'
        if name is None:
            name = repr(hook_fn)
        self.name = name
        self.hook_type = hook_type
        self.function = hook_fn
        self.handles = []
        self.module_to_handle = {}
        if modules:
            self.register(*modules)

    def __call__(self, *args, **kwargs):
        if self.pass_by_pos:
            return self.function(*self._pass_params(*args, **kwargs))
        return self.function(**self._pass_params(*args, **kwargs))

    @property
    def _pos_arg_names(self):
        if self.hook_type == 'forward_pre_hook':
            return ['module', 'inputs']
        elif self.hook_type == 'forward_hook':
            return ['module', 'inputs', 'outputs']
        elif self.hook_type == 'backward_hook':
            if self.retain_forward_cache:
                return ['module', 'grad_in', 'grad_out', 'inputs', 'outputs']
            else:
                return ['module', 'grad_in', 'grad_out']

    def _args2kwargs(self, args):
        kwargs = {}
        pos_arg_names = self._pos_arg_names
        if len(args) != len(pos_arg_names):
            error_string = "invalid number of positional args passed to %s '%s' ." \
                           "Call signature should be (%s) but got %d positional args." % \
                           (self.hook_type, self.name, ', '.join(pos_arg_names), len(args))
            raise TypeError(error_string)
        for i, (arg_name, arg_val) in enumerate(zip(pos_arg_names, args)):
            kwargs[arg_name] = arg_val

        return kwargs

    def _kwargs2args(self, kwargs):
        pos_arg_names = self._pos_arg_names
        args = []
        try:
            for name in pos_arg_names:
                args += [kwargs[name]]
        except KeyError as e:
            raise KeyError('Error: the argument "%s" was not found in passed kwargs.' % e.args[0])
        return args

    def _pass_params(self, *args, **kwargs):
        if len(args) > 0:
            for k, v in self._args2kwargs(args).items():
                kwargs[k] = v
        if self.pass_by_pos:
            return self._kwargs2args(kwargs)
        fn = self.function
        argspec = getfullargspec(fn)
        if argspec.varkw is not None:
            return kwargs
        params = {}
        # if the method is an object method, remove the 'self' variable
        if 'bound method' in str(fn):
            argspec.args.pop(0)
        for var in argspec.args + argspec.kwonlyargs:
            if var not in kwargs:
                raise TypeError("hook method '%s' requested parameter '%s' that is unavailable. Available"
                                " parameters are [%s]." % (self.name, var,
                                                           ', '.join(argspec.args + argspec.kwonlyargs)))
            params[var] = kwargs[var]

        return params

    def register(self, *modules, activate=True, register_to_module=True):
        handles = []
        for module in modules:
            assert hasattr(module, 'name'), 'Module must be given name before hook registration'
            assert module not in self.module_to_handle, \
                'Hook function %s was already registered with module %s' % (self.name, module.name)
            name = self.name + '[' + module.name + ']'
            handle = HookHandle(self, module, name, activate=activate, register_to_module=register_to_module)
            self.module_to_handle[module] = handle
            handles += [handle]
        self.handles += handles
        return handles

    def deactivate_all_handles(self):
        for handle in self.handles:
            handle.deactivate()


class HookHandle:

    def __init__(self, hook_fn, module, name, activate=True, register_to_module=True):
        self.hook_fn = hook_fn
        self.register_to_module = register_to_module
        self.handle = None
        self.active = False
        self.module = module
        self.name = name
        if activate:
            self.activate()

    def __repr__(self):
        return '<%s %s registered to %s (%s)>' % (self.name, type(self), self.module.name,
                                                  'active' if self.is_active() else 'inactive')

    def is_active(self):
        return self.active

    def activate(self, raise_on_active=False):
        if self.is_active():
            if raise_on_active:
                raise AssertionError('Cannot activate hook: Hook is already active')
            return
        if self.register_to_module:
            register_fn = 'register_%s' % self.hook_fn.hook_type
            assert hasattr(self.module, register_fn), 'Module %s has no method %s' % (repr(self.module), register_fn)
            self.handle = getattr(self.module, register_fn)(self.hook_fn.function)
        self.active = True

    def deactivate(self, raise_on_inactive=False):
        if not self.is_active():
            if raise_on_inactive:
                raise AssertionError('Cannot deactivate hook: Hook is already inactive')
            return
        if self.register_to_module:
            self.handle.remove()
            self.handle = None
        self.active = False

    def remove(self):
        if self.handle:
            self.handle.remove()
        self.handle = None

    def set_name(self, name):
        self.name = name


class HookManager:
    """
    Class for centralized handling of PyTorch module hooks
    """

    # TODO modify cache clearing to allow for recursive caching of module vars
    def __init__(self, wrap_calls=True, recursion_depth=0):
        # Lookup tables
        self.hook_fns = set()
        self.modules = set()
        self.name_to_hookfn = {}  # HookFunction.name -> HookFunction
        self.name_to_hookhandle = {}  # HookHandle.name -> HookHandle
        self.module_to_hookhandle = {}  # HookHandle.module -> List[HookHandle]
        self.name_to_module = {}  # Module.name -> Module
        self.function_to_hookfn = {}  # HookFunction.function -> HookFunction

        self.wrap_calls = wrap_calls  # whether to call active hooks from a unified HookManager base hook
        self.base_forward_hook_fn = None
        self.base_forward_pre_hook_fn = None
        if wrap_calls:
            self.base_forward_hook_fn = HookFunction(self._forward_hook_base, 'forward_hook',
                                                     name='HookManager._forward_hook_base')
            self.base_forward_pre_hook_fn = HookFunction(self._forward_pre_hook_base, 'forward_pre_hook',
                                                         name='HookManager._forward_pre_hook_base')

        # tracks which module hooks require forward pass data to be cached until backward pass
        self._retain_forward_cache = {}

        # levels of child-modules at which hooks will be recursively set
        self.max_recursion_depth = recursion_depth
        self.num_module_inputs = {}
        self.num_module_outputs = {}
        self.valid_module_params = {}  # caches valid (not None) params whose gradients will be cached during backward
        self.input_cache = {}
        self.output_cache = {}
        self.param_grad_cache = {}
        self.input_grad_cache = {}  # caches intermediate gradient tensors of module inputs
        self.output_grad_cache = {}  # caches intermediate gradient tensors of module outputs

        self.input_hooks = {}
        self.param_hooks = {}
        self.output_hooks = {}

    def __del__(self):
        self.cleanup()

    def cleanup(self):
        self.num_module_inputs = {}
        self.num_module_outputs = {}
        self.valid_module_params = {}
        self.input_cache = {}
        self.output_cache = {}
        self.param_grad_cache = {}
        self.input_grad_cache = {}
        self.output_grad_cache = {}

        self.deactivate_all_hooks()

        self.hook_fns = set()
        self.modules = set()
        self.name_to_hookfn = {}
        self.name_to_hookhandle = {}
        self.module_to_hookhandle = {}
        self.name_to_module = {}
        self.function_to_hookfn = {}

        self._clear_tensor_hooks()

    def _clear_module_input_hooks(self, module):
        for removable_handle in self.input_hooks[module]:
            if removable_handle is not None:
                removable_handle.remove()
        self.input_hooks[module] = None

    def _clear_module_param_hook(self, module, param):
        self.param_hooks[module][param].remove()
        self.param_hooks[module][param] = None

    def _clear_module_param_hooks(self, module):
        for param in self.param_hooks[module]:
            self._clear_module_param_hook(module, param)
        self.param_hooks[module] = {}

    def _clear_module_output_hooks(self, module):
        for removable_handle in self.output_hooks[module]:
            if removable_handle is not None:
                removable_handle.remove()
        self.output_hooks[module] = None

    def _clear_module_tensor_hooks(self, module):
        self._clear_module_input_hooks(module)
        self._clear_module_output_hooks(module)
        self._clear_module_param_hooks(module)

    def _clear_tensor_hooks(self):
        for module in self.modules:
            self._clear_module_tensor_hooks(module)

    def _clear_forward_module_cache(self, *modules):
        for module in modules:
            self.input_cache[module.name] = None
            self.output_cache[module.name] = None

    def _clear_backward_module_cache(self, *modules):
        for module in modules:
            self.param_grad_cache[module.name] = {}
            self.input_grad_cache[module.name] = None
            self.output_grad_cache[module.name] = None

    def _init_module_cache(self, *modules):
        self._clear_forward_module_cache(*modules)
        self._clear_backward_module_cache(*modules)
        for module in modules:
            self.num_module_inputs[module.name] = -1
            self.num_module_outputs[module.name] = -1
            """self.valid_module_params[module.name] = []"""

    def _register_base_hooks(self, *modules, activate=False):
        assert self.wrap_calls, 'base hooks should only be registered when wrap_calls is set'
        self.base_forward_hook_fn.register(*modules,
                                           activate=activate,
                                           register_to_module=True)
        self.base_forward_pre_hook_fn.register(*modules,
                                               activate=activate,
                                               register_to_module=True)
        # TODO register recursively depending on self.recursion_depth
        self._init_module_cache(*modules)

    def _activate_base_hooks(self, *modules):
        for module in modules:
            for h in self.get_module_hooks(module,
                                           include_active=False,
                                           include_base_hooks=True):
                h.activate()

    def _deactivate_base_hooks(self, *modules):
        for module in modules:
            for h in self.get_module_hooks(module,
                                           include_inactive=False,
                                           include_base_hooks=True):
                h.deactivate()

    def _get_hook_params_forward(self, module, recursion_depth=0):
        ret_dict = {
            'module': module,
            'inputs': self.input_cache[module.name],
            'outputs': self.output_cache[module.name]
        }
        """if recursion_depth < self.max_recursion_depth:
            for name, module in module._modules.items():
                ret_dict[name] = self._get_hook_params_forward(module, recursion_depth=recursion_depth + 1)"""

        return ret_dict

    def _get_hook_params_backward(self, module, retain_forward_cache=False, recursion_depth=0):
        # clone tensors before passing them so backward pass is not disrupted by hook functions
        ret_dict = {
            'module': module,
            'grad_in': _clone_cache(self.input_grad_cache[module.name]),
            'grad_out': _clone_cache(self.output_grad_cache[module.name]),
        }
        if retain_forward_cache:
            ret_dict['inputs'] = _clone_cache(self.input_cache[module.name])
            ret_dict['outputs'] = _clone_cache(self.output_cache[module.name])

        """for param_name, _ in module.named_parameters():
            if param_name not in self.valid_module_params[module.name]:
                val = None
            else:
                val = self.param_grad_cache[module.name][param_name]
            ret_dict['grad_%s' % param_name] = val
        if recursion_depth < self.max_recursion_depth:
            for name, module in module._modules.items():
                ret_dict[name] = self._get_hook_params_backward(module, recursion_depth=recursion_depth + 1)"""

        return ret_dict

    def _execute_forward_pre_hooks(self, module, inputs):
        for handle in self.get_module_hooks(module, category='forward_pre_hook', include_inactive=False):
            hook_fn = handle.hook_fn
            ret = hook_fn(module=module, inputs=inputs)
            if ret is not None:
                assert type(ret) == tuple, 'when altering module inputs, must return result as a tuple'
                inputs = ret

        return inputs

    def _execute_forward_hooks(self, module):
        param_dict = self._get_hook_params_forward(module)
        outputs = param_dict.pop('outputs')
        for handle in self.get_module_hooks(module, category='forward_hook', include_inactive=False):
            hook_fn = handle.hook_fn
            ret = hook_fn(outputs=outputs, **param_dict)
            if ret is not None:
                assert type(ret) == tuple, 'when altering module outputs, must return result as a tuple'
                outputs = ret

        return outputs

    # TODO allow modification of grad_in by backward_hooks
    def _execute_backward_hooks(self, module):
        for handle in self.get_module_hooks(module, category='backward_hook', include_inactive=False):
            hook_fn = handle.hook_fn
            param_dict = self._get_hook_params_backward(module, retain_forward_cache=hook_fn.retain_forward_cache)
            hook_fn(**param_dict)

    def _grads_resolved(self, hook_list, grad_cache):
        map_none = lambda x: 0 if x is None else 1
        return sum(map(map_none, hook_list)) == sum(map(map_none, grad_cache))

    def _have_all_gradients(self, module):
        have_input_grads = self._grads_resolved(self.input_hooks[module], self.input_grad_cache[module.name])
        #print('Gradient computed for %d/%d module inputs' % (len(self.input_grad_cache[module.name]), self.num_module_inputs[module.name]))
        have_param_grads = self._grads_resolved(self.param_hooks[module].values(),
                                                self.param_grad_cache[module.name].values())
        #print('Gradient not computed for params: %s' % ', '.join([p for p in self.valid_module_params[module.name]
        #                                                          if p not in self.param_grad_cache[module.name]]))
        return have_input_grads and have_param_grads

    def _make_backward_hook_input(self, module, input_pos):

        def backward_hook(grad):
            #print('in input backward hook for %s' % module.name)
            self.input_grad_cache[module.name][input_pos] = grad

            # if all input gradients have been accumulated, initiate module backward hooks
            if self._have_all_gradients(module):
                self._initiate_backward_hook(module)

        return backward_hook

    def _make_backward_hook_param(self, module, param_name):

        def backward_hook(grad):
            #print('in param backward hook for %s of %s' % (param_name, module.name))
            self.param_grad_cache[module.name][param_name] = grad

            # if all input gradients have been accumulated, initiate module backward hooks
            if self._have_all_gradients(module):
                self._initiate_backward_hook(module)

        return backward_hook

    def _make_backward_hook_output(self, module, output_pos):

        def backward_hook(grad):
            #print('in output backward hook for %s' % module.name)
            self.output_grad_cache[module.name][output_pos] = grad
            """self._clear_module_output_hooks(module)"""

        return backward_hook

    def _forward_pre_hook_base(self, module, inputs):
        ret = self._execute_forward_pre_hooks(module, inputs)
        if ret is not None:
            assert type(ret) == tuple, 'when altering module inputs, must return result as a tuple'
            inputs = ret
        self.input_cache[module.name] = inputs
        self.num_module_inputs[module.name] = len(inputs)

        # if grad is enabled, setup backward pass
        if torch.is_grad_enabled():
            # instantiate hooks list and grad cache
            self.input_hooks[module] = [None] * len(inputs)
            self.input_grad_cache[module.name] = [None] * len(inputs)

            for i, inp in enumerate(inputs):
                """if not inp.requires_grad:
                    inp.requires_grad = True"""
                if type(inp) == torch.Tensor and inp.requires_grad:
                    # maintain correspondence between input items and grad_in items
                    removable_handle = inp.register_hook(self._make_backward_hook_input(module, i))
                    self.input_hooks[module][i] = removable_handle
        return ret

    def _forward_hook_base(self, module, inputs, outputs):
        return_tensor = False
        if type(outputs) == torch.Tensor:
            return_tensor = True
            outputs = (outputs,)

        self.output_cache[module.name] = outputs
        ret = self._execute_forward_hooks(module)
        if ret is not None:
            assert type(ret) == tuple, 'when altering module outputs, must return result as a tuple'
            outputs = ret
        self.output_cache[module.name] = outputs
        self.num_module_outputs[module.name] = len(outputs)

        # if grad is enabled setup backward pass
        if torch.is_grad_enabled():
            """self.param_hooks[module] = {}"""
            if module in self.param_hooks:
                assert len(self.param_hooks[module]) == 0, \
                    'param hooks for %s were not cleaned up during last backward pass' % module.name
                assert len(self.param_grad_cache[module.name]) == 0, \
                    'param grad cache for %s was not cleaned up during last backward pass' % module.name
            else:
                self.param_hooks[module] = {}
                self.param_grad_cache[module.name] = {}
            self.output_hooks[module] = [None] * len(outputs)
            self.output_grad_cache[module.name] = [None] * len(outputs)

            for i, out in enumerate(outputs):
                if type(out) == torch.Tensor and out.requires_grad:
                    # maintain correspondence between output items and grad_out items
                    removable_handle = out.register_hook(self._make_backward_hook_output(module, i))
                    self.output_hooks[module][i] = removable_handle

            for name, param in module.named_parameters():
                if param is not None and param.requires_grad:
                    """self.valid_module_params[module.name] += [name]"""
                    removable_handle = param.register_hook(self._make_backward_hook_param(module, name))
                    self.param_hooks[module][name] = removable_handle
                else:
                    self.param_hooks[module][name] = None

        if len(self._retain_forward_cache[module]) == 0:
            self._clear_forward_module_cache(module)

        if return_tensor:
            ret, = ret

        return ret

    def _initiate_backward_hook(self, module):
        self._execute_backward_hooks(module)
        self._init_module_cache(module)
        self._clear_module_tensor_hooks(module)

    def add_hook_fn(self, *hook_fns):
        for hook_fn in hook_fns:
            self.hook_fns = self.hook_fns.union({hook_fn})
            self.name_to_hookfn[hook_fn.name] = hook_fn
            self.function_to_hookfn[hook_fn.function] = hook_fn

    def add_hook_handle(self, *hook_handles):
        for handle in hook_handles:
            module = handle.module
            assert handle not in self.module_to_hookhandle[module], \
                "attempted to add duplicate HookHandle '%s'" % handle.name
            self.module_to_hookhandle[module] += [handle]
            self.name_to_hookhandle[handle.name] = handle

    def register_hook(self, hook_type, function, *modules, hook_fn_name=None,
                      activate=True, retain_forward_cache=False, **named_modules):
        # Check if HookFunction obj has already been created for the given function
        if function in self.function_to_hookfn:
            hook_fn = self.function_to_hookfn[function]
        else:
            if hook_fn_name in self.name_to_hookfn:
                hook_fn_name = increment_name(hook_fn_name)
            hook_fn = HookFunction(function, hook_type, name=hook_fn_name,
                                   retain_forward_cache=retain_forward_cache)
            self.add_hook_fn(hook_fn)

        # Check if modules have already been registered with another hook function
        named_modules = [(None, module) for module in modules] + list(named_modules.items())
        new_modules = []
        old_modules = []
        for module_name, module in named_modules:
            if module not in self.modules:
                self.modules = self.modules.union({module})
                self.module_to_hookhandle[module] = []
                if module_name is None:
                    module_name = repr(module)
                if module_name in self.name_to_module:
                    module_name = increment_name(module_name)
                self.name_to_module[module_name] = module
                module.name = module_name

                new_modules += [module]
                self._retain_forward_cache[module] = []
            else:
                old_modules += [module]

        modules = [m for name, m in named_modules]

        # if using base_hooks, register base_hooks on all new modules and activate base_hooks on all old modules
        if self.wrap_calls:
            self._register_base_hooks(*new_modules, activate=activate)
            if activate:
                self._activate_base_hooks(*old_modules)

        # Make sure module names were assigned properly
        assert all([hasattr(m, 'name') for name, m in named_modules])

        # Create hook handle
        handles = hook_fn.register(*modules,
                                   activate=activate,
                                   register_to_module=not self.wrap_calls)

        # TODO move wrap_calls statement above and merge with this one
        if self.wrap_calls and activate and retain_forward_cache:
            for module, handle in zip(modules, handles):
                self._retain_forward_cache[module] += [handle]

        # Update hook handle lookup tables
        self.add_hook_handle(*handles)

    def register_forward_hook(self, function, *modules, hook_fn_name=None,
                              activate=True, **named_modules):
        self.register_hook('forward_hook', function, *modules,
                           hook_fn_name=hook_fn_name, activate=activate, **named_modules)

    def register_backward_hook(self, function, *modules, hook_fn_name=None,
                               activate=True, retain_forward_cache=False, **named_modules):
        self.register_hook('backward_hook', function, *modules,
                           hook_fn_name=hook_fn_name, activate=activate, retain_forward_cache=retain_forward_cache,
                           **named_modules)

    def register_forward_pre_hook(self, function, *modules, hook_fn_name=None,
                                  activate=True, **named_modules):
        self.register_hook('forward_pre_hook', function, *modules,
                           hook_fn_name=hook_fn_name, activate=activate, **named_modules)

    def _remove_base_hook(self, handle):
        # remove from hook fn's lookup tables
        handle.hook_fn.handles.remove(handle)
        handle.hook_fn.module_to_handle.pop(handle.module)

    def remove_hook(self, handle):
        handle.deactivate()
        # remove from hook manager's lookup tables
        self.name_to_hookhandle.pop(handle.name)
        self.module_to_hookhandle[handle.module].remove(handle)

        # remove from hook fn's lookup tables
        handle.hook_fn.handles.remove(handle)
        handle.hook_fn.module_to_handle.pop(handle.module)

        # deactivate base hooks if module is now inactive
        if self.wrap_calls and len(self.module_to_hookhandle[handle.module]) == 0:
            self._deactivate_base_hooks(handle.module)
        # update _retain_forward_cache
        if handle in self._retain_forward_cache[handle.module]:
            self._retain_forward_cache[handle.module].remove(handle)

    def remove_hook_by_name(self, handle_name):
        self.remove_hook(self.name_to_hookhandle[handle_name])

    def remove_hook_function(self, function):
        hook_fn = self.function_to_hookfn[function]
        for handle in copy(hook_fn.handles):
            self.remove_hook(handle)
        # remove from hook manager's lookup tables
        self.name_to_hookfn.pop(hook_fn.name)
        self.function_to_hookfn.pop(function)

    def remove_hook_function_by_name(self, hook_fn_name):
        self.remove_hook_function(self.name_to_hookfn[hook_fn_name])

    def remove_module(self, module):
        for handle in copy(self.module_to_hookhandle[module]):
            self.remove_hook(handle)
        # remove from hook manager lookup tables
        self.name_to_module.pop(module.name)
        self.module_to_hookhandle.pop(module)
        self.modules.remove(module)
        # remove corresponding base hook handles
        for hook_fn in [self.base_forward_hook_fn, self.base_forward_pre_hook_fn]:
            if module in hook_fn.module_to_handle:
                handle = hook_fn.module_to_handle.pop(module)
                hook_fn.handles.remove(handle)

    def remove_module_by_name(self, module_name):
        self.remove_module(self.name_to_module[module_name])

    def is_module_hooked(self, module):
        active_hooks = list(self.get_module_hooks(module, include_inactive=False, include_base_hooks=False))
        return len(active_hooks) > 0

    def is_module_hooked_backward(self, module):
        active_hooks = list(self.get_module_hooks(module, include_inactive=False, include_base_hooks=False,
                                                  category='backward_hook'))
        return len(active_hooks) > 0

    def activate_hook_by_name(self, hook_name):
        handle = self.name_to_hookhandle[hook_name]
        # if we will be activating a previously inactive handle that requires forward cache retention, add to list
        if (not handle.is_active()) and handle.hook_fn.retain_forward_cache:
            self._retain_forward_cache[handle.module] += [handle]
        handle.activate()
        # activate base_hooks if we are wrapping calls
        if self.wrap_calls:
            self._activate_base_hooks(handle.module)

    def deactivate_hook_by_name(self, hook_name):
        handle = self.name_to_hookhandle[hook_name]
        # if we are removing a handle that is requiring forward cache retention, remove from list
        if handle in self._retain_forward_cache[handle.module]:
            self._retain_forward_cache[handle.module].remove(handle)
        handle.deactivate()
        # if we have deactivated the last active hook for a module, deactivate base_hooks as well
        if self.wrap_calls:
            if not self.is_module_hooked(handle.module):
                self._deactivate_base_hooks(handle.module)

    def get_module_hooks(self, module, hook_types=[], category='all', include_active=True, include_inactive=True,
                         include_base_hooks=False):
        if include_base_hooks:
            handles = [hook_fn.module_to_handle[module] for hook_fn in (self.base_forward_pre_hook_fn,
                                                                        self.base_forward_hook_fn)]
        else:
            handles = self.module_to_hookhandle[module]
        for h in handles:
            if len(hook_types) > 0 and h.hook_fn.function not in hook_types:
                continue
            if category != 'all' and category != h.hook_fn.hook_type:
                continue
            if not include_inactive and not h.is_active():
                continue
            if not include_active and h.is_active():
                continue
            yield h

    def get_module_hooks_by_name(self, module_name, hook_types=[], **kwargs):
        return self.get_module_hooks(self.name_to_module[module_name], hook_types=hook_types, **kwargs)

    def activate_module_hooks(self, *modules, hook_types=[], category='all'):
        for module in modules:
            for h in self.get_module_hooks(module, hook_types=hook_types, include_active=False, category=category):
                # if activating a previously inactive handle that requires forward cache retention, add to list
                if h.hook_fn.retain_forward_cache:
                    self._retain_forward_cache[module] += [h]
                h.activate()
        if self.wrap_calls:
            self._activate_base_hooks(*modules)

    def activate_module_hooks_by_name(self, *module_names, hook_types=[], category='all'):
        for module_name in module_names:
            for h in self.get_module_hooks_by_name(module_name, hook_types=hook_types, include_active=False,
                                                   category=category):
                # if activating a previously inactive handle that requires forward cache retention, add to list
                if h.hook_fn.retain_forward_cache:
                    self._retain_forward_cache[h.module] += [h]
                h.activate()
        if self.wrap_calls:
            self._activate_base_hooks(*[self.name_to_module[module_name] for module_name in module_names])

    def deactivate_module_hooks(self, *modules, hook_types=[], category='all'):
        inactive_modules = []
        for module in modules:
            for h in self.get_module_hooks(module, hook_types=hook_types, include_inactive=False, category=category):
                # if we are removing a handle that is requiring forward cache retention, remove from list
                if h in self._retain_forward_cache[module]:
                    self._retain_forward_cache[module].remove(h)
                h.deactivate()
            if self.wrap_calls and not self.is_module_hooked(module):
                inactive_modules += [module]
        if self.wrap_calls:
            self._deactivate_base_hooks(*inactive_modules)

    def deactivate_module_hooks_by_name(self, *module_names, hook_types=[], category='all'):
        inactive_modules = []
        for module_name in module_names:
            for h in self.get_module_hooks_by_name(module_name, hook_types=hook_types, include_inactive=False,
                                                   category=category):
                # if we are removing a handle that is requiring forward cache retention, remove from list
                if h in self._retain_forward_cache[h.module]:
                    self._retain_forward_cache[h.module].remove(h)
                h.deactivate()
            module = self.name_to_module[module_name]
            if self.wrap_calls and not self.is_module_hooked(module):
                inactive_modules += [module]
        if self.wrap_calls:
            self._deactivate_base_hooks(*inactive_modules)

    def activate_all_hooks(self, hook_types=[], category='all'):
        self.activate_module_hooks(*self.modules, hook_types=hook_types, category=category)

    def deactivate_all_hooks(self, hook_types=[], category='all'):
        self.deactivate_module_hooks(*self.modules, hook_types=hook_types, category=category)

    ########################  Context Management  #######################################

    def hook_module_context(self, *modules, hook_types=[], add_enter_fns=[], add_exit_fns=[], category='all'):
        enter_fns = [lambda: self.activate_module_hooks(*modules, hook_types=hook_types, category=category)]
        exit_fns = [lambda: self.deactivate_module_hooks(*modules, hook_types=hook_types, category=category)]
        enter_fns = list(add_enter_fns) + enter_fns
        exit_fns = exit_fns + list(add_exit_fns)
        return CustomContext(enter_fns=enter_fns, exit_fns=exit_fns)

    def hook_module_context_by_name(self, *module_names, hook_types=[], add_enter_fns=[], add_exit_fns=[],
                                    category='all'):
        modules = [self.name_to_module[module_name] for module_name in module_names]
        return self.hook_module_context(*modules,
                                        hook_types=hook_types,
                                        add_enter_fns=add_enter_fns,
                                        add_exit_fns=add_exit_fns,
                                        category=category)

    @validate_hook_category
    def hook_all_context(self, hook_types=[], add_enter_fns=[], add_exit_fns=[], category='all'):
        return self.hook_module_context(*self.modules,
                                        hook_types=hook_types,
                                        add_enter_fns=add_enter_fns,
                                        add_exit_fns=add_exit_fns,
                                        category=category)
