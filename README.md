![logo](logo/logo-words.png "TackleBox")

A framework for improved handling of PyTorch module hooks

## Installation
```shell
pip install tacklebox
```
## Why TackleBox?
PyTorch module hooks are useful for a number of reasons. Debugging module behavior,
quickly altering processing or gradient flow, and studying intermediate
activations are just a few utilities. Module hooks are a powerful tool,
but using them requires keeping track of hook functions,
the modules they are registered to, and the handles that let you remove those
hooks.

#### Improved organization

TackleBox maintains a record of all hooks that have been registered, 
the modules they were registered with, and allows you to
deactivate and reactivate any previously registered hooks on the fly.

#### Backward hook consistency and backward-compatibility

PyTorch autograd can lead to inconsistencies in what gradients are
served to backward hooks registered with PyTorch's
``module.register_backward_hook`` method. This inconsistency has ultimately led
to [the method's deprecation](https://github.com/pytorch/pytorch/pull/46163) altogether. 

TackleBox reimplements module backward hook registration using Tensor
backward hooks registered on a module's inputs and outputs during the
forward pass. This allows us to establish correspondence between
the gradient tensors received by the backward hook and
the input/output tensors received by the forward hook.

With TackleBox you can continue to use module backward hooks and benefit from consistency in the ordering 
of gradient tensors served.

## Quickstart

#### Defining hooks
Hook functions must follow the call signature of the corresponding hook type: 
```python
def my_forward_hook(module, inputs, outputs):
    print('Finished forward pass for module %s' % module.name)
    return outputs

def my_forward_pre_hook(module, inputs):
    print('Beginning forward pass for module %s' % module.name)
    return inputs

def my_backward_hook(module, grad_in, grad_out):
    print('Finished backward pass for module %s' % module.name)
```
The ``inputs`` and ``outputs`` passed to forward hooks and forward pre-hooks are
tuples containing all tensors passed to that module and output by that module, respectively.

``grad_in`` and ``grad_out`` are tuples of the same length as ``inputs`` and ``outputs``,
respectively. The element at each position in ``grad_in`` or ``grad_out``  is the gradient
w.r.t. the input or output
at the same position in ``inputs`` or ``outputs``, respectively.

Forward hooks and forward-pre hooks may **optionally** return a tuple of the same length
with 1 or more of the tensors in it altered. These new inputs or outputs will be passed
to the module's forward pass (in the case of the forward pre-hook)
or returned as the module's output (in the case of the forward hook)

#### Order of execution
Forward hooks
will be called at the end of a module's forward pass, forward pre-hooks will be called
immediately before a module's forward pass and backward hooks will be called at the end 
of a module's backward pass.

#### Registering hooks
Rather than registering hooks on a module directly, hook functions are passed to the hook manager
along with any modules that you would like it to be registered on. Hook functions should be passed
to the corresponding registration function and assigned an id using kwargs:
```python
from tacklebox.hook_management import HookManager

hookmngr = HookManager()

# register a forward hook on my_module, other_module, etc.
hookmngr.register_forward_hook(my_forward_hook,
                               my_module_name=my_module,
                               other_module_name=other_module,
                               **more_named_modules)

# register a forward pre-hook on my_module, other_module, etc.
hookmngr.register_forward_pre_hook(my_forward_pre_hook,
                                   my_module_name=my_module,
                                   other_module_name=other_module,
                                   **more_named_modules)

# register a backward hook on my_module, other_module, etc.
hookmngr.register_backward_hook(my_backward_hook,
                                my_module_name=my_module,
                                other_module_name=other_module,
                                **more_named_modules)
```

#### Activating hooks
Once registered, your hooks can be activated and deactivated on the fly,
using a variety of filtering options. By default, hooks are activated upon registration.
To register a hook without immediately activating it, pass ``activate=False``:
```python
hookmngr.register_forward_hook(my_forward_hook, my_module_id=my_module,
                               activate=False)
```

Following registration, you can select groups of hooks to activate or deactivate using
several different filters:
```python
# activate/deactivate all hooks
hookmngr.activate_all_hooks()
hookmngr.deactivate_all_hooks()

# filter by module:
# activate/deactivate all hooks registered to my_module, other_module, etc.
hookmngr.activate_module_hooks(my_module, other_module, *more_modules)
hookmngr.deactivate_module_hooks(my_module, other_module, *more_modules)

# filter by function:
# activate/deactivate my_forward_hook on all modules it's been registered with
hookmngr.activate_all_hook(hook_types=[my_forward_hook])
hookmngr.deactivate_all_hooks(hook_types=[my_forward_hook]) 

# filter by hook category
# activate/deactivate all forward hooks that have been registered
hookmngr.activate_all_hooks(category='forward_hook')
hookmngr.deactivate_all_hooks(category='forward_hook')
```
``activate_module_hooks`` and ``deactivate_module_hooks`` accept an unpacked, variable-length
array of modules to filter by. These methods can take ``hook_types`` and ``category`` kwargs,
as well, allowing filtering by module, hook function and category in the same call.

The ``hook_types`` kwarg accepts a variable-length array of functions. Only hooks
corresponding to one of the passed functions will be activated.

The ``category`` kwarg accepts the following options:

- "all"
- "forward_hook"
- "forward_pre_hook"
- "backward_hook"

#### Hook activation with python contexts
TackleBox additionally provides python contexts that enable activation and deactivation
of module hooks with a single line of code:
```python
with hookmngr.hook_all_context():
    # all hooks active
    ...
# all hooks inactive

with hookmngr.hook_module_context(my_module, other_module, *more_modules):
    # hooks on my_module, other_module, etc. are active
    ...
# hooks on my_module, other_module, etc. are inactive
```
Both the above context methods accept the same kwargs for filtering as the activate and deactivate
methods (ie. ``hook_types`` and ``category``).

## Tutorials and walkthroughs
For further reference, see "Hook Management - part 1.ipynb" and
"Hook Management - part 2.ipynb" or checkout the [website](https://isaacrehg.com/tacklebox/)
for video
walkthroughs.
