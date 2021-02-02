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

With TackleBox you can continue to use module backward hooks, even with 
older PyTorch versions ( < 1.8.0 ), and benefit from consistency in the ordering 
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
Using PyTorch module hooks, hook registration might look something like this:
```python
my_handle = my_module.register_forward_hook(my_forward_hook)
other_handle = other_module.register_forward_hook(my_forward_hook)
...
```
In this case you must maintain a hook handle for each new module and hook function
you decide to register.

With TackleBox, rather than registering hooks on a module directly, hook functions are passed to the hook manager
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
Note that there is no need to maintain any additional references, other than
that of the hook manager.
#### Activating hooks
Once registered, your hooks can be activated and deactivated on the fly,
using a variety of filtering options. By default, hooks are activated upon registration.
To register a hook without immediately activating it, pass ``activate=False``:
```python
hookmngr.register_forward_hook(my_forward_hook, my_module_name=my_module,
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

#### *HookFunction* and *HookHandle* objects
When a new hook function is registered on a module, the function is wrapped in 
a ``HookFunction`` object. HookFunctions contain the function to be called
at the corresponding entry point as well as a dictionary of modules that it
has been registered to, the corresponding handle for each

Each registration event yields a handle. TackleBox represents these handles as
`HookHandle` objects that maintain references to the corresponding module and 
HookFunction involved in the registration event. ``HookHandle.activate()``
and ``HookHandle.deactivate()`` may be used to activate and deactivate the
corresponding hook function on the corresponding module.
```python
# access the HookHandle obtained from registering hook_function on my_module
hook_handle = hook_function.module_to_handle[my_module]

hook_handle.module  #  = my_module
hook_handle.hook_fn  # = hook_function

# activate/deactivate hook_function for my_module
hook_handle.activate()
hook_handle.deactivate()
```

#### Looking-up registered hooks
The hook manager maintains lookup tables for all HookFunctions, modules and 
HookHandles that have been registered. The user can access these objects using
their id. We saw how module id is assigned when registering a module in the previous
examples. Registered modules can be accessed with:
```python
hookmngr.name_to_module['my_module_name']
```

HookFunction ids can also be assigned during registration by using the 
``hook_fn_name`` kwarg:
```python
hookmngr.register_forward_hook(my_forward_hook, hook_fn_name='my_forward_hook')
```
If no id is passed, the HookFunction will be assigned an id using ``repr(function)``
to convert the passed function to a string.

Registered HookFunctions can be accessed by id:
```python
hookmngr.name_to_hookfn['my_forward_hook']
```

When hook functions are registered to modules by the hook manager, the resulting
HookHandle is given an id of the form `my_hook[my_module]` where ``my_hook`` is the 
id of the HookFunction and ``my_module`` is the id of the module registered to.

Specific handles can be accessed using this id:
```python
# access the HookHandle obtained from registering my_forward_hook on my_module
hookmngr.name_to_hookhandle['my_forward_hook[my_module]']
```
This lookup provides an easy way to visualize all currently registered hook across
all modules.

#### Removing hooks
Removing a hook will deactivate it then purge it from all records maintained by the hook manager.
This means that later calls to activate hooks will be unable to reactivate it.
Hooks can be removed by hook function, module or
hook handle (a module, hook function pair):
```python
hookmngr.remove_hook_function(my_forward_hook)

hookmngr.remove_module_by_name('my_module')

hookmngr.remove_hook_by_name('my_forward_hook[my_module]')
```
This lets us remove registered hooks with varying degrees of selectivity, using
a single line code.

Using the native PyTorch module hook registration, hook removal requires
iteration over maintained handles:

```python
my_handle.remove()
other_handle.remove()
...
```
With TackleBox you need not worry about module handles. Register and remove hooks
to and from groups of modules all at once, filtering the set of active hooks
at any point during experimentation. This is the power of TackleBox.

## Tutorials and walkthroughs
For further reference, see [Hook Management - part 1.ipynb](Hook%20Management%20-%20part%201.ipynb) and
[Hook Management - part 2.ipynb](Hook%20Management%20-%20part%202.ipynb). Before running, install the notebooks' dependencies with

```shell
pip install -r requirements.txt
```

You can also checkout the [website](https://isaacrehg.com/tacklebox/)
for video
walkthroughs.
