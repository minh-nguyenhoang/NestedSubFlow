#----------------------------------------------------------------------------------------------------------------------------------------------
# This part is takem from memcnn https://github.com/silvandeleemput/memcnn/tree/master 
# This is used for any reversible module to save memory during training
#----------------------------------------------------------------------------------------------------------------------------------------------

import functools
import torch
import torch.nn as nn
import numpy as np

try:
    from torch.cuda.amp import custom_fwd, custom_bwd
except ModuleNotFoundError:
    # custom_fwd is a decorator that may or may not be used with arguments, following
    # https://github.com/dabeaz/python-cookbook/tree/master/src/9/defining_a_decorator_that_takes_an_optional_argument.
    # this works:
    #     @custom_fwd
    #     def forward(...):
    # this also works:
    #     @custom_fwd(cast_inputs=torch.float)
    #     def forward(...):
    def custom_fwd(fwd=None, *, cast_inputs=None):
        """
        Helper decorator for ``forward`` methods of custom autograd functions (subclasses of
        :class:`torch.autograd.Function`).  See the :ref:`example page<amp-custom-examples>` for more detail.

        Args:
            cast_inputs (:class:`torch.dtype` or None, optional, default=None):  If not ``None``,
                when ``forward`` runs in an autocast-enabled region, casts incoming
                floating-point CUDA Tensors to the target dtype (non-floating-point Tensors are not affected),
                then executes ``forward`` with autocast disabled.
                If ``None``, ``forward``'s internal ops execute with the current autocast state.

        .. note::
            If the decorated ``forward`` is called outside an autocast-enabled region,
            :func:`custom_fwd<custom_fwd>` is a no-op and ``cast_inputs`` has no effect.
        """
        if fwd is None:
            return functools.partial(custom_fwd, cast_inputs=cast_inputs)
        @functools.wraps(fwd)
        def decorate_fwd(*args, **kwargs):
                return fwd(*args, **kwargs)
        return decorate_fwd

    def custom_bwd(bwd):
        """
        Helper decorator for backward methods of custom autograd functions (subclasses of
        :class:`torch.autograd.Function`).
        Ensures that ``backward`` executes with the same autocast state as ``forward``.
        See the :ref:`example page<amp-custom-examples>` for more detail.
        """
        @functools.wraps(bwd)
        def decorate_bwd(*args, **kwargs):
                return bwd(*args, **kwargs)
        return decorate_bwd


###############################################################################################################################################
###############################################################################################################################################


class InvertibleCheckpointFunction(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, fn, fn_inverse, keep_input, num_bwd_passes, preserve_rng_state, num_inputs, *inputs_and_weights):
        # store in context
        ctx.fn = fn
        ctx.fn_inverse = fn_inverse
        ctx.keep_input = keep_input
        ctx.weights = inputs_and_weights[num_inputs:]
        ctx.num_bwd_passes = num_bwd_passes
        ctx.preserve_rng_state = preserve_rng_state
        ctx.num_inputs = num_inputs
        inputs = inputs_and_weights[:num_inputs]

        if preserve_rng_state:
            ctx.fwd_cpu_state = torch.get_rng_state()
            # Don't eagerly initialize the cuda context by accident.
            # (If the user intends that the context is initialized later, within their
            # run_function, we SHOULD actually stash the cuda state here.  Unfortunately,
            # we have no way to anticipate this will happen before we run the function.)
            ctx.had_cuda_in_fwd = False
            if torch.cuda._initialized:
                ctx.had_cuda_in_fwd = True
                ctx.fwd_gpu_devices, ctx.fwd_gpu_states = get_device_states(*inputs)

        ctx.input_requires_grad = [element.requires_grad for element in inputs]

        with torch.no_grad():
            # Makes a detached copy which shares the storage
            x = [element.detach() for element in inputs]
            outputs = ctx.fn(*x)

        if not isinstance(outputs, tuple):
            outputs = (outputs,)

        # Detaches y in-place (inbetween computations can now be discarded)
        detached_outputs = tuple([element.detach_() for element in outputs])

        # clear memory from inputs
        if not ctx.keep_input:
            # PyTorch 1.0+ way to clear storage
            for element in inputs:
                element.storage().resize_(0)

        # store these tensor nodes for backward pass
        ctx.inputs = [inputs] * num_bwd_passes
        ctx.outputs = [detached_outputs] * num_bwd_passes

        return detached_outputs

    @staticmethod
    @custom_bwd
    def backward(ctx, *grad_outputs):  # pragma: no cover
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError("InvertibleCheckpointFunction is not compatible with .grad(), please use .backward() if possible")
        # retrieve input and output tensor nodes
        if len(ctx.outputs) == 0:
            raise RuntimeError("Trying to perform backward on the InvertibleCheckpointFunction for more than "
                               "{} times! Try raising `num_bwd_passes` by one.".format(ctx.num_bwd_passes))
        inputs = ctx.inputs.pop()
        outputs = ctx.outputs.pop()

        # recompute input if necessary
        if not ctx.keep_input:
            # Stash the surrounding rng state, and mimic the state that was
            # present at this time during forward.  Restore the surrounding state
            # when we're done.
            rng_devices = []
            if ctx.preserve_rng_state and ctx.had_cuda_in_fwd:
                rng_devices = ctx.fwd_gpu_devices
            with torch.random.fork_rng(devices=rng_devices, enabled=ctx.preserve_rng_state):
                if ctx.preserve_rng_state:
                    torch.set_rng_state(ctx.fwd_cpu_state)
                    if ctx.had_cuda_in_fwd:
                        set_device_states(ctx.fwd_gpu_devices, ctx.fwd_gpu_states)
                # recompute input
                with torch.no_grad():
                    inputs_inverted = ctx.fn_inverse(*outputs)
                    if not isinstance(inputs_inverted, tuple):
                        inputs_inverted = (inputs_inverted,)
                    for element_original, element_inverted in zip(inputs, inputs_inverted):
                        element_original.storage().resize_(int(np.prod(element_original.size())))
                        element_original.set_(element_inverted)

        # compute gradients
        with torch.set_grad_enabled(True):
            detached_inputs = tuple([element.detach().requires_grad_() for element in inputs])
            temp_output = ctx.fn(*detached_inputs)
        if not isinstance(temp_output, tuple):
            temp_output = (temp_output,)

        gradients = torch.autograd.grad(outputs=temp_output, inputs=detached_inputs + ctx.weights, grad_outputs=grad_outputs)

        # Setting the gradients manually on the inputs and outputs (mimic backwards)
        for element, element_grad in zip(inputs, gradients[:ctx.num_inputs]):
            element.grad = element_grad

        for element, element_grad in zip(outputs, grad_outputs):
            element.grad = element_grad

        return (None, None, None, None, None, None) + gradients


class InvertibleModuleWrapper(nn.Module):
    def __init__(self, fn, keep_input=False, keep_input_inverse=False, num_bwd_passes=1,
                 disable=False, preserve_rng_state=False):
        """
        The InvertibleModuleWrapper which enables memory savings during training by exploiting
        the invertible properties of the wrapped module.

        Parameters
        ----------
            fn : :obj:`torch.nn.Module`
                A torch.nn.Module which has a forward and an inverse function implemented with
                :math:`x == m.inverse(m.forward(x))`

            keep_input : :obj:`bool`, optional
                Set to retain the input information on forward, by default it can be discarded since it will be
                reconstructed upon the backward pass.

            keep_input_inverse : :obj:`bool`, optional
                Set to retain the input information on inverse, by default it can be discarded since it will be
                reconstructed upon the backward pass.

            num_bwd_passes :obj:`int`, optional
                Number of backward passes to retain a link with the output. After the last backward pass the output
                is discarded and memory is freed.
                Warning: if this value is raised higher than the number of required passes memory will not be freed
                correctly anymore and the training process can quickly run out of memory.
                Hence, The typical use case is to keep this at 1, until it raises an error for raising this value.

            disable : :obj:`bool`, optional
                This will disable using the InvertibleCheckpointFunction altogether.
                Essentially this renders the function as :math:`y = fn(x)` without any of the memory savings.
                Setting this to true will also ignore the keep_input and keep_input_inverse properties.

            preserve_rng_state : :obj:`bool`, optional
                Setting this will ensure that the same RNG state is used during reconstruction of the inputs.
                I.e. if keep_input = False on forward or keep_input_inverse = False on inverse. By default
                this is False since most invertible modules should have a valid inverse and hence are
                deterministic.

        Attributes
        ----------
            keep_input : :obj:`bool`, optional
                Set to retain the input information on forward, by default it can be discarded since it will be
                reconstructed upon the backward pass.

            keep_input_inverse : :obj:`bool`, optional
                Set to retain the input information on inverse, by default it can be discarded since it will be
                reconstructed upon the backward pass.

        Note
        ----
            The InvertibleModuleWrapper can be used with mixed-precision training using
            :obj:`torch.cuda.amp.autocast` as of torch v1.6 and above. However, inputs will always be cast
            to :obj:`torch.float32` internally. This is done to minimize autocasting inputs to a different datatype
            which usually results in a disconnected computation graph and will raise an error on the backward pass.

        """
        super(InvertibleModuleWrapper, self).__init__()
        self.disable = disable
        self.keep_input = keep_input
        self.keep_input_inverse = keep_input_inverse
        self.num_bwd_passes = num_bwd_passes
        self.preserve_rng_state = preserve_rng_state
        self._fn = fn

    def forward(self, *xin):
        """Forward operation :math:`R(x) = y`

        Parameters
        ----------
            *xin : :obj:`torch.Tensor` tuple
                Input torch tensor(s).

        Returns
        -------
            :obj:`torch.Tensor` tuple
                Output torch tensor(s) *y.

        """
        if not self.disable:
            y = InvertibleCheckpointFunction.apply(
                self._fn.forward,
                self._fn.inverse,
                self.keep_input,
                self.num_bwd_passes,
                self.preserve_rng_state,
                len(xin),
                *(xin + tuple([p for p in self._fn.parameters() if p.requires_grad])))
        else:
            y = self._fn(*xin)

        # If the layer only has one input, we unpack the tuple again
        if isinstance(y, tuple) and len(y) == 1:
            return y[0]
        return y

    def inverse(self, *yin):
        """Inverse operation :math:`R^{-1}(y) = x`

        Parameters
        ----------
            *yin : :obj:`torch.Tensor` tuple
                Input torch tensor(s).

        Returns
        -------
            :obj:`torch.Tensor` tuple
                Output torch tensor(s) *x.

        """
        if not self.disable:
            x = InvertibleCheckpointFunction.apply(
                self._fn.inverse,
                self._fn.forward,
                self.keep_input_inverse,
                self.num_bwd_passes,
                self.preserve_rng_state,
                len(yin),
                *(yin + tuple([p for p in self._fn.parameters() if p.requires_grad])))
        else:
            x = self._fn.inverse(*yin)

        # If the layer only has one input, we unpack the tuple again
        if isinstance(x, tuple) and len(x) == 1:
            return x[0]
        return x




# We can't know if the run_fn will internally move some args to different devices,
# which would require logic to preserve rng states for those devices as well.
# We could paranoically stash and restore ALL the rng states for all visible devices,
# but that seems very wasteful for most cases.  Compromise:  Stash the RNG state for
# the device of all Tensor args.
#
# To consider:  maybe get_device_states and set_device_states should reside in torch/random.py?
#
# get_device_states and set_device_states cannot be imported from torch.utils.checkpoint, since it was not
# present in older versions, so we include a copy here.
def get_device_states(*args):
    # This will not error out if "arg" is a CPU tensor or a non-tensor type because
    # the conditionals short-circuit.
    fwd_gpu_devices = list(set(arg.get_device() for arg in args
                               if isinstance(arg, torch.Tensor) and arg.is_cuda))

    fwd_gpu_states = []
    for device in fwd_gpu_devices:
        with torch.cuda.device(device):
            fwd_gpu_states.append(torch.cuda.get_rng_state())

    return fwd_gpu_devices, fwd_gpu_states


def set_device_states(devices, states):
    for device, state in zip(devices, states):
        with torch.cuda.device(device):
            torch.cuda.set_rng_state(state)