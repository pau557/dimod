from random import random
from dimod.core.composite import Composite
from dimod.core.sampler import Sampler
from dimod.sampleset import concatenate
from dimod.vartypes import Vartype
from dimod.decorators import nonblocking_sample_method
import time

class MultiCallComposite(Sampler, Composite):
    """Composite for aggregating multiple calls
    """
    children = None
    parameters = None
    properties = None

    def __init__(self, child):
        self.children = [child]

        self.parameters = parameters = {}
        parameters.update(child.parameters)

        self.properties = {'child_properties': child.properties}

    @nonblocking_sample_method
    def sample(self, bqm, call_kwargs_list, spin_reversals=True, **kwargs):

        """Sample from the binary quadratic model.
        Args:
            bqm (:obj:`~dimod.BinaryQuadraticModel`):
                Binary quadratic model to be sampled from.
            call_kwarg_list (iterable of dictionaries):
                The sampler is called once per each set of kwargs in the iterable
            spin_reversals (Boolean)
                Whether to apply a different spin reveral transformation on each sampler call
        Returns:
            :obj:`.SampleSet`

        """
        # make a main response
        responses = []
        if not spin_reversals:
            for call_kwargs in call_kwargs_list:
                responses.append(self.child.sample(bqm, **call_kwargs, **kwargs))
            yield
            yield concatenate(responses)
        else:
            flipped_bqm = bqm
            transform = {v: False for v in bqm.variables}

            for call_kwargs in call_kwargs_list:
                flipped_bqm = flipped_bqm.copy()
                # flip each variable with a 50% chance
                for v in bqm:
                    if random() > .5:
                        transform[v] = not transform[v]
                        flipped_bqm.flip_variable(v)

                flipped_response = self.child.sample(flipped_bqm, **call_kwargs, **kwargs)
                responses.append((flipped_response, transform.copy()))

            yield

            for flipped_response, transform in responses:

                tf_idxs = [flipped_response.variables.index(v) for v, flip in transform.items() if flip]

                if bqm.vartype is Vartype.SPIN:
                    flipped_response.record.sample[:, tf_idxs] = -1 * flipped_response.record.sample[:, tf_idxs]
                else:
                    flipped_response.record.sample[:, tf_idxs] = 1 - flipped_response.record.sample[:, tf_idxs]

            yield concatenate([flipped_response for flipped_response, transform in responses])

