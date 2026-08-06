"""A stand-in for NVIDIA apex, so Renee runs without compiling it.

Copy this next to Renee's ``main.py`` (the current directory wins on ``sys.path``):

    cp /content/zestxml/benchmarks/colab/apex.py /content/renee/apex.py

Renee imports apex in ``dl_base.py`` and uses exactly two symbols,
``apex.optimizers.FusedSGD`` and ``apex.optimizers.FusedAdam``. Their own
``install2.sh`` notes apex is "need only for apex optimizers", and it is not on PyPI --
``pip install apex`` fetches an unrelated Pyramid authentication library, which fails with
``cannot import name 'UnencryptedCookieSessionFactoryConfig'``. Building the real thing
takes 10-20 minutes and often breaks against the current CUDA/torch pair.

The fused optimizers are a speed optimisation, not a different algorithm, so mapping them
onto torch's own costs some throughput and nothing else:

    FusedSGD  -> torch.optim.SGD
    FusedAdam -> torch.optim.AdamW   (apex defaults to adam_w_mode=True, i.e. decoupled
                                      weight decay, which is AdamW rather than Adam)

Two apex-only keyword arguments are dropped: ``set_grad_none``, whose behaviour is
torch's default since 2.0, and ``bias_correction``, which torch's Adam always applies.
"""

from torch import optim

_APEX_ONLY_KWARGS = ("set_grad_none", "bias_correction", "adam_w_mode")


def _torch_kwargs(kwargs):
    return {k: v for k, v in kwargs.items() if k not in _APEX_ONLY_KWARGS}


class optimizers:  # noqa: N801 - mirrors the apex.optimizers namespace
    @staticmethod
    def FusedSGD(params, **kwargs):  # noqa: N802 - mirrors the apex name
        return optim.SGD(params, **_torch_kwargs(kwargs))

    @staticmethod
    def FusedAdam(params, **kwargs):  # noqa: N802
        return optim.AdamW(params, **_torch_kwargs(kwargs))

    @staticmethod
    def FusedLAMB(params, **kwargs):  # noqa: N802 - unused by Renee, here for safety
        return optim.AdamW(params, **_torch_kwargs(kwargs))
