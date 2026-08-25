"""The MOTOR backbone, ported from JAX to PyTorch.

The port is necessary because JAX defines no library for LoRA, or any PEFT
methodology for that matter. Everything here reproduces the released model --
its layers, its assembled stack, its weights and its vocabulary -- and is
asserted against the numerical oracle dumped from the original.
"""
