# -*- coding: utf-8 -*-
"""Error functions.

This module defines error functions, with the aim of model training being to
minimise the error function given a set of inputs and target outputs.

The error functions will typically measure some concept of distance between the
model outputs and target outputs, averaged over all data points in the data set
or batch.
"""

import numpy as np


class SumOfSquaredDiffsError(object):
    """Sum of squared differences (squared Euclidean distance) error."""

    def __call__(self, outputs, targets):
        error=(((outputs-targets)**2).sum())*(1/(2*outputs.shape[0]))
        return error 

    def grad(self, outputs, targets):
        grad=(1/outputs.shape[0])*(outputs-targets)
        return grad
        raise NotImplementedError()

    def __repr__(self):
        return 'SumOfSquaredDiffsError'
