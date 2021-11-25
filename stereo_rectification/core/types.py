#!/usr/bin/env python
from typing import Generic, TypeVar

import numpy as np

__all__ = ["Array"]

Shape = TypeVar("Shape")
DType = TypeVar("DType")


class Array(np.ndarray, Generic[Shape, DType]):
    pass
