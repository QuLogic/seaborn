from __future__ import annotations
from copy import copy

import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib.scale import LinearScale
from matplotlib.colors import Normalize

from seaborn._core.rules import VarType, variable_type, categorical_order

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Callable
    from numpy.typing import DTypeLike
    from pandas import Series
    from matplotlib.axis import Axis
    from matplotlib.scale import ScaleBase


class Scale:

    def __init__(self, scale_obj: ScaleBase):

        self.scale_obj = scale_obj

        # Initialize attributes that might not be set by subclasses
        self.norm = None
        self.order = None
        self.formatter = None
        ...

    def cast(self, data: Series) -> Series:
        raise NotImplementedError()

    def normalize(self, data: Series) -> Series:
        raise NotImplementedError()

    def forward(self, data: Series) -> Series:

        transform = self.scale_obj.get_transform().transform
        return transform(self.convert(data))

    def reverse(self, data: Series) -> Series:

        transform = self.scale_obj.get_transform().inverted().transform
        return transform(data)

    def convert(self, data: Series, axis: Axis | None = None) -> Series:
        raise NotImplementedError()


class NumericScale(Scale):

    scale_type = VarType("numeric")

    def __init__(self, scale_obj: ScaleBase, norm: Normalize, dtype: DTypeLike):

        super().__init__(scale_obj)
        self.norm = norm
        self.dtype = dtype

    def cast(self, data: Series) -> Series:

        return data.astype(self.dtype)

    def normalize(self, data: Series) -> Series:

        return self.norm(data)

    def setup(self, data: Series) -> NumericScale:

        out = copy(self)
        out.norm.autoscale_None(self.cast(data))
        return out

    def convert(self, data: Series, axis: Axis | None = None) -> Series:

        if axis is None:
            return self.cast(data)
        else:
            array = axis.convert_units(self.cast(data))
            return pd.Series(array, index=data.index, name=data.name)


class CategoricalScale(Scale):

    scale_type = VarType("categorical")

    def __init__(self, scale_obj: ScaleBase, order: list | None, formatter: Callable):

        super().__init__(scale_obj)
        self.order = order
        self.formatter = formatter

    def cast(self, data: Series) -> Series:

        order = pd.Index(categorical_order(data, self.order))

        data = data.map(self.formatter)
        order = order.map(self.formatter)

        assert len(order) == len(order.unique())  # TODO this was coerced, but why?
        return pd.Series(pd.Categorical(data, order), index=data.index, name=data.name)

    def setup(self, data: Series) -> CategoricalScale:

        out = copy(self)
        if out.order is None:
            out.order = categorical_order(data)
        return out

    def convert(self, data: Series, axis: Axis | None = None) -> Series:

        if axis is None:
            array = self.cast(data).cat.codes
        else:
            array = axis.convert_units(self.cast(data))
        return pd.Series(array, index=data.index, name=data.name)


class DateTimeScale(Scale):

    scale_type = VarType("datetime")

    def __init__(self, scale_obj: ScaleBase, format: str | None):

        super().__init__(scale_obj)
        self.norm = mpl.colors.Normalize()
        self.format = format

    def cast(self, data: pd.Series) -> Series:

        return pd.to_datetime(data, format=self.format)

    def normalize(self, data: Series) -> Series:

        return self.norm(self.convert(data))

    def setup(self, data: Series) -> DateTimeScale:

        out = copy(self)
        out.norm.autoscale_None(self.convert(self.cast(data)))

    def convert(self, data: Series, axis: Axis | None) -> Series:

        if axis is None:
            array = mpl.dates.date2num(self.cast(data))
        else:
            array = axis.convert_units(self.cast(data))
        return pd.Series(array, data.index, data.name)


def norm_from_scale(
    scale: ScaleBase, norm: tuple[float | None, float | None] | None,
) -> Normalize:

    if isinstance(norm, Normalize):
        return norm

    if norm is None:
        vmin = vmax = None
    else:
        vmin, vmax = norm  # TODO more helpful error if this fails?

    class ScaledNorm(Normalize):

        transform: Callable

        def __call__(self, value, clip=None):
            # From github.com/matplotlib/matplotlib/blob/v3.4.2/lib/matplotlib/colors.py
            # See github.com/matplotlib/matplotlib/tree/v3.4.2/LICENSE
            value, is_scalar = self.process_value(value)
            self.autoscale_None(value)
            if self.vmin > self.vmax:
                raise ValueError("vmin must be less or equal to vmax")
            if self.vmin == self.vmax:
                return np.full_like(value, 0)
            if clip is None:
                clip = self.clip
            if clip:
                value = np.clip(value, self.vmin, self.vmax)
            # Seaborn changes start
            t_value = self.transform(value).reshape(np.shape(value))
            t_vmin, t_vmax = self.transform([self.vmin, self.vmax])
            # Seaborn changes end
            if not np.isfinite([t_vmin, t_vmax]).all():
                raise ValueError("Invalid vmin or vmax")
            t_value -= t_vmin
            t_value /= (t_vmax - t_vmin)
            t_value = np.ma.masked_invalid(t_value, copy=False)
            return t_value[0] if is_scalar else t_value

    new_norm = ScaledNorm(vmin, vmax)

    # TODO do this, or build the norm into the ScaleWrapper.foraward interface?
    new_norm.transform = scale.get_transform().transform  # type: ignore  # mypy #2427

    return new_norm


def get_default_scale(data: Series):

    axis = data.name
    scale_obj = LinearScale(axis)

    var_type = variable_type(data)
    if var_type == "numeric":
        return NumericScale(scale_obj, norm=mpl.colors.Normalize(), dtype=float)
    elif var_type == "categorical":
        return CategoricalScale(scale_obj, order=None, formatter=format)
    elif var_type == "datetime":
        return DateTimeScale(scale_obj)
