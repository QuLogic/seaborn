from __future__ import annotations

from seaborn._core.rules import variable_type

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Literal, Any, Type, Dict
    from collections.abc import Callable, Generator
    from pandas import DataFrame
    from matplotlib.axes import Axes
    from seaborn._core.mappings import SemanticMapping
    from seaborn._stats.base import Stat

    MappingDict = Dict[str, SemanticMapping]


class Mark:
    """Base class for objects that control the actual plotting."""
    # TODO where to define vars we always group by (col, row, group)
    default_stat: Type[Stat] | None = None
    grouping_vars: list[str] = []
    orient: Literal["x", "y"]
    requires: list[str]  # List of variabes that must be defined
    supports: list[str]  # List of variables that will be used

    def __init__(self, **kwargs: Any):

        self._kwargs = kwargs

    def _adjust(self, df: DataFrame, mappings: dict) -> DataFrame:

        return df

    def _infer_orient(self, scales: dict) -> Literal["x", "y"]:  # TODO type scale

        # TODO The original version of this (in seaborn._oldcore) did more checking.
        # Paring that down here for the prototype to see what restrictions make sense.

        x_type = None if "x" not in scales else scales["x"].type
        y_type = None if "x" not in scales else scales["y"].type

        if x_type is None:
            return "y"

        elif y_type is None:
            return "x"

        elif x_type != "categorical" and y_type == "categorical":
            return "y"

        elif x_type != "numeric" and y_type == "numeric":
            return "x"

        elif x_type == "numeric" and y_type != "numeric":
            return "y"

        else:
            return "x"

    def _plot(
        self, generate_splits: Callable[[], Generator], mappings: MappingDict,
    ) -> None:
        """Main interface for creating a plot."""
        for keys, data, ax in generate_splits():
            kws = self._kwargs.copy()
            self._plot_split(keys, data, ax, mappings, kws)

        self._finish_plot()

    def _plot_split(
        self,
        keys: dict[str, Any],
        data: DataFrame,
        ax: Axes,
        mappings: MappingDict,
        kws: dict,
    ) -> None:
        """Method that plots specific subsets of data. Must be defined by subclass."""
        raise NotImplementedError()

    def _finish_plot(self) -> None:
        """Method that is called after each data subset has been plotted."""
        pass
