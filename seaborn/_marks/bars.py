from __future__ import annotations
from seaborn._marks.base import Mark


class Bar(Mark):

    supports = ["hue"]

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

    def _adjust(self, df):

        # TODO implement dodging/stacking here
        return df

    def _plot_split(self, keys, data, ax, mappings, kws):

        # TODO how do we automatically set orientation depending on the scale
        # used (i.e., if x is numeric and y is categorical, we want horizontal bars).
        # Is it necessary to make that something that can vary across Mark classes?
        if self.orient == "y":
            ax.barh(y=data["y"], width=data["x"], **kws)
        else:
            ax.bar(x=data["x"], height=data["y"], **kws)
