from __future__ import annotations
from seaborn._marks.base import Mark


class Bar(Mark):

    supports = ["hue"]

    def __init__(self, multiple="dodge", **kwargs):

        super().__init__(**kwargs)
        self._multiple = multiple

    def _adjust(self, df, mappings):

        # Abstract out the pos/val axes based on orientation
        if self.orient == "y":
            pos, val = "yx"
        else:
            pos, val = "xy"

        # First augment the df with the other mappings we need: width and baseline
        # Question: do we want "ymin/ymax" or "baseline/y"? Or "ymin/y"?
        # Also note that these could be
        # a) mappings
        # b) "scalar" mappings
        # c) Bar constructor kws?
        defaults = {"baseline": 0, "width": .8}
        df = df.assign(**{k: v for k, v in defaults.items() if k not in df})

        # Bail here if we don't actually need to adjust anything?
        # TODO should the above stuff happen somewhere else?
        if not mappings:  # TODO filter this to grouping vars externally?
            return df

        # Now we need to know the levels of the grouping variables, hmmm.
        # Should `_plot_layer` pass that in here?
        # TODO prototyping with hue, this needs some real thinking!
        # TODO maybe instead of that we have the dataframe sorted by categorical order?

        # Adjust as appropriate
        # TODO currently this does not check that it is necessary to adjust!
        if self._multiple.startswith("dodge"):

            # TODO this is pretty general so probably doesn't need to be in Bar.
            # but it will require a lot of work to fix up, especially related to
            # ordering of groups (including representing groups that are specified
            # in the variable levels but are not in the dataframe

            # TODO this implements "flexible" dodge, i.e. fill the original space
            # even with missing levels, which is nice and worth adding, but:
            # 1) we also need to implement "fixed" dodge
            # 2) we need to think of the right API for allowing that
            # The dodge/dodgefill thing is a provisional idea

            width_by_pos = df.groupby(pos, sort=False)["width"]
            if self._multiple == "dodgefill":  # Not great name given other "fill"
                # TODO e.g. what should we do here with empty categories?
                # is it too confusing if we appear to ignore "dodgefill",
                # or is it inconsistent with behavior elsewhere?
                max_by_pos = width_by_pos.max()
                sum_by_pos = width_by_pos.sum()
            else:
                # TODO meanwhile here, we do get empty space, but
                # it is always to the right of the bars that are there
                max_width = df["width"].max()
                max_by_pos = {p: max_width for p, _ in width_by_pos}
                max_sum = width_by_pos.sum().max()
                sum_by_pos = {p: max_sum for p, _ in width_by_pos}

            df.loc[:, "width"] = width_by_pos.transform(
                lambda x: (x / sum_by_pos[x.name]) * max_by_pos[x.name]
            )

            # TODO maybe this should be building a mapping dict for pos?
            # (It is probably less relevent for bars, but what about e.g.
            # a dense stripplot, where we'd be doing a lot more operations
            # than we need to be doing this way.
            df.loc[:, pos] = (
                df[pos]
                - df[pos].map(max_by_pos) / 2
                + width_by_pos.transform(
                    lambda x: x.shift(1).fillna(0).cumsum()
                )
                + df["width"] / 2
            )

        return df

    def _plot_split(self, keys, data, ax, mappings, kws):

        if "hue" in data:
            kws["color"] = mappings["hue"](data["hue"])
        else:
            kws["color"] = "C0"  # FIXME:default attributes

        if self.orient == "y":
            func = ax.barh
            argmap = dict(y="y", width="x", height="width")
        else:
            func = ax.bar
            argmap = dict(x="x", height="y", width="width")

        func(**{k: data[v] for k, v in argmap.items()}, **kws)
