from distutils.version import LooseVersion
import matplotlib as mpl


def MarkerStyle(marker=None, fillstyle=None):
    """
    Allow MarkerStyle to accept a MarkerStyle object as parameter.

    Supports matplotlib < 3.3.0
    https://github.com/matplotlib/matplotlib/pull/16692

    """
    if isinstance(marker, mpl.markers.MarkerStyle):
        if fillstyle is None:
            return marker
        else:
            marker = marker.get_marker()
    return mpl.markers.MarkerStyle(marker, fillstyle)


def scale_factory(scale, axis, **kwargs):
    """
    Backwards compatability for creation of independent scales.

    Matplotlib scales require an Axis object for instantiation on < 3.4.
    But the axis is not used, aside from extraction of the axis_name in LogScale.

    """
    if isinstance(scale, str):
        class Axis:
            axis_name = axis
        axis = Axis()
    return mpl.scale.scale_factory(scale, axis, **kwargs)


def set_scale_obj(ax, axis, scale):
    """Handle backwards compatability with setting matplotlib scale."""
    if LooseVersion(mpl.__version__) < "3.4":
        # The ability to pass a BaseScale instance to Axes.set_{}scale was added
        # to matplotlib in version 3.4.0: GH: matplotlib/matplotlib/pull/19089
        # Workaround: use the scale name, which is restrictive only if the user
        # wants to define a custom scale; they'll need to update the registry too.
        ax.set(**{f"{axis}scale": scale.scale_obj.name})
    else:
        ax.set(**{f"{axis}scale": scale.scale_obj})
