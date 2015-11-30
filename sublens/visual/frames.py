
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as plticker

def add_prof(axt, axx, dtup, **kwargs):
    rr, dst, dsx = dtup[:3]
    axt.plot(rr, dst, **kwargs)
    axx.plot(rr, dsx, **kwargs)

    return axt, axx


def add_err(axt, axx, dtup, **kwargs):

    rr, dst, dst_e, dsx, dsx_e = dtup[:5]
    axt.errorbar(rr, dst, yerr=dst_e, **kwargs)
    axx.errorbar(rr, dsx, yerr=dsx_e, **kwargs)

    return axt, axx


def make_frame(size=(10., 8.), xlabel="R [$Mpc/h$]",
               ylabel1="$<\Delta\Sigma_t>$ [$h\, M_\odot / pc^3$]",
               ylabel2="$<\Delta\Sigma_\\times>$ [$h\, M_\odot / pc^3$]",
               xscale="log", yscale1="log", yscale2="linear", fontsize=14,
               xrange=(0.02, 30.), yrange1=None, yrange2=None):

    fig = plt.figure(figsize=size)
    gspc = gridspec.GridSpec(6, 1)
    spc1 = gspc.new_subplotspec((0, 0), rowspan=4)
    axt = fig.add_subplot(spc1)

    spc2 = gspc.new_subplotspec((4, 0), rowspan=2)
    axx = fig.add_subplot(spc2)

    if xrange is not None:
        axt.set_xlim(xrange)
        axx.set_xlim(xrange)

    if yrange1 is not None:
        axt.set_ylim(yrange1)

    if yrange2 is not None:
        axx.set_ylim(yrange2)

    axt.set_xscale(xscale)
    axx.set_xscale(xscale)
    axx.set_xlabel(xlabel, fontsize=fontsize)

    axt.set_ylabel(ylabel1, fontsize=fontsize)
    axx.set_ylabel(ylabel2, fontsize=fontsize)

    axt.set_yscale(yscale1)
    axx.set_yscale(yscale2)


    return fig, axt, axx