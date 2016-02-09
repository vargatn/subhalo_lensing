
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

import matplotlib.gridspec as gridspec
import matplotlib.ticker as plticker


import corner
def animate():
    """
    Matplotlib Animation Example

    author: Jake Vanderplas
    email: vanderplas@astro.washington.edu
    website: http://jakevdp.github.com
    license: BSD
    Please feel free to use and modify this, but keep the above information. Thanks!
    """


    # First set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure()
    ax = plt.axes(xlim=(0, 2), ylim=(-2, 2))
    line, = ax.plot([], [], lw=2)

    # initialization function: plot the background of each frame
    def init():
        line.set_data([], [])
        return line,

    # animation function.  This is called sequentially
    def animate(i):
        x = np.linspace(0, 2, 1000)
        y = np.sin(2 * np.pi * (x - 0.01 * i))
        line.set_data(x, y)
        return line,

    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=200, interval=20, blit=True)

    # save the animation as an mp4.  This requires ffmpeg or mencoder to be
    # installed.  The extra_args ensure that the x264 codec is used, so that
    # the video can be embedded in html5.  You may need to adjust this for
    # your system: for more information, see
    # http://matplotlib.sourceforge.net/api/animation_api.html
    anim.save('basic_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

    plt.show()



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