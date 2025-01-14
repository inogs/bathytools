import pylab as pl

def imshow(M2d,title:str=""):
    fig,ax = pl.subplots()
    im=ax.imshow(M2d,origin='lower')
    fig.colorbar(im)
    ax.set_title(title)
    #fig.set_size_inches((40, 30));
    fig.show()
    return fig, ax

def set_same_box(ax_src, ax_dest, fig_dest):
    """
    Graphical help
    1) Zoom in ax_src
    2) execute set_same_box(ax1, ax2, fig2)
    3) Resize fig2, to view the same box
    """
    xlim = ax_src.get_xlim()
    ylim = ax_src.get_ylim()
    ax_dest.set_xlim(xlim)
    ax_dest.set_ylim(ylim)
    fig_dest.show()