import matplotlib.pyplot as plt
"""
Functions setting the plot layout
"""
def set_plotstyles(lattice_spacings):
    """For chiral plots set the marker- and linestyles and color of data points
    as well as the curves associated with them. 

    Inputs
    ------
    lattice_spacings: list of strings encoding the used beta-values

    Returns
    -------
    colors: list of colorvalues truncated to lenght of lattice_spacings
    markerstyles: list of markerstyles truncated to lenght of lattice_spacings
    linestyles: list of linestyles truncated to lenght of lattice_spacings
    datalabels: initial list of lattice spacings 
    """
    # Initialize default lists for lattice spacings
    colors = ['r','b','g']
    markerstyles = ['^','v','o']
    linestyles = ['--',':','-.']
    #TODO: Not sure if it makes more sense to just set that to the lattice
    #spacings
    #datalabels = [r'$a=0.0885$fm',r'$a=0.0815$fm',r'$a=0.0619$fm']
    datalabels = lattice_spacings
    n_spaces = len(lattice_spacings)
    return colors[0:n_spaces], markerstyles[0:n_spaces], linestyles[0:n_spaces], datalabels[0:n_spaces]

# TODO: document these functions
def set_layout(physical_x=None,xlimits=None,ylimits=None,legend_array=None,
              labels=None,labelfontsize=None):
    set_plotlimits(xlimits,ylimits)
    if labels is not None:
      plot_physical_x(physical_x,labels[0])
    else:
      plot_physical_x(physial_x,"x")
    set_legend(legend_array)
    set_labels(labels,fontsize=labelfontsize)
    
def set_legend(legend_array=None):
    if legend_array is not None and len(legend_array == 4):
        plt.legend(loc=legend_array[0],ncol=legend_array[1],
                   numpoints=legendarray[2],fontsize=legend_array[3])
    else:
        plt.legend(loc='lower left',ncol=2,numpoints=1,fontsize=16)

def plot_physical_x(physical_x,x_name):
    if physical_x is not None:
        plt.axvline(x=physical_x, color='k', ls='--', label=x_name+'_phys.')
# TODO: Not finished yet, how to deal with arguments and stuff, perhaps not
# in set_layout
def plot_xcut():
    if xcut is not None:
        if len(xcut) > 1:
            plot_brace(fit_arguments,xcut,fit_function,xpos="low")
            plot_brace(fit_arguments,xcut,fit_function,xpos="up")
        else:
            plot_brace(fit_arguments,xcut,fit_function,xpos="up")

def set_plotlimits(xlimits=None,ylimits=None):
    if xlimits is not None:
        plt.xlim(xlimits[0],xlimits[1])
    if ylimits is not None:
        plt.ylim(ylimits[0],ylimits[1])

def set_labels(labels=None,fontsize = None):
    if labels is not None:
        if fontsize is not None:
            plt.xlabel(labels[0],fontsize=fontsize)
            plt.ylabel(labels[1],fontsize=fontsize)
        else:
            plt.xlabel(labels[0])
            plt.ylabel(labels[1])
        if len(labels) >= 3:
            plt.title(labels[2])
