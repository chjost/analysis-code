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
