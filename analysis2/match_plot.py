
def
plot_match(ana.MatchResult, path, label, name, debug=0):
    outpath = path+ens
    if debug > 0:
        print("Plotting to " + outpath)
    # stack data, be careful with order of masses
    # we want to look at B ensembles matched to D
    x = np.array((mus_fix[0],mus_fix[1],mus_match))
    # now the y values, again careful with order
    y = np.column_stack((data_low[:,0],data_high[:,0],data_ext))
    ext_plot = PdfPages(outpath+name)
    # Plot 50 samples in each plot
    # plot a subset of 50 samples
    # amin and amax of xvalues
    x_min = np.amin(x)
    x_max = np.amax(x)
    for i in range(1500):
        #_y = y[i:i+15]
        _y = y[i]
        #for i,d in enumerate(_y):
        #    plt.plot(x,d,'o--')
        plt.plot(x,_y,'o--')
    plt.xlim(x_min-x_min*0.01,x_max+x_max*0.01)
    plt.title(label[0])
    plt.xlabel(label[1])
    plt.ylabel(label[2])
    ext_plot.savefig()
    plt.clf()
    ext_plot.close()
