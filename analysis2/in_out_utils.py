import ConfigParser as cp

def build_operators_dict(config):
    # op_dict is dict of dicts
    op_dict = {}
    for op in config.options('operator_lists'):
        # one operator list entry looks like g5.d>x,<x,>y,<y,>z,<z.p0
        op_list = op.split('.')
        gammas = {'g': op_list[0].split(',')}
        disps = {'d':op_list[1].split(',')}
        momenta = {'p':op_list[2].split(',')}
        



def inputnames(conf_file, corr_string, h5=False):
    """ Function to build Correlator input names conformal with B. Knippschilds
    naming scheme
        
    The function invokes a config parser which reads the config file (config.ini) regarding the
    specified correlation functions and returns a list of input names

    Args:
        conf_file: string with path to the configuration file 
        corr_string: array of identifiers for correlation functions to be
                    read in
        h5: bool, if True drop file ending and padding corr specifier
    Returns:
        A list of inputnames

    """
    # invoke config parser and read config file
    config = cp.SafeConfigParser()
    config.read(conf_file)
    # Get dictionary of operators
    op_dict = build_operators_dict(config)
    # set section names for get operations, have to be the same in configfile
    #quarks = config.get('quarks')
    #operators = config.get('operator_lists')
    #corr_list = config.get('correlator_lists')
    # result list
    inputnames = []
    # Loop over correlation function names
    for key in config.options('correlator_lists'):
        # get value of each key splited by ':'
        tmp = config.get('correlator_lists',key)
        c0 = tmp.split(':')
        # read only functions in corr_string, sort them into q and op arrays
        #if c0[0] in corr_string:            
        if key in corr_string:            
            print("reading %r" %key)
            q_list = []
            op_list = []
            for val in c0[1:]:
                if val[0] == 'Q':
                    q_list.append(config.get('quarks',val))
                elif val[0] == 'O':
                    op_list.append(config.get('operator_lists',val))
                else:
                    print("Identifier not found")
            # TODO: expand entrys in operator lists
            # join single lists together for filename
            join_q = ''.join(q_list)
            join_op = '_'.join(op_list)
            # build the filename
            if h5 is True:
                corrname = c0[0]+"_"+join_q+"_"+join_op
            else:
                corrname = c0[0]+"/"+c0[0]+"_"+join_q+"_"+join_op+".dat"
            print("in inputnames: corrname = %s" %corrname)
            inputnames.append(corrname)
    return inputnames

