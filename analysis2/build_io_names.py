import ConfigParser as cp
import itertools as it

def build_operators_dict(config):
    # op_dict is dict of dicts
    op_dict = {}
    for op_entry in config.options('operator_lists'):
        # one operator list entry looks like g5.d>x,<x,>y,<y,>z,<z.p0
        tmp = config.get('operator_lists', op_entry)
        op = tmp.split('.')
        #print(op_entry,tmp)
        for o in op:
            if 'g' in o:
                gammas = o[1:].split(',')
            elif 'd' in o:
                #disps = o[1:].replace('0','000').replace('|','').split(',')
                disps = o[1:].replace('|','').split(',')
            elif 'p' in o:
                #momenta = o[1:].replace('0','000').split(',')
                momenta = o[1:].split(',')
            else:
                print("Quantumnumber id not known, has to be one of 'g','d','p'")
        op_dict[op_entry] = [gammas,disps,momenta]
    return op_dict
        
def build_correlators_dict(config):
    # corr_dict is dict of lists
    corr_dict = {}
    for corr_entry in config.options('correlator_lists'):
        # one correlator list entry looks like C2+:Op0:Q0:Op0:Q0
        tmp = config.get('correlator_lists', corr_entry)
        corr = tmp.split(':')
        name = corr[0]
        quarks = []
        operators = []
        for el in corr[1:]:
            if "Q" in el:
                quarks.append(config.get('quarks',el))
            if "Op" in el:
                operators.append(el.lower())
        corr_dict[corr_entry] = [name,quarks,operators]
    return corr_dict

#TODO: At the moment this is just a placeholder for more complicated momenta
#structures talk about that with markus
def expand_momenta(momenta_list):
    # Expanse needs to be list of lists
    return momenta_list

def get_op_member(operators,operator_names):

    gammas = [operators[i][0] for i in operator_names]
    displacements = [operators[i][1] for i in operator_names] 
    momenta = [operators[i][2] for i in operator_names]
    return gammas,displacements,momenta

def get_op_iterators(gammas,disps,moms):
    gamma_iter = it.product(*gammas)
    disp_iter = it.product(*disps)
    mom_iter = it.product(*moms)

    return gamma_iter,disp_iter,mom_iter

def build_dataset_names(operators,corr_dict_entry,h5=False):
    # Build dataset names with itertools
    dataset_names = []
    # We could have two, three or four operators per corr entry
    # Get a tuple of gamma structures, displacements and momenta
    # get operator ids
    gammas, displacements, momenta = get_op_member(operators,corr_dict_entry[2])
    momenta_expanded = expand_momenta(momenta)
    g_iter, d_iter, mom_iter = get_op_iterators(gammas,displacements,
                                             momenta_expanded)
    if h5 is not False:
        ds_name = "%s_%s" %(corr_dict_entry[0],('').join(corr_dict_entry[1]))
    else: 
        ds_name = "%s/%s_%s" %(corr_dict_entry[0],corr_dict_entry[0],
                                ('').join(corr_dict_entry[1])) 
    for op in it.product(mom_iter,d_iter,g_iter):
        #print(op)
        # TODO: Dirty Hack make functions out of that 
        if '2' in ds_name:
            dataset_names.append("%s_p%s.d%s.g%s_p%s.d%s.g%s"%(ds_name,op[0][0],
                                                           op[1][0],op[2][0],
                                                           op[0][1],op[1][1],
                                                           op[2][1]))
        elif '4' in ds_name:
            print(op)
            dataset_names.append("%s_p%s.d%s.g%s_p%s.d%s.g%s_p%s.d%s.g%s_p%s.d%s.g%s"
                                %(ds_name,
                                op[0][0], op[1][0], op[2][0],
                                op[0][1], op[1][1], op[2][1],
                                op[0][2], op[1][2], op[2][2],
                                op[0][3], op[1][3], op[2][3]))

    return dataset_names
# Example dataset name: C2+_uu_p000.d<x<x.g5_p000.d<y<y<y.g5

def build_dataset_list(operators,corr_dict,h5=False):
    ds_list = []
    for co in corr_dict.values():
        ds_list.append(build_dataset_names(operators,co,h5=h5))
    return ds_list

def inputnames(conf_file, corr_string=None, h5=False):
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
    print(op_dict)
    # Get dictionary of correlators
    corr_dict = build_correlators_dict(config)
    # Build a list of all possible datasetnames
    dataset_names = build_dataset_list(op_dict,corr_dict,h5=h5)
    return dataset_names

if __name__ == "__main__":
    pass
