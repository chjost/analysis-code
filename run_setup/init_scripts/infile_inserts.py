import subprocess as sub
def set_ens_info(opt,target):
    sub.check_call(['sed','-i','s@=E=@%s@g'%opt['Ensemble'],target])
    sub.check_call(['sed','-i','s@=L=@%d@g'%opt['L'],target])
    sub.check_call(['sed','-i','s@=T=@%d@g'%opt['T'],target])
    sub.check_call(['sed','-i','s@=T2=@%d@g'%(int(opt['T'])/2),target])
    sub.check_call(['sed','-i','s@=NSAM=@%d@g'%opt['bsamples'],target])
    sub.check_call(['sed','-i','s@=BS_BL=@%d@g'%opt['blocklength'],target])


def infile_pion(opt):
    wrkdir ='/hiskp4/helmes/analysis/scattering/pi_k/I_32_publish' 
    template = wrkdir+'/runs/templates/ens_pi.ini'
    rundir = wrkdir+'/runs/'+opt['Ensemble']+'/'+opt['pi_dir']
    target = rundir+'/'+opt['Ensemble']+'_pi.ini'
    # copy template to directory
    sub.check_call(['cp',template,target])
    set_ens_info(opt,target)
    sub.check_call(['sed','-i','s@=S=@%s/@g'%opt['pi_dir'],target])
    sub.check_call(['sed','-i','s@=WRKDIR=@%s/@g'%wrkdir,target])
    sub.check_call(['sed','-i','s@=IMIN_PI=@%d@g'%opt['pi_min'],target])
    sub.check_call(['sed','-i','s@=IMASS_PI=@%d,%d@g'%(opt['pi_bgn'],opt['pi_end']),target])

def infile_kaon(opt):
    wrkdir ='/hiskp4/helmes/analysis/scattering/pi_k/I_32_publish' 
    template = wrkdir+'/runs/templates/ens_k.ini'
    rundir = wrkdir+'/runs/'+opt['Ensemble']+'/'+opt['mu_s_dir']
    target = rundir+'/'+opt['Ensemble']+'_k.ini'
    # copy template to directory
    sub.check_call(['cp',template,target])
    set_ens_info(opt,target)
    sub.check_call(['sed','-i','s@=S=@%s/@g'%opt['mu_s_dir'],target])
    sub.check_call(['sed','-i','s@=WRKDIR=@%s/@g'%wrkdir,target])
    sub.check_call(['sed','-i','s@=IMIN_K=@%d@g'%opt['k_min'],target])
    sub.check_call(['sed','-i','s@=IMASS_K=@%d,%d@g'%(opt['k_bgn'],opt['k_end']),target])

def infile_pik(opt):
    wrkdir ='/hiskp4/helmes/analysis/scattering/pi_k/I_32_publish' 
    template = wrkdir+'/runs/templates/ens_pik.ini'
    rundir = wrkdir+'/runs/'+opt['Ensemble']+'/'+opt['mu_s_dir']
    target = rundir+'/'+opt['Ensemble']+'_pik.ini'
    # copy template to directory
    sub.check_call(['cp',template,target])
    set_ens_info(opt,target)
    sub.check_call(['sed','-i','s@=S=@%s/@g'%opt['mu_s_dir'],target])
    sub.check_call(['sed','-i','s@=WRKDIR=@%s/@g'%wrkdir,target])
    sub.check_call(['sed','-i','s@=IMIN_E=@%d@g'%opt['pik_min'],target])
    sub.check_call(['sed','-i','s@=IC4=@%d,%d@g'%(opt['pik_bgn'],opt['pik_end']),target])

