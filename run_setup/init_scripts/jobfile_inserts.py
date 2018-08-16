import subprocess as sub
def set_job_info(opt,target):
    sub.check_call(['sed','-i','s@=PARTITION=@cpubatch@g',target])
    sub.check_call(['sed','-i','s@=RUNTIME=@01:30:00@g',target])
    sub.check_call(['sed','-i','s@=E=@%s@g'%opt['Ensemble'],target])

def jobfile_pion(opt):
    wrkdir ='/hiskp4/helmes/analysis/scattering/pi_k/I_32_publish' 
    codedir = '/hiskp4/helmes/projects/analysis-code'
    template = wrkdir+'/runs/templates/job_script.slurm'
    rundir = wrkdir+'/runs/'+opt['Ensemble']+'/'+opt['pi_dir']
    target = rundir+'/'+'fit_pion.slurm'
    # copy template to directory
    sub.check_call(['cp',template,target])
    set_job_info(opt,target)
    sub.check_call(['sed','-i','s@=N=@%s@g'%(opt['Ensemble']+'_fit_pi'),target])
    sub.check_call(['sed','-i','s@=S=@%s/@g'%opt['pi_dir'],target])
    sub.check_call(['sed','-i','s@=WRKDIR=@%s/@g'%wrkdir,target])
    sub.check_call(['sed','-i','s@=CODEDIR=@%s/@g'%codedir,target])
    sub.check_call(['sed','-i','$ a date',target])
    sub.check_call(['sed','-i','$ a ./pik_32_fit_pi.py \${WORKDIR}/\${LAT}_pi.ini',target])
    sub.check_call(['sed','-i','$ a date',target])

def jobfile_kaon(opt):
    wrkdir ='/hiskp4/helmes/analysis/scattering/pi_k/I_32_publish' 
    codedir = '/hiskp4/helmes/projects/analysis-code'
    template = wrkdir+'/runs/templates/job_script.slurm'
    rundir = wrkdir+'/runs/'+opt['Ensemble']+'/'+opt['mu_s_dir']
    target = rundir+'/'+'fit_kaon.slurm'
    # copy template to directory
    sub.check_call(['cp',template,target])
    set_job_info(opt,target)
    sub.check_call(['sed','-i','s@=N=@%s@g'%(opt['Ensemble']+'_fit_k'),target])
    sub.check_call(['sed','-i','s@=S=@%s/@g'%opt['mu_s_dir'],target])
    sub.check_call(['sed','-i','s@=WRKDIR=@%s/@g'%wrkdir,target])
    sub.check_call(['sed','-i','s@=CODEDIR=@%s/@g'%codedir,target])
    sub.check_call(['sed','-i','$ a date',target])
    sub.check_call(['sed','-i','$ a ./pik_32_fit_k.py \${WORKDIR}/\${LAT}_k.ini',target])
    sub.check_call(['sed','-i','$ a date',target])

def jobfile_pik(opt):
    wrkdir ='/hiskp4/helmes/analysis/scattering/pi_k/I_32_publish' 
    codedir = '/hiskp4/helmes/projects/analysis-code'
    template = wrkdir+'/runs/templates/job_script.slurm'
    rundir = wrkdir+'/runs/'+opt['Ensemble']+'/'+opt['mu_s_dir']
    target = rundir+'/'+'fit_c4.slurm'
    # copy template to directory
    sub.check_call(['cp',template,target])
    set_job_info(opt,target)
    sub.check_call(['sed','-i','s@=N=@%s@g'%(opt['Ensemble']+'_fit_pik_corr_false'),target])
    sub.check_call(['sed','-i','s@=S=@%s/@g'%opt['mu_s_dir'],target])
    sub.check_call(['sed','-i','s@=WRKDIR=@%s/@g'%wrkdir,target])
    sub.check_call(['sed','-i','s@=CODEDIR=@%s/@g'%codedir,target])
    sub.check_call(['sed','-i','$ a date',target])
    sub.check_call(['sed','-i','$ a hostname',target])
    sub.check_call(['sed','-i','$ a ./pik_32_fit_pik_all.py \${WORKDIR}/\${LAT}_pik.ini',target])
    sub.check_call(['sed','-i','$ a date',target])

def jobfile_scat_len(opt):
    wrkdir ='/hiskp4/helmes/analysis/scattering/pi_k/I_32_publish' 
    codedir = '/hiskp4/helmes/projects/analysis-code'
    template = wrkdir+'/runs/templates/job_script.slurm'
    rundir = wrkdir+'/runs/'+opt['Ensemble']+'/'+opt['mu_s_dir']
    target = rundir+'/'+'scat_len.slurm'
    # copy template to directory
    sub.check_call(['cp',template,target])
    set_job_info(opt,target)
    sub.check_call(['sed','-i','s@=N=@%s@g'%(opt['Ensemble']+'_scat_len'),target])
    sub.check_call(['sed','-i','s@=E=@%s@g'%opt['Ensemble'],target])
    sub.check_call(['sed','-i','s@=S=@%s/@g'%opt['mu_s_dir'],target])
    sub.check_call(['sed','-i','s@=WRKDIR=@%s/@g'%wrkdir,target])
    sub.check_call(['sed','-i','s@=CODEDIR=@%s/@g'%codedir,target])
    sub.check_call(['sed','-i','$ a date',target])
    sub.check_call(['sed','-i','$ a ./pik_32_scat_len.py \${WORKDIR}/\${LAT}_pik.ini',target])
    sub.check_call(['sed','-i','$ a date',target])

