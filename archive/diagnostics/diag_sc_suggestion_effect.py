"""Diagnostic: step-count suggestion effect averaged over the fitted data (main effect + interactions x covariates),
compared between env_para_vanilla and the 20261005 backup. Usage: python diag_sc_suggestion_effect.py"""
import json,glob,numpy as np,pandas as pd,re,ast,sys
src=open('07_fit_testbed.py').read()
ns={'np':np,'pd':pd}
def grab(fn):
    m=re.search(rf"^def {fn}\(.*?(?=^def |\Z)",src,re.S|re.M); return m.group(0)
exec(grab('fill_nan_with_mean'),ns)
m=re.search(r"^THETA_FOURSC_NAMES\s*=\s*\[.*?\]",src,re.S|re.M); exec(m.group(0),ns)
exec(grab('build_fourSC_bayes_data'),ns)
for lab,d in [('new','env_para_vanilla'),('old','env_para_vanilla_backup_20261005')]:
    df=pd.read_csv(f'{d}/df_fit_11week.csv')
    X=ns['build_fourSC_bayes_data'](df,sorted(df.ParticipantIdentifier.unique()))
    names=ns['THETA_FOURSC_NAMES']
    ai=[n for n in names if n.startswith('WalkingSuggestion')]
    effs=[];main=[]
    for u,g in X.groupby('ParticipantIdentifier'):
        p=json.load(open(f'{d}/params_env_{u}.json')); th=dict(zip(p['theta_fourSC_names'],p['theta_fourSC']))
        e=th['WalkingSuggestion']+sum(th[n]*g[n[len('WalkingSuggestion_by_'):]].values for n in ai[1:])
        effs.append(e.mean()); main.append(th['WalkingSuggestion'])
    effs=np.array(effs)
    print(lab,'main mean %.4f  avg effect over data: mean %.4f median %.4f sd %.4f'%(np.mean(main),effs.mean(),np.median(effs),effs.std()))
    print('   covariate means:',X[[n[len('WalkingSuggestion_by_'):] for n in ai[1:]]].mean().round(3).to_dict())
