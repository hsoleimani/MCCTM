import pylab
import numpy as np
import matplotlib.gridspec as gridspec

dataset = 'ag'

fig_width_pt = 246.0  # Get this from LaTeX using \showthe\columnwidth
inches_per_pt = 1.0/72.27               # Convert pt to inch
golden_mean = (np.sqrt(5)-1.0)/2.0         # Aesthetic ratio
fig_width = fig_width_pt*inches_per_pt  # width in inches
fig_height = fig_width*golden_mean      # height in inches
fig_size =  [fig_width,fig_height]
params = {'backend': 'ps', 'axes.labelsize': 10,
           'text.fontsize': 10,
           'legend.fontsize': 8,
           'xtick.labelsize': 9,
           'ytick.labelsize': 9,
           'text.usetex': True,
           'figure.figsize': fig_size}
pylab.rcParams.update(params)

#fig, axes = pylab.subplots(nrows=2, ncols=4)
#fig.tight_layout()

ccn = np.loadtxt('%s_mcctm.txt'%dataset)
slda = np.loadtxt('%s_sslda.txt'%dataset)
dmr = np.loadtxt('%s_cclda.txt'%dataset)

#ind = np.array([1,2,4,6,7,8,9])-1
ind = np.array([1,2,3,4,5,6,7,8,9])-1
prop = ccn[:,0][ind]

# test
fig = pylab.figure(figsize=(6.5, 2.5))

gs2 = gridspec.GridSpec(2, 2)
gs2.update(left=0.08, right=0.98, hspace=0.05, wspace = 0.18, bottom = 0.18, top = 0.95)
ax = plt.subplot(gs2[:, :-1])
#ax = fig.add_subplot(1,2,1)
pylab.errorbar(prop,ccn[ind,1], yerr=ccn[ind,2],fmt='-*', color='r')
pylab.errorbar(prop,slda[ind,1], yerr=slda[ind,2],fmt='-<',color='b')
pylab.errorbar(prop,dmr[ind,1], yerr=dmr[ind,2],fmt='-o',color = 'g')
#pylab.plot(prop,svm[ind,2],'-v')
#ax.set_xscale('log')
pylab.ylabel('Test CCR')
pylab.xlabel('Label proportion')
pylab.legend(['MCCTM','ssLDA','ccLDA'],loc = 4,ncol=2)

#ax = fig.add_subplot(1,2,2)
#pylab.errorbar(prop,ccn[ind,7], yerr=ccn[ind,8],fmt='-*')
#pylab.errorbar(prop,slda[ind,7], yerr=slda[ind,8],fmt='-<')
#pylab.plot(prop,dmr[ind,7],'-^')
ax5 = plt.subplot(gs2[:-1, -1])
pylab.errorbar(prop,ccn[ind,5], yerr=ccn[ind,6],fmt='-*', color='r')
pylab.errorbar(prop,slda[ind,5], yerr=slda[ind,6],fmt='-<',color='b')
pylab.ylabel('log-likelihood')
ax5.get_xaxis().set_ticklabels([])
ax6 = plt.subplot(gs2[-1, -1])
pylab.errorbar(prop,dmr[ind,5],yerr=dmr[ind,6],fmt='-o',color = 'g')
#ax.set_xscale('log')
#pylab.plot(prop,svm[ind,1],'-v')
pylab.ylabel('log-likelihood')
pylab.xlabel('Label proportion')
#fig.tight_layout(pad = .5)
pylab.show()
pylab.savefig('%s_ccr.pdf' %dataset, bbox_inches='tight')

