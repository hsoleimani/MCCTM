import pylab
import numpy as np
import matplotlib.gridspec as gridspec

dataset = 'yahoo'

fig_width_pt = 246.0  # Get this from LaTeX using \showthe\columnwidth
inches_per_pt = 1.0/72.27               # Convert pt to inch
golden_mean = (np.sqrt(5)-1.0)/2.0         # Aesthetic ratio
fig_width = fig_width_pt*inches_per_pt  # width in inches
fig_height = fig_width*golden_mean      # height in inches
fig_size =  [fig_width,fig_height]
params = {'backend': 'ps', 'axes.labelsize': 10,
           'text.fontsize': 10,
           'legend.fontsize': 9,
           'xtick.labelsize': 10,
           'ytick.labelsize': 10,
           'text.usetex': True,
           'figure.figsize': fig_size}
pylab.rcParams.update(params)

#fig, axes = pylab.subplots(nrows=2, ncols=4)
#fig.tight_layout()

ccn = np.loadtxt('%s_mcctm.txt'%dataset)
vccn = np.loadtxt('%s_vmcctm.txt'%dataset)
slda = np.loadtxt('%s_sslda.txt'%dataset)
dmr = np.loadtxt('%s_cclda.txt'%dataset)

#ind = np.array([1,2,4,6,7,8,9])-1
ind = np.array([1,2,3,4,5,6,7,8,9])-1
prop = ccn[:,0][ind]

# test
fig = pylab.figure(figsize=(6.5, 2.5))

gs2 = gridspec.GridSpec(2, 2)
gs2.update(left=0.08, right=0.98, hspace=0.05, wspace = 0.24, bottom = 0.18, top = 0.95)
ax = plt.subplot(gs2[:-1, :-1])
pylab.errorbar(prop,vccn[ind,1], yerr=vccn[ind,2],fmt='-*', color='r')
pylab.errorbar(prop,slda[ind,1], yerr=slda[ind,2],fmt='-<',color='b')
pylab.ylabel('Test CCR')
pylab.legend(['MCCTM','ssLDA'],loc = 4,ncol=2)
pylab.yticks([0.63,0.66,0.69,0.72])
ax.get_xaxis().set_ticklabels([])
ax = plt.subplot(gs2[1, 0])
pylab.errorbar(prop,dmr[ind,1], yerr=dmr[ind,2],fmt='-o',color = 'g')
#pylab.plot(prop,svm[ind,2],'-v')
#ax.set_xscale('log')
pylab.ylabel('Test CCR')
pylab.xlabel('Label proportion')
pylab.legend(['ccLDA'],loc = 4,ncol=2)
pylab.yticks([0.30,0.40,0.50,0.60])

ccn /= 1e7
vccn /= 1e7
slda /= 1e7
dmr /= 1e7
#ax = fig.add_subplot(1,2,2)
#pylab.errorbar(prop,ccn[ind,7], yerr=ccn[ind,8],fmt='-*')
#pylab.errorbar(prop,slda[ind,7], yerr=slda[ind,8],fmt='-<')
#pylab.plot(prop,dmr[ind,7],'-^')
ax5 = plt.subplot(gs2[:-1, -1])
#pylab.errorbar(prop,ccn[ind,5], yerr=ccn[ind,6],fmt='-*', color='r')
pylab.errorbar(prop,vccn[ind,5], yerr=vccn[ind,6],fmt='-*', color='r')
pylab.errorbar(prop,slda[ind,5], yerr=slda[ind,6],fmt='-<',color='b')
#pylab.ylim([-9.6, -8.3])
pylab.yticks([-1.83,-1.85,-1.87])
pylab.ylabel('log-likelihood')
ax5.get_xaxis().set_ticklabels([])
ax6 = plt.subplot(gs2[-1, -1])
pylab.errorbar(prop,dmr[ind,5],yerr=dmr[ind,6],fmt='-o',color = 'g')
pylab.yticks([-4.5,-3,-1.5])
pylab.text(1.19, 2.02, r'$10^7$', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
#ax.set_xscale('log')
pylab.ylabel('log-likelihood')
pylab.xlabel('Label proportion')
pylab.show()
pylab.savefig('%s_ccr.pdf' %dataset, bbox_inches='tight')

