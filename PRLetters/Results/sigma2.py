import pylab
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.cbook as cbook

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

def take_average(pp):
	props = [0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
	p = props.index(pp)
	out = np.zeros((10,7))
	tccr = np.zeros((7,5))
	trccr = np.zeros((7,5))
	lkh = np.zeros((7,5))
	vccr= np.zeros((7,5))
	for t in range(5):
		fname = '../../Experiments/20ng/vmcctm/%d/%d/results.txt' %(t+1,p+1)
		res = np.loadtxt(fname)
		if t == 0:
			out[0,:] = np.unique(res[:,1])
		for s, s2 in enumerate(out[0,:]):
			ind = np.where(res[:,1]==s2)[0]
			trmax = ind[np.argmax(res[ind,4])]
			
			tccr[s,t] = res[trmax,6]
			vccr[s,t] = res[trmax,5]
			trccr[s,t] = res[trmax,4]
			lkh[s,t] = res[trmax,8]
	out[1,:] = np.mean(tccr,1)
	out[2,:] = np.std(tccr,1)
	out[3,:] = np.mean(trccr,1)
	out[4,:] = np.std(trccr,1)
	out[5,:] = np.mean(lkh,1)
	out[6,:] = np.std(lkh,1)
	out[7,:] = np.mean(vccr,1)
	out[8,:] = np.std(vccr,1)

	return(out)

fig = pylab.figure(figsize=(6.5, 2.5))


res = take_average(0.9)
res2 = take_average(0.5)
res3 = take_average(0.2)
res4 = take_average(0.05)
res5 = take_average(0.01)
s2 = res[0,:]
ax = fig.add_subplot(1,2,1)
pylab.errorbar(s2,res[1,:], yerr=res[2,:],fmt='-+', color='r')
pylab.errorbar(s2,res2[1,:], yerr=res2[2,:],fmt='-*', color='k')
pylab.errorbar(s2,res3[1,:], yerr=res3[2,:],fmt='-<', color='b')
pylab.errorbar(s2,res4[1,:], yerr=res4[2,:],fmt='-o', color='g')

#pylab.errorbar(s2,res5[1,:], yerr=res5[2,:],fmt='-*', color='y')

ax.set_xscale('log')
pylab.ylabel('Test CCR')
pylab.xlabel('$\sigma^2$')
pylab.ylim([0.62,0.77])
ax = fig.add_subplot(1,2,2)
res /=1e6
res2 /=1e6
res3 /=1e6
res4 /=1e6
pylab.errorbar(s2,res[5,:], yerr=res[6,:],fmt='-+', color='r')
pylab.errorbar(s2,res2[5,:], yerr=res2[6,:],fmt='-*', color='k')
pylab.errorbar(s2,res3[5,:], yerr=res3[6,:],fmt='-<', color='b')
pylab.errorbar(s2,res4[5,:], yerr=res4[6,:],fmt='-o', color='g')
#pylab.errorbar(s2,res5[5,:], yerr=res5[6,:],fmt='-*', color='y')
pylab.legend(['0.9','0.5','0.2','0.05'],loc = 4,ncol=2)
pylab.yticks([-3.72, -3.67, -3.63])
pylab.text(0.01, 1.05, r'$10^6$', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
pylab.ylabel('log-likelihood')
pylab.xlabel('$\sigma^2$')
ax.set_xscale('log')
#ax.grid(b=True, which='both')

'''res = take_average(0.5)
s2 = res[0,:]
ax = fig.add_subplot(3,2,3)
pylab.errorbar(s2,res[1,:], yerr=res[2,:],fmt='-*', color='r')
ax.set_xscale('log')
#pylab.ylabel('Test CCR')
#pylab.xlabel('$\sigma^2$')
ax = fig.add_subplot(3,2,4)
pylab.errorbar(s2,res[5,:],yerr=res[6,:],fmt='-*', color='r')
#pylab.ylabel('log-likelihood')
#pylab.xlabel('$\sigma^2$')
pylab.suptitle('label proportion = 0.5')
ax.set_xscale('log')

res = take_average(0.05)
s2 = res[0,:]
ax = fig.add_subplot(3,2,5)
pylab.errorbar(s2,res[1,:], yerr=res[2,:],fmt='-*', color='r')
ax.set_xscale('log')
#pylab.ylabel('Test CCR')
pylab.xlabel('$\sigma^2$')
ax = fig.add_subplot(3,2,6)
pylab.errorbar(s2,res[5,:],yerr=res[6,:],fmt='-*', color='r')
pylab.ylabel('log-likelihood')
pylab.xlabel('$\sigma^2$')
ax.set_xscale('log')
pylab.suptitle('label proportion = 0.05')
'''
fig.tight_layout(pad = .5)
pylab.show()
pylab.savefig('sigma2.pdf' , bbox_inches='tight')

