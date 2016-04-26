import pylab
import numpy as np
import matplotlib.gridspec as gridspec


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
dataset_list = ['20ng', 'ag','yahoo','dbpedia']
vccn = []
slda = []
for dataset in dataset_list:
	vccn.append(np.loadtxt('%s_vmcctm.txt'%dataset))
	slda.append(np.loadtxt('%s_sslda.txt'%dataset))

#ind = np.array([1,2,4,6,7,8,9])-1
ind = np.array([1,2,3,4,5,6,7,8,9])-1
prop = vccn[0][:,0][ind]

# test
fig = pylab.figure(figsize=(6.5, 2.5))

gs2 = gridspec.GridSpec(2, 2)
gs2.update(left=0.08, right=0.98, hspace=0.3, wspace = 0.24, bottom = 0.18, top = 0.9)
for dt,dataset in enumerate(dataset_list):
	ax = plt.subplot(gs2[dt/2, dt%2])
	if (dt < 2):
		ax.get_xaxis().set_ticks([])
	ax.set_title(['(a)', '(b)', '(c)', '(d)'][dt])
	#pylab.errorbar(prop,vccn[dt][ind,9], yerr=vccn[dt][ind,10],fmt='-*', color='r')
	#pylab.errorbar(prop,slda[dt][ind,9], yerr=slda[dt][ind,10],fmt='-<',color='b')
	pylab.errorbar(prop,vccn[dt][ind,1], yerr=vccn[dt][ind,2],fmt='-*', color='r')
	pylab.errorbar(prop,slda[dt][ind,1], yerr=slda[dt][ind,2],fmt='-<',color='b')
	pylab.ylabel('Test CCR')
	if dt > 1:
		pylab.xlabel('Label proportion')
pylab.legend(['MCCTM','ssLDA'],loc = 4,ncol=2)
pylab.show()
pylab.savefig('transd.pdf', bbox_inches='tight')

