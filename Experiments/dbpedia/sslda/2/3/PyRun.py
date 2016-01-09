import numpy as np
import os, re

seed0 = 100000001 + 100*int(os.getcwd().split('/')[-2])
np.random.seed(seed0)

datapath = '../../../../../data/dbpedia'
codepath = '../../../../../Codes/class-slda'

trdocfile = datapath + '/train-data.dat'
tdocfile = datapath + '/test-data.dat'
vdocfile = datapath + '/valid-data.dat'
tlblfile = datapath + '/test-label.dat'
vlblfile = datapath + '/valid-label.dat'
vocabfile = datapath + '/vocabs.txt'

alpha = 0.1
done_num = -1
if False:
	temp = np.loadtxt('results.txt')
	done_num = temp.shape[0]
	try:
		tt = temp.shape[1]
	except IndexError:
		done_num = 1
else:
	fp = open('results.txt','w')
	fp.close()

Dtr = sum(1 for line in open(trdocfile))
N = sum(1 for line in open(vocabfile))
Dt = sum(1 for line in open(tdocfile))
Dv = sum(1 for line in open(vdocfile))

path = 'dir'
os.system('mkdir -p %s' %path)
M = 50
C = 10
for lpnum, lp in enumerate([0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]):
	dirnum = int(os.getcwd().split('/')[-1])-1
	if lpnum != dirnum:
		continue

	trlblfile = datapath + '/train_'+str(lp)+'-label.dat'

	for lambda_num, Lambda in enumerate([0.01, 0.1, 1.0, 10.]):
		print(lambda_num, done_num)
		if lambda_num < done_num:
			continue

		# write settings file
		fp = open('settings.txt','w+')
		fp.write('var max iter 50\nvar convergence 5e-4\nem max iter 100\nem convergence 5e-4')
		fp.write('L2 penalty ' + str(Lambda)+'\nalpha estimate')
		fp.close()

		# train 
		seed = np.random.randint(seed0)
		cmdtxt = codepath+'/SLDA est ' + trdocfile + ' ' + trlblfile + ' settings.txt ' + str(alpha) + ' ' + str(M) + ' seeded ' + path + ' ' + str(seed)
		print(cmdtxt)
		os.system(cmdtxt)

		# measure trccr
		pred = np.loadtxt(path+'/final-labels.dat',dtype = np.int32)
		trgt = np.loadtxt(trlblfile,delimiter = ',',dtype = np.int32)
		cnt = 0.0
		trccr = 0.0
		for d in range(pred.shape[0]):
			gtlbl = trgt[d,1]
			if trgt[d,0] != -1:
				continue
			if gtlbl == pred[d]:
				trccr += 1.0
			cnt += 1.0
		trccr = trccr/cnt
		print(trccr)

		# valid
		cmdtxt = codepath+'/SLDA inf ' + vdocfile + ' ' + vlblfile + ' settings.txt ' + path + '/final.model '  + path
		os.system(cmdtxt)

		# measure ccr
		pred = np.loadtxt(path+'/inf-labels.dat', dtype = np.int32)
		gt = np.loadtxt(vlblfile,delimiter = ',',dtype = np.int32)
		cnt = 0.0
		vccr = 0.0
		for d in range(pred.shape[0]):
			gtlbl = gt[d,1]
			if gt[d,0] != -1:
				continue
			if gtlbl == pred[d]:
				vccr += 1.0
			cnt += 1.0
		vccr = vccr/cnt


		# test
		cmdtxt = codepath+'/SLDA inf ' + tdocfile + ' ' + tlblfile + ' settings.txt ' + path + '/final.model '  + path
		os.system(cmdtxt)

		# measure ccr
		pred = np.loadtxt(path+'/inf-labels.dat', dtype = np.int32)
		gt = np.loadtxt(tlblfile,delimiter = ',',dtype = np.int32)
		cnt = 0.0
		tccr = 0.0
		correct_macro = np.zeros(C)
		num_macro = np.zeros(C)
		for d in range(pred.shape[0]):
			gtlbl = gt[d,1]
			if gt[d,0] != -1:
				continue
			if gtlbl == pred[d]:
				tccr += 1.0
				correct_macro[gtlbl] += 1.0
			cnt += 1.0
			num_macro[gtlbl] += 1.0
		tccr = tccr/cnt
		tccr_macro = np.mean(correct_macro/num_macro)


		## compute wrd lkh
		# load theta
		theta = np.loadtxt(path+'/inf-gamma.dat')
		theta /= np.sum(theta,1).reshape(-1,1)
		# load beta
		fp = open(path+'/final.model.text')
		fp.readline()
		fp.readline()
		txt = fp.readline()
		nterms = int(txt.split(": ")[1].split('\n')[0])
		beta = np.zeros((M,nterms))
		fp.readline() #betas:
		fp.readline() #betas:
		for j in range(M):
			ln = fp.readline()
			beta[j,:] = np.array([np.exp(float(x)) for x in ln.split()])
			beta[j,:] /= np.sum(beta[j,:])

		# comp lkh
		lkh = 0.0;
		for d,doc in enumerate(open(tdocfile).readlines()):
			wrds = re.findall(r'([0-9]*):[0-9]*', doc)
			cnts = re.findall(r'[0-9]*:([0-9]*)', doc)
			for n,ws in enumerate(wrds):
				w = int(ws)
				lkh += float(cnts[n])*np.log(np.dot(theta[d,:],beta[:,w]))


		# Transductive Inference 
		'''transdocfile = path + '/trans-doc.dat'
		translblfile = path + '/trans-lbl.dat'
		cmdtxt = 'cat %s %s > %s' %(trdocfile, tdocfile, transdocfile)
		os.system(cmdtxt)
		cmdtxt = 'cat %s %s > %s' %(trlblfile, tlblfile, translblfile)
		os.system(cmdtxt)

		seed = np.random.randint(seed0)
		cmdtxt = codepath+'/SLDA est ' + transdocfile + ' ' + translblfile + ' settings.txt ' + str(alpha) + ' ' + str(M) + ' seeded ' + path + ' ' + str(seed)
		os.system(cmdtxt)
		# measure ccr
		pred = np.loadtxt(path+'/final-labels.dat', dtype = np.int32)
		gt = np.loadtxt(tlblfile,delimiter = ',',dtype = np.int32)
		cnt = 0.0
		transccr = 0.0
		correct_macro = np.zeros(C)
		num_macro = np.zeros(C)
		for d in range(Dt):
			gtlbl = gt[d,1]
			if gt[d,0] != -1:
				continue
			if gtlbl == pred[d+Dtr]:
				transccr += 1.0
				correct_macro[gtlbl] += 1.0
			cnt += 1.0
			num_macro[gtlbl] += 1.0
		transccr = transccr/cnt
		transccr_macro = np.mean(correct_macro/num_macro)

		#trans_lkh
		# load theta
		theta = np.loadtxt(path+'/final.gamma')
		theta /= np.sum(theta,1).reshape(-1,1)
		# load beta
		fp = open(path+'/final.model.text')
		fp.readline()
		fp.readline()
		txt = fp.readline()	
		nterms = int(txt.split(": ")[1].split('\n')[0])
		beta = np.zeros((M,nterms))
		fp.readline() #betas:
		fp.readline() #betas:
		for j in range(M):
			ln = fp.readline()
			beta[j,:] = np.array([np.exp(float(x)) for x in ln.split()])
			beta[j,:] /= np.sum(beta[j,:])

		# comp lkh
		translkh = 0.0;
		for d,doc in enumerate(open(tdocfile).readlines()):
			wrds = re.findall(r'([0-9]*):[0-9]*', doc)
			cnts = re.findall(r'[0-9]*:([0-9]*)', doc)
			for n,ws in enumerate(wrds):
				w = int(ws)
				translkh += float(cnts[n])*np.log(np.dot(theta[d+Dtr,:],beta[:,w]))

		'''
		transccr = 0
		translkh = 0
		transccr_macro = 0
		# save results
		fp = open('results.txt','a')
		fp.write("%f %f %d %f %f %f %f %e %f %f %e\n" %(lp, Lambda, M, trccr, vccr, tccr, tccr_macro, lkh, transccr, transccr_macro, translkh))
		fp.close()

os.system('rm -rf %s' %path)
