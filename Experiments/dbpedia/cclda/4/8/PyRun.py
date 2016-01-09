import numpy as np
import os

seed0 = 100000001 + int(os.getcwd().split('/')[-2])
np.random.seed(seed0)

datapath = '../../../../../data/dbpedia'
codepath = '../../../../../Codes/CCLDA'
ldacode = '../../../../../Codes/LDA_VB/lda_vb'

trdocfile = datapath + '/train-data.dat'
tdocfile = datapath + '/test-data.dat'
vdocfile = datapath + '/valid-data.dat'
tlblfile = datapath + '/test-label.dat'
vlblfile = datapath + '/valid-label.dat'
N = len(open(datapath + '/vocabs.txt').readlines())

C = 10
done_num = -1
if False:
	temp = np.loadtxt('results.txt')
	done_num = temp.shape[0] 
else:
	fp = open('results.txt','w')
	fp.close()

Dtr = sum(1 for line in open(trdocfile))
Dt = sum(1 for line in open(tdocfile))

M = 50
for lpnum, lp in enumerate([0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]):
	dirnum = int(os.getcwd().split('/')[-1])-1
	if lpnum != dirnum:
		continue

	path = 'dir' + str(lp)
	os.system('mkdir -p '+path)
	trlblfile = datapath + '/train_'+str(lp)+'-label.dat'

	# only keep labeled samples
	train_docs = path + '/trdocs.dat'
	train_lbls = path + '/trlbls.dat'
	fpdoc = open(train_docs, 'w')
	fplbl = open(train_lbls, 'w')
	fp2 = open(trdocfile, 'r')
	fp3 = open(trlblfile, 'r')
	dtr = 0
	lblind = {}
	while True:
		docln = fp2.readline()
		if len(docln) == 0:
			break
		lblln = fp3.readline()
		lbl = int(lblln.split(',')[0])
		if lbl == -1:
			dtr += 1
			continue
		if lbl in lblind:
			lblind[lbl].append(dtr)
		else:
			lblind.update({lbl:[dtr]})
		dtr += 1
		fpdoc.write(docln)
		fplbl.write(lblln)
	fp2.close()
	fp2.close()
	fplbl.close()
	fpdoc.close()

	# run lda
	alpha = 1.0/M
	nu = 0.001
	s1 = np.random.randint(seed0)
	cmdtxt = ldacode + ' ' + str(s1) + ' est ' + trdocfile + ' ' + str(M) + ' seeded '+ path + '/vblda'
	cmdtxt += ' ' + str(alpha) + ' ' + str(nu)
	os.system(cmdtxt)
	os.system('rm %s/vblda/0*' %path)


	# LDA topics for initialization
	beta = np.loadtxt(path + '/vblda/final.beta')
	np.savetxt(path + '/init.beta',np.log(beta/np.sum(beta,0)),'%f')
	'''alpha = np.loadtxt(path + '/vblda/final.theta')
	alpha /= np.sum(alpha, 1).reshape(-1,1)
	initalpha = np.zeros((M,C))
	for c in range(C):
		means = np.mean(alpha[lblind[c],:], 0)
		vars = np.var(alpha[lblind[c],:], 0)
		if np.any(vars == 0):
			initalpha[:,c] = means/np.sum(means)
		else:
			initalpha[:,c] = means*(means*(1-means)/vars - 1.0)
			initalpha[:,c] += 0.05 + np.min(initalpha[:,c])

	fp = open(path + '/init.alpha','w')
	for c in range(C):
		#temp = np.mean(alpha[lblind[c],:], 0)
		#print(np.sum(temp))
		for j in range(M):
			fp.write('%f ' %np.log(initalpha[j,c]))
		fp.write('\n')
	fp.close()'''



	trdocfile = train_docs
	trlblfile = train_lbls
	# train 
	seed = np.random.randint(seed0)
	cmdtxt = codepath+'/ccLDA ' + str(seed) + ' train ' + trdocfile + ' ' + trlblfile
	cmdtxt +=  ' ' + str(C) + ' ' + str(M) +  ' '+str(N)+' load ' + path + ' '+ path + '/init'
	print(cmdtxt)
	os.system(cmdtxt)


	# valid
	seed = np.random.randint(seed0)
	cmdtxt = codepath+'/ccLDA ' + str(seed) + ' test ' + vdocfile + ' ' +vlblfile +' '+ path + '/final ' + path
	os.system(cmdtxt)

	# measure ccr
	theta_test = np.loadtxt(path+'/testfinal.mu')
	tgt = np.loadtxt(vlblfile,delimiter = ',',dtype = np.int32)
	cnt = 0.0
	vccr = 0.0
	for d in range(theta_test.shape[0]):		
		gtlbl = tgt[d,1]	
		pred = np.argmax(theta_test[d,:])
		if tgt[d,0] != -1:
			continue
		if gtlbl == pred:
			vccr += 1.0
		cnt += 1.0
	vccr = vccr/cnt


	# test
	seed = np.random.randint(seed0)
	cmdtxt = codepath+'/ccLDA ' + str(seed) + ' test ' + tdocfile + ' ' +tlblfile + ' ' + path + '/final ' + path
	print(cmdtxt)
	os.system(cmdtxt)
	
	# read test lkh
	lk = np.loadtxt(path+'/test-lhood.dat')
	Etlk = lk[4]

	# measure ccr
	theta_test = np.loadtxt(path+'/testfinal.mu')
	tgt = np.loadtxt(tlblfile,delimiter = ',',dtype = np.int32)
	cnt = 0.0
	tccr = 0.0
	correct_macro = np.zeros(C)
	num_macro = np.zeros(C)
	for d in range(theta_test.shape[0]):		
		gtlbl = tgt[d,1]	
		pred = np.argmax(theta_test[d,:])
		if tgt[d,0] != -1:
			continue
		if gtlbl == pred:
			tccr += 1.0
			correct_macro[gtlbl] += 1.0
		cnt += 1.0
		num_macro[gtlbl] += 1.0
	tccr = tccr/cnt
	tccr_macro = np.mean(correct_macro/num_macro)

	# save results
	fp = open('results.txt','a')
	fp.write("%f %d %f %f %f %e\n" %(lp, M, vccr, tccr, tccr_macro, Etlk))
	fp.close()

os.system('rm -r %s' %path)

