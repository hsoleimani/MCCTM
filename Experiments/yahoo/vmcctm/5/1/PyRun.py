import numpy as np
import os

seed0 = 100000001 + 100*int(os.getcwd().split('/')[-2])
np.random.seed(seed0)

datapath = '../../../../../data/yahoo'
codepath = '../../../../../Codes/VMCCTM'
ldacode = '../../../../../Codes/LDA_VB/lda_vb'

trdocfile = datapath + '/train-data.dat'
tdocfile = datapath + '/test-data.dat'
vdocfile = datapath + '/valid-data.dat'
tlblfile = datapath + '/test-label.dat'
vlblfile = datapath + '/valid-label.dat'

C = 10
done_num = -1
if True:
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
	for nunum, nu in enumerate([0.001, 0.01, 0.1, 1.0]):
		for ssnum, sigma2 in enumerate([0.001, 0.0015, 0.01, 0.025, 0.05, 0.1, 0.5]):
			
			if ssnum+nunum*7 < done_num:
				continue

			path = 'dir' + str(lp)
			os.system('mkdir -p '+path)
			trlblfile = datapath + '/train_'+str(lp)+'-label.dat'

			#************************************** init run of lda 
			# prepare training docs
			fp2 = open(trdocfile, 'r')
			fp3 = open(trlblfile, 'r')
			dtr = 0
			lblind = {}
			while True:
				docln = fp2.readline()
				if len(docln) == 0:
					break
				lbl = int(fp3.readline().split(',')[0])
				if lbl == -1:
					dtr += 1
					continue
				if lbl in lblind:
					lblind[lbl].append(dtr)
				else:
					lblind.update({lbl:[dtr]})
				dtr += 1
			fp2.close()
			fp2.close()


			# run lda
			alpha = 1.0/M
			#nu = 1.0
			s1 = np.random.randint(seed0)
			cmdtxt = ldacode + ' ' + str(s1) + ' est ' + trdocfile + ' ' + str(M) + ' seeded '+ path + '/vblda'
			cmdtxt += ' ' + str(alpha) + ' ' + str(nu)
			if ssnum == 0:
				os.system(cmdtxt)
				os.system('rm %s/vblda/0*' %path)
			

			# read files and save in ClassCondMix format
			#beta = np.loadtxt(path + '/vblda/final.beta')
			#np.savetxt(path + '/init.beta',beta,'%f')
			os.system('cp %s/vblda/final.beta %s/init.beta' %(path,path))
			#np.savetxt(path + '/init.beta',np.log(beta/np.sum(beta,0)),'%f')
			alpha = np.loadtxt(path + '/vblda/final.theta')
			alpha /= np.sum(alpha, 1).reshape(-1,1)
			initalpha = np.zeros((M,C))
			for c in range(C):
				initalpha[:,c] = np.mean(alpha[lblind[c],:], 0)
				
			
			fp = open(path + '/init.alpha','w')
			for j in range(M):
				#temp = np.mean(alpha[lblind[c],:], 0)
				#print(np.sum(temp))
				for c in range(C):
					fp.write('%f ' %np.log(initalpha[j,c]))
				fp.write('\n')
			fp.close()

			# train 
			#nu = 0.0001
			seed = np.random.randint(seed0)
			cmdtxt = codepath+'/MCCTM ' + str(seed) + ' train ' + trdocfile + ' ' + trlblfile
			cmdtxt +=  ' ' + str(C) + ' ' + str(M) +' ' + str(sigma2) + ' ' + str(nu) + ' load ' + path + ' ' + path + '/init'
			print(cmdtxt)
			os.system(cmdtxt)
			os.system('cp %s %s%d' %(path+'/likelihood.dat',path+'/likelihood.dat',ssnum))
			# measure trccr
			theta = np.loadtxt(path+'/final.mu')
			trgt = np.loadtxt(trlblfile,delimiter = ',',dtype = np.int32)
			cnt = 0.0
			trccr = 0.0
			for d in range(theta.shape[0]):		
				gtlbl = trgt[d,1]	
				pred = np.argmax(theta[d,:])
				if trgt[d,0] != -1:
					continue
				if gtlbl == pred:
					trccr += 1.0
				cnt += 1.0
			trccr = trccr/cnt
		
			# valid
			seed = np.random.randint(seed0)
			cmdtxt = codepath+'/MCCTM ' + str(seed) + ' test ' + vdocfile + ' ' + path + '/final ' + path
			print(cmdtxt)
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
			cmdtxt = codepath+'/MCCTM ' + str(seed) + ' test ' + tdocfile + ' ' + path + '/final ' + path
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
			print(tccr)
			
			#transductive
			transdocfile = path + '/trans-doc.dat'
			translblfile = path + '/trans-lbl.dat'
			cmdtxt = 'cat %s %s > %s' %(trdocfile, tdocfile, transdocfile)
			os.system(cmdtxt)
			cmdtxt = 'cat %s %s > %s' %(trlblfile, tlblfile, translblfile)
			os.system(cmdtxt)


			seed = np.random.randint(seed0)
			cmdtxt = codepath+'/MCCTM ' + str(seed) + ' train ' + transdocfile + ' ' + translblfile
			cmdtxt +=  ' ' + str(C) + ' ' + str(M) +' ' + str(sigma2) + ' ' + str(nu) + ' load ' + path + ' ' + path + '/init'
			print(cmdtxt)
			os.system(cmdtxt)
			# measure ccr
			theta = np.loadtxt(path+'/final.mu')
			trgt = np.loadtxt(tlblfile,delimiter = ',',dtype = np.int32)
			cnt = 0.0
			transccr = 0.0
			correct_macro = np.zeros(C)
			num_macro = np.zeros(C)
			for d in range(Dt):#theta.shape[0]):		
				gtlbl = trgt[d,1]	
				pred = np.argmax(theta[d+Dtr,:])
				if trgt[d,0] != -1:
					continue
				if gtlbl == pred:
					transccr += 1.0
					correct_macro[gtlbl] += 1.0
				cnt += 1.0
				num_macro[gtlbl] += 1.0
			transccr = transccr/cnt
			transccr_macro = np.mean(correct_macro/num_macro)
			
			#trans_lkh
			temp_lkh = np.loadtxt('%s/doclkh.dat' %path)
			translkh = np.sum(temp_lkh[Dtr:])
			# save results
			fp = open('results.txt','a')
			#fp.write(str(lp)+' '+str(sigma2)+''+str(M)+' '+str(trccr) + ' ' + str(tccr)+' '+str(Etlk)+'\n')
			fp.write("%f %f %f %d %f %f %f %f %e %f %f %e\n" %(lp, sigma2, nu, M, trccr, vccr, tccr, tccr_macro, Etlk, transccr, transccr_macro, translkh))
			fp.close()

	os.system('rm -r %s' %path)
