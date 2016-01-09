import numpy as np
import sys

#def collect_results():
dataset = sys.argv[1]
method = sys.argv[2]

num = 9
final_res = np.zeros((num,15))

T = 5

for i in range(num):
	tccrT = np.zeros(T)
	trccrT = np.zeros(T)
	tccr_macroT = np.zeros(T)
	trccr_macroT = np.zeros(T)
	translkhT = np.zeros(T)
	lkhT = np.zeros(T)
	for t in range(T):
		print('../Experiments/%s/%s/%d/%d/results.txt'%(dataset,method,t+1,i+1))
		res = np.loadtxt('../Experiments/%s/%s/%d/%d/results.txt'%(dataset,method,t+1,i+1))
		if method == 'cclda':
			p = res[0]
			tccrT[t] = res[3]
			tccr_macroT[t] = res[4]
			lkhT[t] = res[5]
		#elif method == 'svm':
			#p = res[0]
			#tccr = res[1]
			#trccr = 0
			#tlkh = 0
		elif method == 'mcctm':
			p = res[0,0]
			vmax = np.argmax(res[:,-7])
			tccrT[t] = res[vmax,-6]
			tccr_macroT[t] = res[vmax,-5]
			lkhT[t] = res[vmax,-4]
			trccrT[t] = res[np.argmax(res[:,-8]),-3] # transductive ccr
			trccr_macroT[t] = res[np.argmax(res[:,-8]),-2]
			translkhT[t] = res[np.argmax(res[:,-8]),-1]
		elif method == 'sslda':
			if len(res.shape)==2:
				p = res[0,0]
				vmax = np.argmax(res[:,-7])
				tccrT[t] = res[vmax,-6]
				tccr_macroT[t] = res[vmax,-5]
				lkhT[t] = res[vmax,-4]
				best_tr = np.argmax(res[:,-8])
			elif len(res.shape)==1:
				p = res[0]
				tccrT[t] = res[-6]
				tccr_macroT[t] = res[-5]
				lkhT[t] = res[-4]
				best_tr = 0
			trres = np.loadtxt('../Experiments/%s/%s-trans/%d/%d/results.txt'%(dataset,method,t+1,i+1))
			if len(trres.shape)==2:
				trccrT[t] = trres[best_tr,-3] 
				trccr_macroT[t] = trres[best_tr,-2] 
				translkhT[t] = trres[best_tr,-1] 
			elif len(trres.shape)==1:
				trccrT[t] = trres[-3] # transductive ccr
				trccr_macroT[t] = trres[-2] # transductive ccr
				translkhT[t] = trres[-1] # transductive ccr

	
	final_res[i,0] = p
	final_res[i,1:3] = [np.mean(tccrT), np.std(tccrT)]
	final_res[i,3:5] = [np.mean(tccr_macroT), np.std(tccr_macroT)]
	final_res[i,5:7] = [np.mean(lkhT), np.std(lkhT)]
	final_res[i,7:9] = [np.mean(trccrT), np.std(trccrT)]
	final_res[i,9:11] = [np.mean(trccr_macroT), np.std(trccr_macroT)]
	final_res[i,11:13] = [np.mean(translkhT), np.std(translkhT)]

np.savetxt('%s_%s.txt' %(dataset,method), final_res, '%f')



#if __name__ == '__main__':
#	collect_results()
