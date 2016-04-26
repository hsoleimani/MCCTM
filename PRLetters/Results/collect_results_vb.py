import numpy as np
import sys

#def collect_results():
dataset = sys.argv[1]
#method = sys.argv[2]
method = 'vmcctm'

num = 9
final_res = np.zeros((num,15+2))

T = 5

for i in range(num):
	tccrT = np.zeros(T)
	trccrT = np.zeros(T)
	tccr_macroT = np.zeros(T)
	trccr_macroT = np.zeros(T)
	translkhT = np.zeros(T)
	lkhT = np.zeros(T)
	sigma2T = np.zeros(T)
	nuT = np.zeros(T)
	for t in range(T):
		print('%s/%s/%d/%d/results.txt'%(dataset,method,t+1,i+1))
		res = np.loadtxt('%s/%s/%d/%d/results.txt'%(dataset,method,t+1,i+1))

		p = res[0,0]
		vmax = np.argmax(res[:,-8]) #-8 is training error
		sigma2T[t] = res[vmax, 1]
		nuT[t] = res[vmax, 2]
		tccrT[t] = res[vmax,-6]
		tccr_macroT[t] = res[vmax,-5]
		lkhT[t] = res[vmax,-4]
		trccrT[t] = res[np.argmax(res[:,-8]),-3] # transductive ccr
		trccr_macroT[t] = res[np.argmax(res[:,-8]),-2]
		translkhT[t] = res[np.argmax(res[:,-8]),-1]
	
	final_res[i,0] = p
	final_res[i,1:3] = [np.mean(tccrT), np.std(tccrT)]
	final_res[i,3:5] = [np.mean(tccr_macroT), np.std(tccr_macroT)]
	final_res[i,5:7] = [np.mean(lkhT), np.std(lkhT)]
	final_res[i,7:9] = [np.mean(trccrT), np.std(trccrT)]
	final_res[i,9:11] = [np.mean(trccr_macroT), np.std(trccr_macroT)]
	final_res[i,11:13] = [np.mean(translkhT), np.std(translkhT)]
	final_res[i,13:15] = [np.mean(sigma2T), np.std(sigma2T)]
	final_res[i,15:17] = [np.mean(nuT), np.std(nuT)]

np.savetxt('Results/%s_%s.txt' %(dataset,method), final_res, '%f')



#if __name__ == '__main__':
#	collect_results()
