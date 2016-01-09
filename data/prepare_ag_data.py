import re, os
import numpy as np
import urllib2
from itertools import groupby
from nltk.stem.porter import *
stemmer = PorterStemmer()
np.random.seed(100000001)

# keeping only 10 classes
Allclasses = ['World', 'Sports', 'Business', 'Sci/Tech', 'Entertainment', 'Software and Developement']
Allclasses = set(Allclasses)

# download and extract http://www.di.unipi.it/~gulli/newsspace200.xml.bz
request = urllib2.urlopen('http://www.di.unipi.it/~gulli/newsspace200.xml.bz')
output = open("ag/newsspace200.xml.bz","w")
output.write(request.read())
output.close()
os.system('bzip2 -d ag/newsspace200.xml.bz ag/')

# download and read stopword list
import urllib2
request = urllib2.urlopen('http://www.textfixer.com/resources/common-english-words.txt')
output = open("ag/stopwords.txt","w")
output.write(request.read())
output.close()

stpfile = open('ag/stopwords.txt', 'r')
stpwrdlist = stpfile.readlines()[0].split(',')
stpfile.close()
stpwrds = set(stpwrdlist)

# crawl all folders

lbl_map = {}
C = 0
rawvocabs = set()
documents = list() 
doc_lbls = list()
num_doc_lbl = list()
# some reg exp patterns
remove_pattern = re.compile(r'in article .*? writes', re.IGNORECASE)
title_pattern = re.compile(r'<title>([^<>]*?)</title>')
description_pattern = re.compile(r'<description>([^<>]*?)</description>')
lbl_pattern = re.compile(r'<category>([^<>]*?)</category>')
nonchar = re.compile(r'[^a-zA-Z]')

# read training docs
filename = 'ag/newsspace200.xml'
fp = open(filename)
d = 0
while True:
	ln = fp.readline()
	if len(ln) == 0:
		break
	d += 1
	if d%50000==0:
		print(d)
	lbl = lbl_pattern.findall(ln)
	if len(lbl)==0:
		continue
	lbl = lbl[0]

	if lbl not in Allclasses:
		continue
	try:
		lbl_num = lbl_map[lbl]
	except KeyError:
		lbl_num = C
		lbl_map.update({lbl:C})
		C += 1	

	maintxt = title_pattern.findall(ln)[0] + ' ' + description_pattern.findall(ln)[0]

	try:
		num_this_lbl = num_doc_lbl[lbl_num]
	except IndexError:
		num_this_lbl = 0
	if (num_this_lbl >= 5000) or (np.random.random() > 0.1):
		continue

	lntxt = [x.lower() for x in maintxt.split() if len(x)>=3 and nonchar.search(x)==None and x not in stpwrds] #remove words with <= 2
	rawvocabs = rawvocabs.union(set(lntxt))
	documents.extend(['%s' %(' '.join(lntxt))])
	doc_lbls.append(lbl_num)
	try:
		num_doc_lbl[lbl_num] += 1.0
	except IndexError:
		num_doc_lbl.append(1.0)			
	#if d>2000:
	#	break
print('Done with reading docs; %d classes, %d docs' %(C,len(doc_lbls)))

 
print('Make train/test/valid splits')
doc_lbls = np.array(doc_lbls)
train_ind = []
valid_ind = []
test_ind = []
for c in range(C):
	ind = np.where(doc_lbls == c)[0]
	num = int(len(ind)*0.7)
	trind = np.random.choice(ind, num, replace = False)
	left = np.setdiff1d(ind, trind)
	num = int(len(left)*0.1)
	vind = np.random.choice(left, num, replace = False)
	tind = np.setdiff1d(left, vind)
	train_ind.extend(list(trind))
	valid_ind.extend(list(vind))
	test_ind.extend(list(tind))


print('Creating stemmed vocab list')    
N = 0
vocabs = {}
stem_to_vocabs = {}
trrawvocabs = set()
for d in train_ind:
	trrawvocabs = trrawvocabs.union(set(documents[d].split()))

for w in trrawvocabs:
	#if w not in stem_to_vocabs.keys():
	sw = stemmer.stem(w)
	try:
		wnum = vocabs[sw]	
	except KeyError:
		wnum =  N
		vocabs.update({sw:N})
		N += 1
	#stem_to_vocabs.update({w:wnum})

# compute word counts (to remove less frequent ones)
vocabcnts = np.zeros(N)
for d in train_ind:
	docraw = documents[d]
	for wrd in docraw.split():
		try:
			w = stemmer.stem(wrd)#stem_to_vocabs[wrd]
			vocabcnts[vocabs[w]] += 1.0
		except KeyError:
			continue

# remove words with <= 3 occurrences
N = 0
vocabs_reduced = {}
import operator
sorted_dic = sorted(vocabs.items(), key=operator.itemgetter(1)) # sort by wrd index
for vpair in sorted_dic[1:]: # wrd 0 is ''; skipping that; don't know where it came from
	w = vpair[0]
	n_old = vpair[1]
	if vocabcnts[n_old] <= 3:
		continue
	vocabs_reduced.update({w:N})
	N += 1 
vocabs = vocabs_reduced.copy()

for w in rawvocabs:
	if w not in stem_to_vocabs.keys():
		sw = stemmer.stem(w)
		try:
			wnum = vocabs[sw]	
			stem_to_vocabs.update({w:wnum})
		except KeyError:
			continue



print('Writing dictionary')
import operator
fp = open('ag/vocabs.txt','w')
for vv in sorted(vocabs.items(), key = operator.itemgetter(1)):
	fp.write('%s, %d\n' %(vv[0], vv[1]))
fp.close()
print('total words: %d, total training words: %d, final vocabs(stemmed): %d, final words: %d'\
	%(len(rawvocabs), len(trrawvocabs), len(vocabs), len(stem_to_vocabs)))


print('Writing docs')
fptr = open('ag/train-data.dat','w')
train_lbls = []
for d in train_ind:
	docraw = documents[d]
	docwrds = []
	for wrd in docraw.split():
		try:
			w = stem_to_vocabs[wrd]
			docwrds.append(w)
		except KeyError:
			continue
	grouped_docwrds = [list(g) for k,g in groupby(sorted(docwrds))]		
	nd = len(grouped_docwrds)
	txt = str(nd)
	if nd < 5:
		continue
	for wlist in grouped_docwrds:
		txt += ' ' + '%d:%d' %(wlist[0], len(wlist))
	fptr.write(txt+'\n')
	train_lbls.append(doc_lbls[d])
fptr.close()


fpt = open('ag/test-data.dat','w')
fpv = open('ag/valid-data.dat','w')
fpt_lbl = open('ag/test-label.dat','w')
fpv_lbl = open('ag/valid-label.dat','w')
for d, doc in enumerate(documents):
	if d in train_ind:
		continue
	docraw = documents[d]
	docwrds = []
	for wrd in docraw.split():
		try:
			w = stem_to_vocabs[wrd]
			docwrds.append(w)
		except KeyError:
			continue
	grouped_docwrds = [list(g) for k,g in groupby(sorted(docwrds))]		
	nd = len(grouped_docwrds)
	txt = str(nd)
	if nd < 5:
		continue
	for wlist in grouped_docwrds:
		txt += ' ' + '%d:%d' %(wlist[0], len(wlist))

	if d in test_ind:
		fpt.write(txt+'\n')
		fpt_lbl.write('-1, %d\n' %doc_lbls[d])
	elif d in valid_ind:
		fpv.write(txt+'\n')
		fpv_lbl.write('-1, %d\n' %doc_lbls[d])
	
fpt.close()
fpv.close()
fpt_lbl.close()
fpv_lbl.close()

# generate semisupervised data sets
print('write label files')
portions = [0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
train_lbls = np.array(train_lbls)
for p in portions:
	lbled_ind = []
	for c in range(C):
		docinclass = np.where(train_lbls  == c)[0]
		num = max(1, np.floor(len(docinclass)*p))
		cind = np.random.choice(docinclass, num, replace=False)
		lbled_ind.extend(cind)

	fp = open('ag/train_%s-label.dat' %(str(p)),'w')
	for d, lb in enumerate(train_lbls ):
		if d not in lbled_ind:
			fp.write("%s, %s\n" %(-1,lb))
		else:
			fp.write("%s, %s\n" %(lb,lb))
	fp.close()



