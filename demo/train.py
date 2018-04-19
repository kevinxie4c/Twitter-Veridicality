# Authors: Sandesh Swamy, Alan Ritter, and Marie-Catherine de Marneffe
# Copyright, 2017 
# Demo for paper in EMNLP 2017.
import sys
import os
import csv
import re

try:
	import json
except ImportError:
	import simplejson as json
from twitter import Twitter, OAuth, TwitterHTTPError, TwitterStream

#things needed to run NER in background 

import subprocess
import platform
import time
import codecs
import redis
import nltk

from signal import *

from optparse import OptionParser

BASE_DIR = 'twitter_nlp.jar'
print os.environ
if os.environ.has_key('TWITTER_NLP'):
	BASE_DIR = os.environ['TWITTER_NLP']

sys.path.append('%s/python' % (BASE_DIR))
sys.path.append('%s/python/ner' % (BASE_DIR))
sys.path.append('%s/hbc/python' % (BASE_DIR))

import Features
import twokenize
from LdaFeatures import LdaFeatures
from Dictionaries import Dictionaries
from Vocab import Vocab

sys.path.append('%s/python' % (BASE_DIR))
sys.path.append('%s/python/cap' % (BASE_DIR))

print sys.path
import numpy as np
import cap_classifier
import pos_tagger_stdin
import chunk_tagger_stdin
import event_tagger_stdin

from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
import csv
import ast
from Vocab import *
import pickle
###########################################################
# Helper functions
###########################################################
class LRmulticlass(object):
		def __init__(self):
				self.model = None
		
		def json2Vocab(self, jsonInstance):
				vocabd = {}
				for k in jsonInstance.keys():
						vocabd[self.vocab.GetID(k)] = jsonInstance[k]
				return vocabd
		
		def json2Vector(self, jsonInstance):
				result = np.zeros(self.vocab.GetVocabSize())
				for k in jsonInstance.keys():
						if self.vocab.GetID(k) > 0:
								result[self.vocab.GetID(k)-1] = jsonInstance[k]
				return result
		
		def Train(self, xmat, yvec):
				lrmulti = LogisticRegression(solver='lbfgs', multi_class='multinomial')
				lrmulti.fit(xmat, yvec)
				self.model = lrmulti
		
		def Predict(self, jsonInstance):
				fsvoc = open("vocab_train.save", 'rb')
				self.vocab = pickle.load(fsvoc)
				self.vocab.Lock()
				#print self.vocab.GetVocabSize()
				return self.model.predict(self.json2Vector(jsonInstance).reshape(1,-1))
		
		def PredictProba(self, jsonInstance):
				return self.model.predict_proba(self.json2Vector(jsonInstance).reshape(1,-1))
		
		def printWeights(self, outFile):
				fwout = open(outFile, 'w')
				classes = self.model.coef_.shape[0]
				for i in range(classes):
						fwout.write("Class %s\n" % i)
						curCatWeights = self.model.coef_[i]
						for j in np.argsort(curCatWeights):
								try:
										fwout.write("%s\t%s\n" % (self.vocab.GetWord(j+1), curCatWeights[j]))
								except KeyError:
										pass


def isAscii(s): 
	try:		
		s.decode('ascii')
	except Exception:
		return False
	return True			   

def setFeature(features, feature):
	if isAscii(feature):
		features[feature] = 1

def computeEntityFeaturesTarget1(tweet, tags, features):
		tgtidx = [i for i,x in enumerate(tweet) if x == "TARGET1"]
		for idx in tgtidx:
				for k in [1, 2, 3, 4]:
						tarLeftContext = ""
						tarRightContext = ""
						leftContext = []
						rightContext = []
						for tidx in range(max(0, idx-k), idx):
								leftContext.append(tweet[tidx])
						for tidx in range(idx+1, min(idx+1+k, len(tweet))):
								rightContext.append(tweet[tidx])
						for item in leftContext:
								tarLeftContext += " " + item
						for item in rightContext:
								tarRightContext += " " + item
						tarLeftContext = tarLeftContext.strip()
						tarRightContext = tarRightContext.strip()
						setFeature(features, "CONTEXT: TARGET1 %s" % tarRightContext)
						setFeature(features, "CONTEXT: %s TARGET1" % tarLeftContext)
						setFeature(features, "CONTEXT: %s TARGET1 %s" %(tarLeftContext, tarRightContext))

def computePairFeatures(t, k, tweet, tags, features):
		tidx = [i for i,x in enumerate(tweet) if x == t]
		kidx = [i for i,x in enumerate(tweet) if x.strip().lower() == k.lower()]
		for tid in tidx:
				for kid in kidx:
						fContext = []
						wordFeatures = ""
						if tid < kid:
								for w in range(tid+1, kid):
										fContext.append(tweet[w])
						else:
								for w in range(kid+1, tid):
										fContext.append(tweet[w])
						for item in fContext:
								wordFeatures += " " + item
						wordFeatures = wordFeatures.strip()
						if len(wordFeatures) > 0:
								if tid < kid:
										setFeature(features, "PAIR CONTEXT: %s %s KEYWORD" % (t,wordFeatures))
								else:
										setFeature(features, "PAIR CONTEXT: KEYWORD %s %s" % (wordFeatures,t))

def computeOpponentFeatures(tweet, tags, features):
		tgtidx = [i for i,x in enumerate(tweet) if x == "OPPONENT"]
		for idx in tgtidx:
				for k in [1, 2, 3, 4]:
						oppLeftContext = ""
						oppRightContext = ""
						leftContext = []
						rightContext = []
						for tidx in range(max(0, idx-k), idx):
								leftContext.append(tweet[tidx])
						for tidx in range(idx+1, min(idx+1+k, len(tweet))):
								rightContext.append(tweet[tidx])
						for item in leftContext:
								oppLeftContext += " " + item
						for item in rightContext:
								oppRightContext += " " + item
						oppLeftContext = oppLeftContext.strip()
						oppRightContext = oppRightContext.strip()
						setFeature(features, "OPP CONTEXT: OPPONENT %s" % oppRightContext)
						setFeature(features, "OPP CONTEXT: %s OPPONENT" % oppLeftContext)
						setFeature(features, "OPP CONTEXT: %s OPPONENT %s" %(oppLeftContext, oppRightContext))

def computeOpponentKeywordFeatures(o, k, tweet, tags, features):
		oidx = [i for i, x in enumerate(tweet) if x == o]
		kidx = [i for i,x in enumerate(tweet) if x.strip().lower() == k.lower()]
		for oid in oidx:
				for kid in kidx:
						fContext = []
						wordFeatures = ""
						if  oid<kid:
								for w in range(oid+1, kid):
										fContext.append(tweet[w])
						else:
								for w in range(kid+1, oid):
										fContext.append(tweet[w])
						for item in fContext:
								wordFeatures += " " + item
						wordFeatures = wordFeatures.strip()
						if len(wordFeatures) > 0:
								if oid < kid:
										setFeature(features, "PAIR CONTEXT: %s %s KEYWORD" % (o, wordFeatures))
								else:
										setFeature(features, "PAIR CONTEXT: KEYWORD %s %s" % (o, wordFeatures))

def EndsWithExclamation(tweet, tags, features):
		last = tweet[-2].strip()
		if last == "!": 
				setFeature(features, "Ends with !")											

def EndsWithQuestion(tweet, tags, features):
		last = tweet[-2].strip()
		if last == "?": 
				setFeature(features, "Ends with ?")

def EndsWithPeriod(tweet, tags, features):
		last = tweet[-2].strip()
		if last == ".": 
				setFeature(features, "Ends with .")

def containsQuestion(tweet, tags, features):
		for i in range(len(tweet)-2):
				if "?" in tweet[i]:
						setFeature(features, "Contains ?")
def containsExclamation(tweet, tags, features):
		for i in range(len(tweet)-2):
				if "!" in tweet[i]:
						setFeature(features, "Contains !")
						break							   

def entityHasNegation(t, k, tweet, tags, features):
		tidx = [i for i,x in enumerate(tweet) if x == t]
		kidx = [i for i,x in enumerate(tweet) if x.strip().lower() == k.lower()]
		tweetmod = nltk.sentiment.util.mark_negation(tweet)
		for tid in tidx:
				for kid in kidx:
						hasNegation = False
						if tid < kid:
								for w in range(tid, kid+1):
										if "NEG" in tweetmod[w]:
												hasNegation = True
												break
						else:
								for w in range(kid, tid+1):
										if "NEG" in tweetmod[w]:
												hasNegation = True
												break
						if hasNegation:
								setFeature(features, "Has Negation")					   

def distanceToKwd(entity, opp, kwd, tweet, tags, features):
		tidx = [i for i,x in enumerate(tweet) if x == entity]
		oppidx = [i for i,x in enumerate(tweet) if x == opp]
		kwdidx = [i for i,x in enumerate(tweet) if kwd in x.strip().lower()]
		print tweet	 
		print "target ids"
		print tidx	  
		print "opp ids" 
		print oppidx
		print "kwd ids" 
		print kwdidx	
		if len(tidx) > 0 and len(kwdidx) > 0:
				tdistance = abs(tidx[0]-kwdidx[0])
				odistance = 99
				if len(oppidx) > 0:
						odistance = abs(oppidx[0]-kwdidx[0])
				if tdistance < odistance:
						setFeature(features, "TARGET closer to KEYWORD")
				else:   
						setFeature(features, "OPPONENT closer to KEYWORD")
						
def getsegments(tweet, annots, tag, lower=False, getIndices=True):
	result = []
	start = None
	for i in range(len(tweet)):
		if annots[i] == "B-%s" % tag:
			if start != None:
				if getIndices:
					result.append((' '.join(tweet[start:i]), (start,i)))
				else:
					result.append(' '.join(tweet[start:i]))
			start = i
		elif annots[i] == 'O' and start != None:
			if getIndices:
				result.append((' '.join(tweet[start:i]), (start, i)))
			else:
				result.append(' '.join(tweet[start:i]))
			start = None
	if start != None:
		if getIndices:
			result.append((' '.join(tweet[start:i+1]), (start, i+1)))
		else:
			result.append(' '.join(tweet[start:i+1]))
	if lower:
		if getIndices:
			result = [(x[0].lower(), x[1]) for x in result]
		else:
			result = [(x.lower()) for x in result]
	return result

def modTweetTargetEnt1(tweet, indices):
		start = indices[0]
		end = indices[1]
		del tweet[start:end]
		tweet.insert(start, "TARGET1")
		return tweet

def modTweetTargetOpp(tweet, indices):
		start = indices[0]
		end = indices[1]
		del tweet[start:end]
		tweet.insert(start, "OPPONENT")
		return tweet

def removeHashTags(tweet):
		mtweet = []
		for word in tweet:
				mtweet.append(word.replace("#", ""))
		return mtweet

def reinstateHT(modtweet, tweet):
		for i in range(len(modtweet)):
				if not "TARGET" in modtweet[i] and not "OPPONENT" in modtweet[i]:
						modtweet[i] = tweet[i]
		ntweet = []
		for w in modtweet:
				ntweet.append(w)
		return ntweet

def collapseEntities(tweet, indices):
		start = indices[0]
		end = indices[1]
		del tweet[start:end]
		tweet.insert(start, "ENTITY")
		return tweet

def collapseEntityTags(tags, indices):
		start = indices[0]
		end = indices[1]
		del tags[start:end]
		tags.insert(start, "MOD")
		return tags

def modTweetTarTags(tags, indices):
		start = indices[0]
		end = indices[1]
		del tags[start:end]
		tags.insert(start, "MOD")
		return tags

def DFSwithsource(e, d, visited, path, words, source, keyword, store):
	#print "path before - " + path 
	#print visited
	visited.add(source)
#	print words[source-1][0]
	path += words[source-1][0].lower()
	#print keyword
	if keyword in path:
		if keyword == "win":
			path = path.split()
			path[0] = "TARGET1"																																																		 
			path[-1] = "KEYWORD"
		else:
			path = path.split()																																																		 
			path[0] = "TARGET1"
			path[-1] = "TARGET2"
		modPath = ""
		for i in range(len(path)):
			modPath += path[i] + " "
		modPath = modPath.strip()
		#print "path " + keyword + "\t" ,
		#print modPath + "\t" ,
		if modPath not in store:
			modPath = normalizePath(modPath)
			store.append(modPath)
		return
	for i,edge in enumerate(e[source]):
		if edge == 1 and i not in visited:
			if d[source][i] == 2:
				#path += " in "
				DFSwithsource(e, d, visited, path + " <- ", words, i, keyword, store)
			else:
				#path += " out "
				DFSwithsource(e, d, visited, path + " -> ", words, i, keyword, store)

def DFSwithsourceopponent(e, d, visited, path, words, source, keyword, store):
	#print "path before - " + path 
	#print visited
	visited.add(source)
#	print words[source-1][0]
	path += words[source-1][0].lower()
	#print keyword
	if keyword in path:
		if keyword == "win":
			path = path.split()
			path[0] = "OPPONENT"
			path[-1] = "KEYWORD"
		else:
			path = path.split()
			path[0] = "OPPONENT"
			path[-1] = "TARGET2"
		modPath = ""
		for i in range(len(path)):
			modPath += path[i] + " "
		modPath = modPath.strip()
		#print "path " + keyword + "\t" ,
		#print modPath + "\t" ,
		if modPath not in store:
			modPath = normalizePath(modPath)
			store.append(modPath)
		return
	for i,edge in enumerate(e[source]):
		if edge == 1 and i not in visited:
			if d[source][i] == 2:
				#path += " in "
				DFSwithsourceopponent(e, d, visited, path + " <- ", words, i, keyword, store)
			else:
				#path += " out "
				DFSwithsourceopponent(e, d, visited, path + " -> ", words, i, keyword, store)

def checkDanglingNegationWord(word):
	if word.lower().strip() == "won't" or word.lower().strip() == "doesn't" or word.lower().strip() == "not" or word.lower().strip() == "'t" or word.lower().strip() == "cannot" or word.lower().strip() == "can't" or word.lower().strip() == "couldn't":
		return True
	return False


def danglingNegation(matrix, visited, tweetWords, kwd, store):#TODO pass adjacency matrix and tweet to get the index of the words connected to keyword
	#first find index of keyword
	winIndices = []
	for i in range(len(tweetWords)):
		if "win" == tweetWords[i][0].strip().lower():
			winIndices.append(i+1) #store index of keyword
	#once we have all the winIndices, check matrix for that particular row and find one-hop words
	for item in winIndices:
		getRow = matrix[item]
		for idx, val in enumerate(getRow):
			if val == 1: #connected
				#need to check word[idx-1]
				isDangling = checkDanglingNegationWord(tweetWords[idx-1][0])
				if isDangling:
					store.append("Dangling Negation")
					break

def normalizePath(path):
	# for the input path provided, convert to standard form
	if "cannot" in path or "cant" in path or "can't" in path:
		normpath = "can -> not"
		if "cannot" in path:
			path = path.replace("cannot", normpath)
		elif "cant" in path:
			path = path.replace("cant", normpath)
		else:
			path = path.replace("can't", normpath)
	elif "wouldn't" in path or "wouldnt" in path:
		normpath = "would -> not"
		if "wouldn't" in path:
			path = path.replace("wouldn't", normpath)
		else:
			path = path.replace("wouldnt", normpath)
	elif "didn't" in path or "didnt" in path or "dint" in path:
		normpath = "did -> not"
		if "didn't" in path:
			path = path.replace("didn't", normpath)
		elif "didnt" in path:
			path = path.replace("didnt", normpath)
		else:
			path = path.replace("dint", normpath)
	elif "won't" in path or "wont" in path:
		normpath = "will -> not"
		if "won't" in path:
			path = path.replace("won't", normpath)
		else:
			path = path.replace("wont", normpath)
	elif "doesn't" in path or "doesnt" in path:
		normpath = "does -> not"
		if "doesn't" in path:
			path = path.replace("doesn't", normpath)
		else:
			path = path.replace("doesnt", normpath)
	elif "don't" in path or "dont" in path:
		normpath = "do -> not"
		if "don't" in path:
			path = path.replace("don't", normpath)
		else:
			path = path.replace("dont", normpath)

	return path

def GetNer(ner_model):
	#return subprocess.Popen('java -Xmx256m -cp %s/mallet-2.0.6/lib/mallet-deps.jar:%s/mallet-2.0.6/class cc.mallet.fst.SimpleTaggerStdin --weights sparse --model-file %s/models/ner/%s' % (BASE_DIR, BASE_DIR, BASE_DIR, ner_model),
	return subprocess.Popen('java -Xmx512m -cp %s/mallet-2.0.6/lib/mallet-deps.jar:%s/mallet-2.0.6/class cc.mallet.fst.SimpleTaggerStdin --weights sparse --model-file %s/models/ner/%s' % (BASE_DIR, BASE_DIR, BASE_DIR, ner_model), shell=True, close_fds=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE)

def GetLLda():
	return subprocess.Popen('%s/hbc/models/LabeledLDA_infer_stdin.out %s/hbc/data/combined.docs.hbc %s/hbc/data/combined.z.hbc 100 100' % (BASE_DIR, BASE_DIR, BASE_DIR), shell=True, close_fds=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE)

def recomputeScores():
	for item in scores:
		computedScores[item] = (scores[item]+ 1)/((scores[item] + len(scores))*1.0)

def displayScores():
	for item in computedScores:
		print str(item) + " : " + str(computedScores[item])


posTagger = pos_tagger_stdin.PosTagger()
chunkTagger = chunk_tagger_stdin.ChunkTagger()
eventTagger = event_tagger_stdin.EventTagger()
llda = GetLLda()

ner_model = 'ner.model'
ner = GetNer(ner_model)
fe = Features.FeatureExtractor('%s/data/dictionaries' % (BASE_DIR))


capClassifier = cap_classifier.CapClassifier()

vocab = Vocab('%s/hbc/data/vocab' % (BASE_DIR))

dictMap = {}
i = 1
for line in open('%s/hbc/data/dictionaries' % (BASE_DIR)):
	dictionary = line.rstrip('\n')
	dictMap[i] = dictionary
	i += 1

dict2index = {}
for i in dictMap.keys():
	dict2index[dictMap[i]] = i

if llda:
	dictionaries = Dictionaries('%s/data/LabeledLDA_dictionaries3' % (BASE_DIR), dict2index)
entityMap = {}
i = 0
if llda:
	for line in open('%s/hbc/data/entities' % (BASE_DIR)):
		entity = line.rstrip('\n')
		entityMap[entity] = i
		i += 1

dict2label = {}
for line in open('%s/hbc/data/dict-label3' % (BASE_DIR)):
	(dictionary, label) = line.rstrip('\n').split(' ')
	dict2label[dictionary] = label


ACCESS_TOKEN = 'YOUR-ACCESS-TOKEN'
ACCESS_SECRET = 'YOUR-ACCESS-SECRET'
CONSUMER_KEY = 'YOUR-CONSUMER-KEY'
CONSUMER_SECRET = 'YOUR-CONSUMER-SECRET'

#oauth = OAuth(ACCESS_TOKEN, ACCESS_SECRET, CONSUMER_KEY, CONSUMER_SECRET)

#twitter_stream = TwitterStream(auth=oauth)

#iterator = twitter_stream.statuses.filter(track="Trump", language="en")

infile = open(sys.argv[1])

count = 0
event = ""
contenders = []
for line in infile.readlines():
	if count == 0:
		count += 1
		event = line.strip()
	else:
		contenders.append(line.strip())

scores = {}
for item in contenders:
	scores[item] = 0

computedScores = {}
for item in contenders:
	computedScores[item] = ((scores[item] + 1)*1.0)/len(contenders)

formedQueries = []
for item in contenders:
	query = event + " " + item + " win"
	formedQueries.append(query)

#print formedQueries

completeTrack = ""
for item in formedQueries:
	completeTrack += item + ", "
completeTrack.strip()


print completeTrack, formedQueries
#iterator = twitter_stream.statuses.filter(track=completeTrack, language="en")
xmat = np.load('../data/xdata.npy')
yfile = open('../data/ydata.txt', 'r')
lines = yfile.readlines()

hash = {
	'DY': 3,
	'PY': 3,
	'UC': 2,
	'PN': 1,
	'DN': 1,
}
yvec = []
for item in lines:
	yvec.append(hash[item.strip()])
yvec = np.array(yvec)
print yvec
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(xmat, yvec, test_size=0.3, random_state=42)
lrmulti = LRmulticlass()
lrmulti.Train(X_train, y_train)
#pickle.dump(lrmulti.model, 't.save')
y_pred = lrmulti.model.predict(X_test)
from sklearn.metrics import *
print classification_report(y_test, y_pred)
