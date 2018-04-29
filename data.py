import torch
import numpy as np
import pickle as pc
import random
import copy



class Dictionary(object):
    def __init__(self, path,embtype):
        self.word2idx = pc.load(open(path + embtype + "/" + embtype + '.6B.vocab.refined.pickle','rb'))
        self.idx2word = {}
        

        for it in self.word2idx:
        	self.idx2word[self.word2idx[it]] = it







class Data(object):
	def __init__(self, path,embtype,train_data, dev_data, test_data):
		self.dictionary = Dictionary(path,embtype)
		self.train_temp = pc.load(open(path +train_data+'.pickle','rb'))
		self.dev_temp = pc.load(open(path +dev_data+'.pickle','rb'))
		self.test_temp = pc.load(open(path +test_data+'.pickle','rb'))

		self.train = self.tokenize(self.train_temp)
		self.dev = self.tokenize(self.dev_temp)
		self.redev = self.tokenize(self.dev_temp,True)

		#self.dev = self.tokenize(pc.load(open(path + 'mydata/dev_big_5','rb')))
		
		#temp = random.sample(temp)
		self.test = self.tokenize(self.test_temp)
		self.retest = self.tokenize(self.test_temp,True)

	def get_relation(self,relation):
		train_temp = [it for it in self.train_temp if it[2] == relation]
		dev_temp = [it for it in self.dev_temp if it[2] == relation]
		test_temp = [it for it in self.test_temp if it[2] == relation]

		return self.tokenize(train_temp), self.tokenize(dev_temp), self.tokenize(dev_temp,True), self.tokenize(test_temp), self.tokenize(test_temp, True)

	def get_zero(self,relation):
		train_temp = [it for it in self.train_temp if it[2] != relation]
		dev_temp = [it for it in self.dev_temp if it[2] == relation]
		test_temp = [it for it in self.test_temp if it[2] == relation]

		return self.tokenize(train_temp), self.tokenize(dev_temp), self.tokenize(dev_temp,True), self.tokenize(test_temp), self.tokenize(test_temp, True)

	def tokenize(self, data_list, reverse=False):
		label = {}
		label[-1] = 0
		label[0] = 1
		label[1] = 2
		label[-42] = 3		
		data_list = [it for it in data_list if it[3] in label and it[0] in self.dictionary.word2idx and it[1] in self.dictionary.word2idx and it[2] in self.dictionary.word2idx ]
		ids = torch.LongTensor(len(data_list),6).zero_()
		count = 0
		oppo_relation = {"tall":"short", "expensive": "cheap", "dense" : "light", "mobile":"immobile", "heroic": "villainous", "dangerous":"safe", "round":"squarish", "liberal":"conservative", "shapeless":"shaped", "compressible": "incompressible", "dry" : "wet", "long":"brief", "delicious":"tasteless", "fury" : "furless", "loud" : "quiet", "sharp":"dull", "bright":"dark", "viscous":"watery", "social" : "solitary", "intelligent":"stupid", "hot":"cold", "rough":"smooth","aerodynamic":"clumsy", "healthy":"unhealthy", "thick":"thin", "northern":"southern","western":"eastern","big":"small","heavy":"light","strong":"breakable","rigid":"flexible","fast":"slow" }

		if reverse:
			for it in data_list:
			
				ids[count, 0 ] = self.dictionary.word2idx[it[0]]
				ids[count, 1 ] = self.dictionary.word2idx[it[1]]
				ids[count, 2 ] = self.dictionary.word2idx[oppo_relation[it[2]]]
				ids[count, 3 ] = self.dictionary.word2idx["similar"]
				ids[count, 4 ] = self.dictionary.word2idx[it[2]]			
				ids[count, 5] = label[it[3]]
				count = count + 1
		else:
			for it in data_list:
			
				ids[count, 1 ] = self.dictionary.word2idx[it[0]]
				ids[count, 0 ] = self.dictionary.word2idx[it[1]]
				ids[count, 2 ] = self.dictionary.word2idx[oppo_relation[it[2]]]
				ids[count, 3 ] = self.dictionary.word2idx["similar"]
				ids[count, 4 ] = self.dictionary.word2idx[it[2]]			
				ids[count, 5] = label[it[3]]
				count = count + 1

		return ids


class Active_Data(object):
	def __init__(self, path ,embtype, test_data, train_data):

		self.train_data = train_data
		self.path = path
		self.dictionary = Dictionary(path,embtype)
		self.train_pool = pc.load(open(path +train_data+'.pickle','rb'))		
		
		self.train_list = []
		#self.random_get(50)
		self.train = self.tokenize(self.train_list)

		self.score = []

		#self.train_pool_copy = copy.deepcopy(self.train_pool)
		#self.train_list_copy = copy.deepcopy(self.train_list)
		self.test = self.tokenize(pc.load(open(path +test_data+'.pickle','rb')))
		self.retest = self.tokenize(pc.load(open(path +test_data+'.pickle','rb')), True)



	def get_pool_ids(self):
		return self.tokenize(self.train_pool), self.tokenize(self.train_pool,True)


	def tokenize(self, data_list, reverse=False):
		label = {}
		label[-1] = 0
		label[0] = 1
		label[1] = 2
		label[-42] = 3		
		data_list = [it for it in data_list if it[3] in label and it[0] in self.dictionary.word2idx and it[1] in self.dictionary.word2idx and it[2] in self.dictionary.word2idx ]
		ids = torch.LongTensor(len(data_list),6).zero_()
		count = 0
		oppo_relation = {"tall":"short", "expensive": "cheap", "dense" : "light", "mobile":"immobile", "heroic": "villainous", "dangerous":"safe", "round":"squarish", "liberal":"conservative", "shapeless":"shaped", "compressible": "incompressible", "dry" : "wet", "long":"brief", "delicious":"tasteless", "fury" : "furless", "loud" : "quiet", "sharp":"dull", "bright":"dark", "viscous":"watery", "social" : "solitary", "intelligent":"stupid", "hot":"cold", "rough":"smooth","aerodynamic":"clumsy", "healthy":"unhealthy", "thick":"thin", "northern":"southern","western":"eastern","big":"small","heavy":"light","strong":"breakable","rigid":"flexible","fast":"slow" }
		#print(len(oppo_relation))
		if not reverse:
			for it in data_list:
			
				ids[count, 0 ] = self.dictionary.word2idx[it[0]]
				ids[count, 1 ] = self.dictionary.word2idx[it[1]]
				ids[count, 2 ] = self.dictionary.word2idx[oppo_relation[it[2]]]
				ids[count, 3 ] = self.dictionary.word2idx["similar"]
				ids[count, 4 ] = self.dictionary.word2idx[it[2]]			
				ids[count, 5] = label[it[3]]
				count = count + 1
		else:
			for it in data_list:
			
				ids[count, 1 ] = self.dictionary.word2idx[it[0]]
				ids[count, 0 ] = self.dictionary.word2idx[it[1]]
				ids[count, 2 ] = self.dictionary.word2idx[oppo_relation[it[2]]]
				ids[count, 3 ] = self.dictionary.word2idx["similar"]
				ids[count, 4 ] = self.dictionary.word2idx[it[2]]			
				ids[count, 5] = label[it[3]]
				count = count + 1

		return ids


	
	def random_get(self, num = 1):
		sampled_examples = random.sample(self.train_pool, num)
		for it in sampled_examples:
			self.train_pool.remove(it)
		self.train_list += sampled_examples
		self.train = self.tokenize(self.train_list)
		return self.tokenize(sampled_examples)

	def active_get(self, num = 1):
		sorted_score = sorted(range(len(self.score)), key=lambda k: self.score[k], reverse=True)
		sampled_examples = []
		for i in range(num):
			sampled_examples.append(self.train_pool[sorted_score[i]])
		for it in sampled_examples:
			self.train_pool.remove(it)
		self.train_list += sampled_examples
		self.train = self.tokenize(self.train_list)
		return self.tokenize(sampled_examples),self.score[sorted_score[0]]

	def update_score(self, new_score):
		self.score = new_score

	def set_default(self):
		#self.train_pool = copy.deepcopy(self.train_pool_copy)
		#self.train_list = copy.deepcopy(self.train_list_copy)
		self.train_pool = pc.load(open(self.path +self.train_data+'.pickle','rb'))	
		self.train_list = []
		self.score = []
		self.train = self.tokenize(self.train_list)





