import torch
from torch.autograd import Variable






class PCE_onePole(torch.nn.Module):
	def __init__(self, nhid, embeddings):
		super(PCE_onePole, self).__init__()
		self.nhid = nhid
		self.activation = torch.nn.ReLU()
		self.drop = torch.nn.Dropout()
		embedding_temp = torch.nn.Embedding(embeddings.size(0), embeddings.size(1))
		embedding_temp.weight = torch.nn.Parameter(embeddings)
		embedding_temp.weight.requires_grad = False
		self.embedding = embedding_temp
		self.encoder2 = torch.nn.Linear(nhid*2,nhid)
		self.encoderp = torch.nn.Linear(nhid,nhid)
		self.encodern = torch.nn.Linear(nhid,nhid)
		self.encoders = torch.nn.Linear(nhid,nhid)

	def reset(self):
		
		def init_weights(m):
			if type(m) == torch.nn.Linear:
				m.reset_parameters()
		self.apply(init_weights)

	def forward(self,input_data):

		input_data = self.embedding(input_data)
		x = input_data.select(1,0)
		y = input_data.select(1,1)
		r1 = input_data.select(1,2)
		r2 = input_data.select(1,3)
		r3 = input_data.select(1,4)
		input_embedding = torch.cat([x,y],1)
		h = self.encoder2(input_embedding)
		h = self.drop(h)
		r2 = self.encoders(r1)
		r2 = self.drop(r2)
		r3 = self.encodern(r1)
		r3 = self.drop(r3)
		r1 = self.encoderp(r1)
		r1 = self.drop(r1)
		h = torch.unsqueeze(h,1)
		r1 = torch.unsqueeze(r1,2)
		r2 = torch.unsqueeze(r2,2)
		r3 = torch.unsqueeze(r3,2)
		out1 = torch.bmm(h,r1).squeeze()
		out2 = torch.bmm(h,r2).squeeze()
		out3 = torch.bmm(h,r3).squeeze()
		out = torch.stack([out1,out2,out3],1)

		return out







class PCE_four_way(torch.nn.Module):
	def __init__(self, nhid,embeddings):
		super(PCE_four_way, self).__init__()

		self.nhid = nhid
		self.activation = torch.nn.ReLU()
		self.drop = torch.nn.Dropout()
		embedding_temp = torch.nn.Embedding(embeddings.size(0), embeddings.size(1))
		embedding_temp.weight = torch.nn.Parameter(embeddings)
		embedding_temp.weight.requires_grad = False
		self.embedding = embedding_temp	

		self.encoder1 = torch.nn.Linear(nhid,nhid)
		self.encoder2 = torch.nn.Linear(nhid*2,nhid)


	def reset(self):
		
		def init_weights(m):
			if type(m) == torch.nn.Linear:
				m.reset_parameters()
		self.apply(init_weights)


	def forward(self,input_data):

		input_data = self.embedding(input_data)
		x = input_data.select(1,0)
		y = input_data.select(1,1)
		r1 = input_data.select(1,2)
		r2 = input_data.select(1,3)
		r3 = input_data.select(1,4)		
		input_embedding = torch.cat([x,y],1)
		h1 = self.encoder1(x)
		h2 = self.encoder1(y)
		h1 = self.drop(h1)
		h2 = self.drop(h2)
		h = self.encoder2(input_embedding)
		h = self.drop(h)
		h = torch.unsqueeze(h,1)
		h1 = torch.unsqueeze(h1,1)
		h2 = torch.unsqueeze(h2,1)
		r1 = torch.unsqueeze(r1,2)
		r2 = torch.unsqueeze(r2,2)
		r3 = torch.unsqueeze(r3,2)
		out1 = torch.bmm(h,r1).squeeze()
		out2 = torch.bmm(h,r2).squeeze()
		out3 = torch.bmm(h,r3).squeeze()
		out4 = (torch.bmm(h1,r1).squeeze() + torch.bmm(h1,r3).squeeze()  + torch.bmm(h2,r1).squeeze() + torch.bmm(h2,r3).squeeze())
		out = torch.stack([out1,out2,out3,out4],1)


		return out



class PCE_three_way(torch.nn.Module):
	def __init__(self, nhid, embeddings):
		super(PCE_three_way, self).__init__()
		self.nhid = nhid
		self.activation = torch.nn.ReLU()
		self.drop = torch.nn.Dropout()
		embedding_temp = torch.nn.Embedding(embeddings.size(0), embeddings.size(1))
		embedding_temp.weight = torch.nn.Parameter(embeddings)
		embedding_temp.weight.requires_grad = False
		self.embedding = embedding_temp
		self.encoder2 = torch.nn.Linear(nhid*2,nhid)


	def reset(self):
		
		def init_weights(m):
			if type(m) == torch.nn.Linear:
				m.reset_parameters()
		self.apply(init_weights)


	def forward(self,input_data):

		input_data = self.embedding(input_data)
		x = input_data.select(1,0)
		y = input_data.select(1,1)
		r1 = input_data.select(1,2)
		r2 = input_data.select(1,3)
		r3 = input_data.select(1,4)
		input_embedding = torch.cat([x,y],1)
		h = self.encoder2(input_embedding)
		h = self.drop(h)
		h = torch.unsqueeze(h,1)
		r1 = torch.unsqueeze(r1,2)
		r2 = torch.unsqueeze(r2,2)
		r3 = torch.unsqueeze(r3,2)
		out1 = torch.bmm(h,r1).squeeze()
		out2 = torch.bmm(h,r2).squeeze()
		out3 = torch.bmm(h,r3).squeeze()
		out = torch.stack([out1,out2,out3],1)

		return out






