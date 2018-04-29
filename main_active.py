import argparse
import time
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import pickle as pc
import data
import model
import copy
import random
import os



parser = argparse.ArgumentParser(description='PyTorch PCE Model Active Learning')


parser.add_argument('--data', type=str, default='data/',
                    help='location of the data corpus')


parser.add_argument('--test', type=str, default= "test_data" ,
                    help='what attribute to test ')

parser.add_argument('--train', type=str, default= "train_data" ,
                    help='what attribute to test ')



parser.add_argument('--num',type=int, default=200, help='number of training examples')


parser.add_argument('--embtype', type=str, default="word2vec",
                    help='embedding size')

parser.add_argument('--emb', type=int, default=300,
                    help='embedding size')
parser.add_argument('--epochs', type=int, default=20,
                    help='upper epoch limit')
parser.add_argument('--decay', type=float, default=0,
                    help='weight_decay')

args = parser.parse_args()


###set random seed


seed = 1112

torch.manual_seed(seed)

torch.cuda.manual_seed(seed)

random.seed(seed)




embedding = torch.FloatTensor(np.load(args.data + args.embtype+ "/" + args.embtype+'.6B.' + str(args.emb) + 'd'+'-weights-norm'+'.refined.npy'))

model1 = model.PCE_four_way(args.emb, embedding)

model1.cuda()

corpus = data.Active_Data(args.data ,args.embtype,args.test, args.train)


train_ids = corpus.train

test_ids = corpus.test

retest_ids = corpus.retest

test_label = Variable(test_ids[:,5]).cuda()
test_input1 = Variable(test_ids[:,(0,1,2,3,4)],volatile=True).cuda()
test_input2 = Variable(retest_ids[:,(0,1,2,3,4)],volatile=True).cuda()



oppo_relation = {"tall":"short", "expensive": "cheap", "dense" : "light", "mobile":"immobile", "heroic": "villainous", "dangerous":"safe", "round":"squarish", "liberal":"conservative", "shapeless":"shaped", "compressible": "incompressible", "dry" : "wet", "long":"brief", "delicious":"tasteless", "fury" : "furless", "loud" : "quiet", "sharp":"dull", "bright":"dark", "viscous":"watery", "social" : "solitary", "intelligent":"stupid", "hot":"cold", "rough":"smooth","aerodynamic":"clumsy", "healthy":"unhealthy", "thick":"thin", "northern":"southern","western":"eastern","big":"small","heavy":"light","strong":"breakable","rigid":"flexible","fast":"slow" }

relations = list(oppo_relation.keys())



criterion = nn.CrossEntropyLoss()


def prob_zero(input1):
    # Turn on evaluation mode which disables dropout.
    model1.eval()
    input1 = Variable(input1[:,(0,1,2,3,4)],volatile=True).cuda()


    output = model1(input1)

    output = torch.nn.functional.softmax(output)

    prob ,prediction = torch.max(output ,1)
    prob =float( prob.data.cpu().numpy())
    prob = 1 - prob
    


    return prob


def evaluate_zero(input1, input2,label):
    # Turn on evaluation mode which disables dropout.
    model1.eval()


    output2 = model1(input2)
    output1 = model1(input1)

    output = output1.data + output2.data[:,(2,1,0,3)]

    _ ,prediction = torch.max(output ,1)
    
    label = label.data

    count = 0 

    for i in range(len(label)):
     if label[i] == prediction[i]:
          count += 1


    return count / len(label)



def train_zero(train_ids,epochs=args.epochs):
    input_data = Variable(train_ids[:,(0,1,2,3,4)]).cuda()
    label = Variable(train_ids[:,5]).cuda()
    model1.train()
    parameters = filter(lambda p: p.requires_grad, model1.parameters())
    optimizer = torch.optim.Adam(parameters,weight_decay=args.decay)
    for i in range(epochs):
        output = model1(input_data)
        loss = criterion(output, label) 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def get_score_zero(corpus):
    pool1, pool2 = corpus.get_pool_ids()

     
    model1.eval()

    input1 = Variable(pool1[:,(0,1,2,3,4)],volatile=True).cuda()
    input2 = Variable(pool2[:,(0,1,2,3,4)],volatile=True).cuda()


    output1 = model1(input1)
    output2 = model1(input2)
    output = (output1.data + output2.data[:,(2,1,0,3)])/2
    output = torch.nn.functional.softmax(Variable(output))
    prob = output
    prob ,prediction = torch.max(output ,1)
    prob = prob.data.cpu().numpy()
    prob = 1 - prob
    #prob = - prob[:,2] + prob[:,1]
    #prob = -np.sum( prob * np.log(prob), axis =1)
    prob = prob.tolist()
    corpus.update_score(prob)


def get_EMC(corpus):
    pool1, pool2 = corpus.get_pool_ids()

     
    model1.eval()

    input1 = Variable(pool1[:,(0,1,2,3,4)]).cuda()

    label1 = Variable(torch.LongTensor([0])).cuda()
    label2 = Variable(torch.LongTensor([1])).cuda()
    label3 = Variable(torch.LongTensor([2])).cuda()
    label4 = Variable(torch.LongTensor([3])).cuda()

    label_list = [label1, label2, label3, label4]

    parameters = filter(lambda p: p.requires_grad, model1.parameters())
    optimizer = torch.optim.Adam(parameters,weight_decay=args.decay)

    num_obs = input1.size()[0]
    output = model1(input1)
    prob = torch.nn.functional.softmax(output)
    prob = prob.data.cpu().numpy()
    #prob = prob.data()

    emc = np.zeros(num_obs)

    count = 0
    ps = [p for p in model1.parameters() if p.requires_grad]
    total_grad = 0
    e_grad = 0
    #print(num_obs)

    for i in range(num_obs):
        #print(i)
        example = input1[i,:].view(1,-1)
        e_grad = 0

        

        #prob = prob.data.cpu().numpy()
        for k in range(4):
            
            output = model1(example)
            loss = criterion(output, label_list[k])
            optimizer.zero_grad()
            total_grad = 0

            loss.backward()
            for p in ps:
         
                    #print(p.grad)
                total_grad += p.grad.data.norm()**2
            #temp = total_grad.sqrt()
            #print(temp)
            e_grad += prob[i,k] * ( total_grad ** 0.5 )
        emc[i] = e_grad

    corpus.update_score(emc.tolist())








def train_active_zero(corpus, num ,round, random = True):


    test_accuracy = []
    uc = []

    corpus.random_get(200)

    train_zero(corpus.train,epochs=800)
    print("test accuracy:")
    test_res = evaluate_zero(test_input1, test_input2, test_label)
    test_accuracy.append(test_res)
    print(test_res)

    if random:
        for i in range(num):
            print("round:"+str(round))
            print(i)

            new = corpus.random_get(1)


            uc.append(prob_zero(new))
            train_zero(corpus.train)
            print("test accuracy:")
            test_res = evaluate_zero(test_input1, test_input2, test_label)
            test_accuracy.append(test_res)
            

            
  
            print(test_res)          

    else:
        for i in range(num):
            print("round:"+str(round))
            print(i)
            get_score_zero(corpus)
            new,prob = corpus.active_get(1)
            uc.append(prob_zero(new))

            train_zero(corpus.train)
            print("test accuracy:")
            test_res = evaluate_zero(test_input1, test_input2, test_label)
            test_accuracy.append(test_res)


            print(test_res)                
    return test_accuracy, uc


random_res = []
active_res = []
random_uc = []
active_uc = []


for i in range(20):

    res, uc = train_active_zero(corpus, args.num,i,random=False)
    active_res.append(res)
    active_uc.append(uc)
    model1.reset()
    corpus.set_default()
    res, uc = train_active_zero(corpus, args.num,i)
    random_res.append(res)
    random_uc.append(uc)
    model1.reset()
    corpus.set_default()









