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



parser = argparse.ArgumentParser(description='PyTorch PCE Model')

parser.add_argument('--data', type=str, default='data/',
                    help='location of the data corpus')

parser.add_argument('--test', type=str, default= "verb_physics_test_5" ,
                    help='Name of test data ')
parser.add_argument('--train', type=str, default= "verb_physics_train_5" ,
                    help='Name of train data')

parser.add_argument('--dev', type=str, default= "verb_physics_dev_5" ,
                    help='Name of dev data')



parser.add_argument('--embtype', type=str, default="word2vec",
                    help='embedding size')

parser.add_argument('--emb', type=int, default=300,
                    help='embedding size')

parser.add_argument('--epochs', type=int, default=800,
                    help='upper epoch limit')

parser.add_argument('--decay', type=float, default=0,
                    help='weight_decay')


parser.add_argument('--relation', type=str, default="all",
                    help='test which relation(property), big for size, heavy for weight, strong for strength, rigid for rigidness, fast for speed' )

parser.add_argument('--train_all', type=str, default="yes",
                    help='whether train all the properties, yes or no')

parser.add_argument('--zero', type=str , default= "no",
                    help='whether perform zero shot learning to the given relation(property),yes or no')



parser.add_argument('--ensemble', type=str, default="yes",
                    help='baseline')

args = parser.parse_args()


###set random seed


seed = 1112

torch.manual_seed(seed)

torch.cuda.manual_seed(seed)

random.seed(seed)










embedding = torch.FloatTensor(np.load(args.data + args.embtype+ "/" + args.embtype+'.6B.' + str(args.emb) + 'd'+ '-weights-norm' + '.refined.npy'))


model = model.PCE_three_way(args.emb, embedding)
model.cuda()

corpus = data.Data(args.data ,args.embtype,args.train,args.dev,args.test)


if args.relation == "all":
    train_ids = corpus.train
    dev_ids = corpus.dev
    redev_ids = corpus.redev
    test_ids = corpus.test
    retest_ids = corpus.retest

else:
    train_ids, dev_ids, redev_ids, test_ids, retest_ids = corpus.get_relation(args.relation)
    if args.train_all == "yes":
        train_ids = corpus.train
    if args.zero == "yes":
        train_ids, dev_ids, redev_ids, test_ids, retest_ids = corpus.get_zero(args.relation)



oppo_relation = {"tall":"short", "expensive": "cheap", "dense" : "light", "mobile":"immobile", "heroic": "villainous", "dangerous":"safe", "round":"squarish", "liberal":"conservative", "shapeless":"shaped", "compressible": "incompressible", "dry" : "wet", "long":"brief", "delicious":"tasteless", "fury" : "furless", "loud" : "quiet", "sharp":"dull", "bright":"dark", "viscous":"watery", "social" : "solitary", "intelligent":"stupid", "hot":"cold", "rough":"smooth","aerodynamic":"clumsy", "healthy":"unhealthy", "thick":"thin", "northern":"southern","western":"eastern","big":"small","heavy":"light","strong":"breakable","rigid":"flexible","fast":"slow" }

relations = list(oppo_relation.keys())


criterion = nn.CrossEntropyLoss()




test_label = Variable(test_ids[:,5]).cuda()
test_input1 = Variable(test_ids[:,(0,1,2,3,4)],volatile=True).cuda()
test_input2 = Variable(retest_ids[:,(0,1,2,3,4)],volatile=True).cuda()


dev_label = Variable(dev_ids[:,5]).cuda()
dev_input1 = Variable(dev_ids[:,(0,1,2,3,4)],volatile=True).cuda()
dev_input2 = Variable(redev_ids[:,(0,1,2,3,4)],volatile=True).cuda()


def evaluate_zero(input1, input2, label):

    model.eval()

    output2 = model(input2)
    output1 = model(input1)
    output = output1.data + output2.data[:,(2,1,0)]

    _ ,prediction = torch.max(output ,1)
    
    label = label.data

    count = 0 
    count0 = 0
    count1 = 0
    count2 = 0
    for i in range(len(label)):
     if label[i] == prediction[i]:
          count += 1




    return count / len(label)


def evaluate(input1, input2, label):

    model.eval()


    output = model(input1)

    _ ,prediction = torch.max(output.data ,1)
    
    label = label.data

    count = 0 
    count0 = 0
    count1 = 0
    count2 = 0
    for i in range(len(label)):
     if label[i] == prediction[i]:
          count += 1




    return count / len(label)


def train_zero():


     input_data = Variable(train_ids[:,(0,1,2,3,4)]).cuda()

     label = Variable(train_ids[:,5]).cuda()


     parameters = filter(lambda p: p.requires_grad, model.parameters())

     optimizer = torch.optim.Adam(parameters,weight_decay=args.decay)
     for i in range(args.epochs):

          model.train()
          output = model(input_data)
          #print(output)
          loss = criterion(output, label) 
          #print(i,loss.data[0])

          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
          if i % 200 ==0:
               print("train loss:" + str(loss.data[0]))


if args.ensemble == "yes":
    print(evaluate_zero(test_input1,test_input2,test_label))
    train_zero()

    print("dev accuracy:")

    print(evaluate_zero(dev_input1,dev_input2,dev_label))

    print("test accuracy:")
    
    print(evaluate_zero(test_input1,test_input2,test_label))
else:
    print(evaluate(test_input1,test_input2,test_label))
    train_zero()

    print("dev accuracy:")

    print(evaluate(dev_input1,dev_input2,dev_label))

    print("test accuracy:")
    
    print(evaluate(test_input1,test_input2,test_label))











