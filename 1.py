# how to read in an event information 

# some import you wil need
import json
import glob
import torch
import torch.nn.utils.rnn as rnn_utils
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import os
import copy
from torch.utils.data import DataLoader
import torch.optim as optim



# Output: {'name': 'Bob', 'languages': ['English', 'French']}
main_dir = './EVENTS'
left_events_dir = main_dir + '/' + 'LEFT' # 'RIGHT' or right, 'NONE' for no-lane change
right_events_dir = main_dir + '/' + 'RIGHT' # 'RIGHT' or right, 'NONE' for no-lane change
lk_events_dir = main_dir + '/' + 'NONE' # 'RIGHT' or right, 'NONE' for no-lane change

def get_sorted_bbs_of_all_events(events_dir):
    # get all events file names in specified events directory
    all_events_filenames = sorted(glob.glob(events_dir + '/*.json'))

    bbs_of_all_events = []
    for i in range(len(all_events_filenames)):
        # read just one event
        with open(all_events_filenames[i], 'r') as f:
            event_info = json.load(f) # this will read in the event as a dictornary

        # each event is stored as a dictionary
        #         event_info['drive'] = path # drive path
        #         event_info['id'] = 0 # id taken from lane_changes.txt or 0 if no lane change event
        #         event_info['lc_type'] = 0 # 3 = left, 4 = right, 0 = no-change
        #         event_info['f0'] = f0 # start frame of event which is actualy f0 - 30 for lane changes
        #         event_info['f1'] = f1 # mid point of event
        #         event_info['f2'] = f2 # end of event window
        #         event_info['blinker'] = 0 # 1 = blinker, 0 = no blinker (always 0 for no lane changes)

        #         event_info['unique ids'] = [id0, id1, ..., idM] # list of unique ids, with M unique cars involved in event frames
        #         event_info['bbs by frame'] = vehicle_bbs_by_frame # a dictionary of boxes indexed by frame
        #                vehicle_bbs_by_frame[frame][id] = (xi, yi, xf, yf) # this allows you to find a box on a given frame for a given vehicle
        # 
        #print(event_info) # show what we read in

        # how to get tracked bbs per frame for a given event so that they are in the id order

        # 1. get the unique vehicle ids for the event
        unique_ids = event_info['unique ids']
        unique_ids.sort()
        #print(unique_ids)

        # 2. get the range of frames for the event
        f0 = event_info['f0'] # this is actually f0-N where f0 is taken from the lane change
        f2 = event_info['f2'] # can go to f1 if you want to do it up only to the event

        # 3. bounding box information, indexed by frame and vehicle id: this is the main information in the event json
        vehicle_bbs_by_frame = event_info['bbs by frame']
        #print(list(vehicle_bbs_by_frame.keys()))

        #print(f0, f2)
        # 4. for each frame

        bbs_of_this_event = []

        for f in range(f0, f2): # range of frames we want tracked boxes in order vehicle

            #print('frame {:d}'.format(f))

            assert(str(f) in vehicle_bbs_by_frame) # should always be OK! note keys will be strings on read

            # now iterate over the ids we have (which is a list)
            bbs_of_this_frame = []
            for id in unique_ids:

                #print(list(vehicle_bbs_by_frame[str(f)].keys()))
                # note the use of str(f) and str(id) to access the frame and ids in the 
                # event_info read back using json
                if (str(id) in vehicle_bbs_by_frame[str(f)]): # not all vehicles are on all frames!
                    bbox = vehicle_bbs_by_frame[str(f)][str(id)]
                    #print('id ={:d}'.format(id), ':', bbox) # this will be 4-tuple (xi, yi, xf, yf)
                else:
                    bbox = ()
                    #bbox = (0, 0, 0, 0) # no detection 
                    #print('id = {:d}'.format(id), ':', bbox) # this will be 4-tuple (xi, yi, xf, yf)
                bbs_of_this_frame.extend(bbox)
            bbs_of_this_event.extend(bbs_of_this_frame)
#             box = (0, 0, 0, 0)
#             bbs_of_this_event.extend(box)

        bbs_of_all_events.append(bbs_of_this_event)
    return bbs_of_all_events
    
    
bbs_of_left_events = get_sorted_bbs_of_all_events(left_events_dir)

bbs_of_right_events = get_sorted_bbs_of_all_events(right_events_dir)

bbs_of_lk_events = get_sorted_bbs_of_all_events(lk_events_dir)

bbs_of_lk_events = bbs_of_lk_events[0:340]


left_data = [] 
for i in bbs_of_left_events:
    temp = torch.FloatTensor(i)
    left_data.append(temp)

left_data_len = len(left_data)
print('left_data:', len(left_data))

right_data = [] 
for j in bbs_of_right_events:
    temp = torch.FloatTensor(j)
    right_data.append(temp)
    
right_data_len = len(right_data)
print('right_data:', len(right_data))


non_data = []
for k in bbs_of_lk_events:
    temp = torch.FloatTensor(k)
    non_data.append(temp)
    
non_data_len = len(non_data)
print('non_data:', len(non_data))



left_data.extend(right_data)
left_data.extend(non_data)

all_data = left_data
print('all_data:', len(all_data))

data_length = [len(sq) for sq in all_data]
data = rnn_utils.pad_sequence(all_data, batch_first=True, padding_value=0.0)     # 用零补充，使长度对齐

data = data.cuda()

#把拆分成帧的 数据 改成 60帧一个样本
left_frame_tensor = data[0:left_data_len]
right_frame_tensor = data[left_data_len:left_data_len+right_data_len]
non_frame_tensor = data[left_data_len+right_data_len:]


#len(left_tensor), len(right_tensor)
left_tensor = []
right_tensor = []
non_tensor = []
left_lable = torch.empty(1,dtype=torch.int64).cuda()
left_lable[0] = 1
right_lable = torch.empty(1,dtype=torch.int64).cuda()
right_lable[0] = 2
lk_lable = torch.empty(1,dtype=torch.int64).cuda()
lk_lable[0] = 0


for i in range(left_data_len):
    #temp = left_frame_tensor[(i)*60:(i+1)*60]
    temp = left_frame_tensor[i]
    temp = [temp,left_lable]
    left_tensor.append(temp)

for i in range(right_data_len):
    #temp = right_frame_tensor[(i)*60:(i+1)*60]
    temp = right_frame_tensor[i]
    temp = [temp,right_lable]
    right_tensor.append(temp)
    
for i in range(non_data_len):
    #temp = non_frame_tensor[(i)*60:(i+1)*60]
    temp = non_frame_tensor[i]
    temp = [temp,lk_lable]
    non_tensor.append(temp)

size = len(left_tensor)
training_set = left_tensor[:int(size*0.7)]
testing_set = left_tensor[int(size*0.7):]

size = len(right_tensor)
training_set.extend(right_tensor[:int(size*0.7)])
testing_set.extend(right_tensor[int(size*0.7):])

size = len(non_tensor)
training_set.extend(non_tensor[:int(size*0.7)])
testing_set.extend(non_tensor[int(size*0.7):])


class LSTMC(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, label_size, batch_size, use_gpu):
        super(LSTMC, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.use_gpu = use_gpu

        #self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2label = nn.Linear(hidden_dim, label_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        if self.use_gpu:
            h0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim).cuda())
            c0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim).cuda())
        else:
            h0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim))
            c0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim))
        return (h0, c0)

    def forward(self, sentence):
        #embeds = self.word_embeddings(sentence)
        embeds = sentence

        x = embeds.view(len(sentence), self.batch_size, -1)
        #print(x)
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        y  = self.hidden2label(lstm_out[-1])
        
        return y



use_plot = True
use_save = True
if use_save:
    import pickle
    from datetime import datetime

## parameter setting
epochs = 100
batch_size = 1
use_gpu = torch.cuda.is_available()
learning_rate = 0.01
nlabel = 3
hidden_dim = 100
embedding_dim = 1


def adjust_learning_rate(optimizer, epoch):
    lr = learning_rate * (0.1 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer



### create model
model = LSTMC(embedding_dim=embedding_dim,hidden_dim=hidden_dim,
                       label_size=nlabel, batch_size=batch_size, use_gpu=use_gpu)
if use_gpu:
    model = model.cuda()

dtrain_set = training_set
dtest_set = testing_set

####----------------------------------------------------------------------------------------

#optimizer = optim.SGD(model.parameters(), lr=0.001)

#loss_function = nn.MSELoss() 

loss_function = nn.NLLLoss() #loss(x,label)=−xlabel

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#optimizer = optim.SGD(model.parameters(), lr=0.001)


####----------------------------------------------------------------------------------------


train_loss_ = []
test_loss_ = []
train_acc_ = []
test_acc_ = []
### training procedure
for epoch in range(epochs):
    optimizer = adjust_learning_rate(optimizer, epoch)

    ## training epoch
    total_acc = 0.0
    total_loss = 0.0
    total = 0.0
    model.train()
    
    print('epoch: ', epoch)
        
    for train_inputs, train_labels in (dtrain_set):

        model.zero_grad()
        model.batch_size = 1
        model.hidden = model.init_hidden()
####----------------------------------------------------------------------------------------

        
        output = model(train_inputs.t())
        
        #_, predicted = torch.max(output.data, 1)
        #loss = loss_function(predicted, Variable(train_labels))
        
        loss = loss_function(output, train_labels)

####----------------------------------------------------------------------------------------
        loss.backward()
        optimizer.step()

        # calc training acc
        _, predicted = torch.max(output.data, 1)
        total_acc += (predicted == train_labels).sum()
        total += len(train_labels)
        total_loss += loss.data
        
        #print(train_labels.cpu(), predicted.cpu(), total_acc/total)


    train_loss_.append(total_loss / total)
    train_acc_.append(total_acc / total)


    ## testing epoch
    total_acc = 0.0
    total_loss = 0.0
    total = 0.0
    
    model.eval()
    with torch.no_grad():

        for test_inputs, test_labels in (dtest_set):
            #print(test_inputs)
            model.batch_size = 1
            #model.hidden = model.init_hidden()

####----------------------------------------------------------------------------------------

            output = model(test_inputs.t())
            #print("%.10f,%.10f,%.10f,"%(output[0][0],output[0][1],output[0][2]))
            
            #_, predicted = torch.max(output.data, 1)
            #loss = loss_function(predicted, Variable(train_labels))
            
            loss = loss_function(output, Variable(test_labels))
####----------------------------------------------------------------------------------------


            # calc testing acc
            _, predicted = torch.max(output.data, 1)
            total_acc += (predicted == test_labels).sum()
            total += len(test_labels)
            total_loss += loss.data

            #print(test_labels.cpu(), predicted.cpu(), total_acc.cpu()/total)

        test_loss_.append(total_loss / total)
        test_acc_.append(total_acc / total)

        print('[Epoch: %3d/%3d] Training Loss: %.3f, Testing Loss: %.3f, Training Acc: %.3f, Testing Acc: %.3f'
              % (epoch, epochs, train_loss_[epoch], test_loss_[epoch], train_acc_[epoch], test_acc_[epoch]))

param = {}
param['lr'] = learning_rate
param['batch size'] = batch_size
#param['embedding dim'] = embedding_dim
param['hidden dim'] = hidden_dim
#param['sentence len'] = sentence_len

result = {}
result['train loss'] = train_loss_
result['test loss'] = test_loss_
result['train acc'] = train_acc_
result['test acc'] = test_acc_
result['param'] = param

if use_plot:
    import PlotFigure as PF
    PF.PlotFigure(result, use_save)
if use_save:
    filename = 'log/LSTM_classifier_' + datetime.now().strftime("%d-%h-%m-%s") + '.pkl'
    result['filename'] = filename

    fp = open(filename, 'wb')
    pickle.dump(result, fp)
    fp.close()
    print('File %s is saved.' % filename)
