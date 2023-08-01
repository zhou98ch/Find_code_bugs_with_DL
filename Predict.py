#!/usr/bin/python3
# python Predict.py --model 'ensemble-model-40000sample-loss35.pth' --source "shared_resources/real_test_for_milestone3/real_inconsistent.json" --destination t.json
#final -v
import argparse
import libcst as cst
import json
import pickle
import nltk
from nltk.tokenize import word_tokenize
import libcst as cst

nltk.download('punkt')
#from sklearn import preprocessing
import numpy as np
import random

from gensim.models import FastText
import gensim
import math
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import torch.utils.data as Data
from tokenize import tokenize
from io import BytesIO
import json

parser = argparse.ArgumentParser()
parser.add_argument(
    '--model', help="Path of trained model.", required=True)
parser.add_argument(
    '--source', help="Path of the test function file.", required=True)
parser.add_argument(
    '--destination', help="Path to output JSON file with predictions.", required=True)

# configuration for the embeding--------------------------
embed_dim = 32
test_seq_length = 16
raise_seq_length = 20
# configuration for the model--------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
sequence_length = 32
input_size = 32  # emebedding vector feature
hidden_size = 128  # LSTM hidden_size
num_layers = 3  # LSTM layers
num_classes = 2
batch_size = 32
num_epochs = 70
learning_rate = 0.001
weight_decay = 0.0
test_sequence_length = 16  # test input length to feed into the model
raise_sequence_length = 20  # raise input length to feed into the model


# two sub-bidirectional-LSTM, feed [test] and [raise] into each
class subBLSTM(nn.Module):
    def _init_lstm(self, weight):
        for w in weight.chunk(4, 0):
            nn.init.xavier_uniform_(w)

    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(subBLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, bidirectional=True
        )

        self._init_lstm(self.lstm.weight_ih_l0)
        self._init_lstm(self.lstm.weight_hh_l0)
        self.lstm.bias_ih_l0.data.zero_()
        self.lstm.bias_hh_l0.data.zero_()

    def forward(self, x):
    
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return out


@classmethod
# for updating learning rate
def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

#ensembled model
class MyEnsemble(nn.Module):
    def __init__(self, modelA, modelB, input_len, nb_classes=2):
        super(MyEnsemble, self).__init__()

        # two sub-LSTM model
        self.modelA = modelA
        self.modelB = modelB

        # linear layers
        self.classifier1 = nn.Linear(hidden_size * 4, hidden_size)
        self.dropout1 = nn.Dropout(p = 0.4)
        self.classifier2 = nn.Linear(hidden_size, nb_classes)
        self.dropout2 = nn.Dropout(p = 0.4)

        # initialization
        nn.init.xavier_uniform_(self.classifier1.weight, gain=math.sqrt(2.0))
        nn.init.xavier_uniform_(self.classifier2.weight, gain=math.sqrt(2.0))

    def forward(self, xa, xb):
        # sub models ouputs
        x1 = self.modelA(xa)  # condition/test
        x2 = self.modelB(xb)  # raise

        # concatenation
        x_con = torch.cat((x1, x2), dim=1)

        # linear layers
        x_c1 = self.classifier1(x_con)
        x_c1 = self.dropout1(x_c1)
        x_c2 = self.classifier2(x_c1)
        x_c2 = self.dropout2(x_c2)
        return x_c2

#for finding [raise]
class FindRaise(cst.CSTVisitor):
    raise_flag = False

    def __init__(self):
        self.raise_flag = False
        self.raises = []

    def visit_Raise(self, node: cst.Raise):
        try:
            self.raise_flag = True
            self.raises.append(node)

        except Exception as e:
            print(e)

#for checking if exists [if]
class CheckIf(cst.CSTVisitor):

    def __init__(self):
        self.if_flag = False

    def visit_If(self, node: cst.If):
        try:
            self.if_flag = True

        except Exception as e:
            print(e)


def check_end(body):
    if_checker = CheckIf()
    _ = body.visit(if_checker)
    if (not if_checker.if_flag):  # if this is the end sentence/leaf node. (no child if-pharse any more )
        return True
    else:
        return False


def get_end_raise(body):
    # if raise exist here
    raise_finder = FindRaise()
    _ = body.visit(raise_finder)
    raises = raise_finder.raises
    if (len(raises) != 0):  # has raises
        return raises
    else: # no raise in this body
        return None


class FindIf(cst.CSTVisitor):
    if_stack = []
    tests = []
    else_flags = []
    raises = []

    def __init__(self):
        self.if_stack = []
        self.tests = []
        self.else_flags = []
        self.raises = []
        self.cnt = 0

    # ONLY STORE INFORMATION WITH LEAF BRANCH
    def visit_If(self, node: cst.If):
        try:
            self.if_stack.append(node)

            # check mainbody branch------------------------------
            body = node.body
            if (check_end(body)):  # if this branch is already a leaf node!

                tmp_raise = get_end_raise(body)
                if (tmp_raise != None):
                    self.raises.append(tmp_raise)  # append real code or None
                else:
                    self.raises.append(False)

                self.tests.append(node.test)  # if mainbody is a leaf, then add "NOT" +if-test
                self.else_flags.append(False)

            # check orelse branch------------------------------
            if (node.orelse != None and str(type(node.orelse))!="<class 'libcst._nodes.statement.If'>"):

                elsebody = node.orelse.body
                if (check_end(elsebody)):  # if else branch is already a leaf node! ONLY DO WITH LEAF BRANCH
                    tmp_else_raise = get_end_raise(elsebody)
                    if (tmp_else_raise != None):
                        self.raises.append(tmp_else_raise)  # append real code or None
                    else:
                        self.raises.append(False)
                    self.tests.append(node.test)  # if else is a leaf, then add "NOT if-test"
                    self.else_flags.append(True)
            self.cnt += 1
        except Exception as e:
            print(e)

def vectorize_code(tokens, model):
    vector = []
    for t in tokens:
        vector.append(model.wv[t])
    return vector

def vectorize_trim_pad(sequences, embd, embed_dim, seq_length=48):
    trimed_stmts = []
    for seq in sequences:
        if len(seq) >= seq_length:
            seq_vec = vectorize_code(seq[:seq_length], embd)
        else:
            seq_vec = vectorize_code(seq, embd) + [np.zeros(embed_dim) for _ in range(len(seq), seq_length, 1)]

        trimed_stmts.append(seq_vec)
    return np.array(trimed_stmts)


def tokenize_nlt(code):
    try:
        return word_tokenize(code.replace('`', '').replace("'", ''))
    except Exception as e:
        print(e)


def tokenize_python(code):
    g = tokenize(BytesIO(code.encode('utf-8')).readline)
    try:
        tokens = [c[1] for c in g if c[1] != '' and c[1] != '\n'][1:]
    except:
        tokens = tokenize_nlt(code)
    clean_tokens = []
    for t in tokens:
        if ' ' in t:
            clean_tokens += tokenize_nlt(t.replace('"', '').replace("'", ''))
        else:
            clean_tokens.append(t)
    return clean_tokens

def get_line_num(funcs):
    all_line_num = []
    for func in funcs:
        fun_line_num = []
        lines = func.split("\n")[1:] #the first one is an empty line, dump it
        for i in range(len(lines)):
            if("raise " in lines[i]):
                print("line number:",i+1,"   |raise code:",lines[i])
                fun_line_num.append(i+1)
        all_line_num.append(fun_line_num)
    return all_line_num

def predict(model, test_files):
    print("--------------------")
    print("extracting all the test(condition) codes and raise codes")
    with open(test_files) as tf:
        test_func_list = json.load(tf)

    ##get rraise codes and tcodes
    rcodes = []
    tcodes = []
    for code in test_func_list:
        mdl = cst.parse_module(code)
        if_finder = FindIf()
        _ = mdl.visit(if_finder)
        all_if = if_finder.if_stack
        all_test = if_finder.tests
        elses = if_finder.else_flags
        raises = if_finder.raises
        t = if_finder.cnt

        for t, e, r in zip(all_test, elses, raises):
            if (r != False):

                raise_code = cst.Module(r).code
                test_code = cst.Module([t]).code
                if (e):  # else
                    tcodes.append("not " + test_code)
                    rcodes.append(raise_code)

                else:  # not else
                    tcodes.append(test_code)
                    rcodes.append(raise_code)
    print("extracted phrases:")
    for a, b in zip(tcodes, rcodes):
        print(a, ",", b)
    print("--------------------")
    # process tcodes, rcodes
    print("loading FastText model...")
    ##try loading the tokenizer
    #PATH = 'shared_resources/pretrained_fasttext/embed_if_32.mdl'
    PATH = 'embed_if_32.mdl'
    embed_model = gensim.models.FastText.load(PATH)
    print("FastText model loaded")

    # get test vectors---------------------------------------

    test_vectors = np.empty((0, test_seq_length, 32))
    print("---------------------------------")
    print("creating word embedding vectors for tests code,sequence length=", test_seq_length, ",embed dimension=",
          embed_dim)
    for i in range(len(tcodes)):
        # then transfer it to code and then tokenize the code
        t_code = tcodes[i]
        t = tokenize_python(t_code)
        test_tokens = np.array(t)[np.newaxis, :]  # new dimension 1x32
        #  get the vector for all item-string in the raises[] and tests[]
        test_vec = vectorize_trim_pad(test_tokens, embed_model, embed_dim, seq_length=test_seq_length)
        test_vectors = np.append(test_vectors, test_vec, axis=0)
    print("get test-vectors of shape ", test_vectors.shape)  # (71, 32, 32)#(85819, 32, 32)

    # get raise vectors------------------------------------------

    raise_vectors = a = np.empty((0, raise_seq_length, 32))
    print("---------------------------------")
    print("creating word embedding vectors for raises code,sequence length=", raise_seq_length, ",embed dimension=",
          embed_dim)
    for i in range(len(rcodes)):
        r_code = rcodes[i]
        t = tokenize_python(r_code)
        raises_tokens = np.array(t)[np.newaxis, :]
        raise_vec = vectorize_trim_pad(raises_tokens, embed_model, embed_dim, seq_length=raise_seq_length)
        raise_vectors = np.append(raise_vectors, raise_vec, axis=0)
    print("get raise-vectors of shape ", raise_vectors.shape)
    input_vectors = np.concatenate((test_vectors, raise_vectors), axis=1)
    print("concatenate[test_vectors,raise_vectors],get test samples of shape", input_vectors.shape)

    print("making predictions now")
    torchinput_vectors = torch.from_numpy(input_vectors)
    torchinput_vectors = torchinput_vectors.float()

    model.eval()
    with torch.no_grad():

        out = model(torchinput_vectors[:, 0:test_sequence_length, :].to(device),
                    torchinput_vectors[:, test_sequence_length:test_sequence_length + raise_sequence_length, :].to(device))
        # print("out.data--------------------------------------------------------")
        # print(out.data)
        # Softmax = nn.Softmax(dim=1)
        # incon_score = Softmax(out.data)[:,0].cpu().detach().data.numpy()
        sig = nn.Sigmoid()
        incon_score = sig(out.data)[:,0].cpu().detach().data.numpy()
        _, pre = torch.max(out.data, 1)
        prediction = pre.cpu().detach().data.numpy()
    print("prediction labels (0:inconsistent,1:consistent):")
    print(prediction)
    print("inconsistent scores:")
    print(incon_score)
    all_line_no = get_line_num(test_func_list)
    return [all_line_no,incon_score]


def load_model(source):

    if (torch.cuda.is_available()):
        model = torch.load(source,map_location=torch.device('cuda:0'))
    else:
        model = torch.load(source, map_location=torch.device('cpu'))
    return model


def write_predictions(destination, predictions):
    """
    TODO: Implement your code to write predictions to file. For format
    of the JSON file refer to project description.
    """
    print("writing prediction to file", destination)
 
    all_line_no = predictions[0]
    prediction_score = predictions[1]

    idx = 0
    dic_output = []

    for lines in all_line_no:
        rec = {}
        for no in lines:
            rec[str(no)] =float(prediction_score[idx])
            idx += 1
        dic_output.append(rec)
    print(dic_output)


    with open(destination, "w") as f:
        json.dump(dic_output, f)


if __name__ == "__main__":
    args = parser.parse_args()

    # load the serialized model
    model = load_model(args.model)

    # predict incorrect location for each test example.
    predictions = predict(model, args.source)

    # write predictions to file
    write_predictions(args.destination, predictions)

# python Predict_colab.py --model 'model4.pth' --source "shared_resources/real_test_for_milestone3/real_consistent1.json" --destination t.json

