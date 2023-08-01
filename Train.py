#!/usr/bin/python3
#final -v
import argparse

import math
import libcst as cst
import json
import pickle
import nltk
from nltk.tokenize import word_tokenize
import libcst as cst

nltk.download('punkt')

import numpy as np
import random

from tokenize import tokenize
from io import BytesIO
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import torch.utils.data as Data
from gensim.models import FastText
import gensim

parser = argparse.ArgumentParser()
parser.add_argument(
    '--source', help="Path to functions file used for training.", required=True)
parser.add_argument(
    '--destination', help="Path to save your trained model.", required=True)

#define embeding
embed_dim = 32
test_seq_length = 16
raise_seq_length = 20

# define ensenmble bi-lstm model
sequence_length = 32
input_size = 32  # emebedding vector feature
hidden_size = 128  # hidden_size for LSTM
num_layers = 3  # layer_size for LSTM
num_classes = 2  # output 2 classes
batch_size = 32
num_epochs = 30
learning_rate = 0.001
weight_decay = 0.0
test_sequence_length = 16  # test length to feed into the model
raise_sequence_length = 20  # raise length to feed into the model


#  findif class, search and append if
class FindIf(cst.CSTVisitor):
    ifs = []

    def __init__(self):
        self.ifs = []

    def visit_If(self, node: cst.If):
        try:
            self.ifs.append(node)
        except Exception as e:
            print(e)


#  find raise class, test if there is raise
class FindRaise(cst.CSTVisitor):
    raise_flag = False

    def __init__(self):
        self.raise_flag = False

    def visit_Raise(self, node: cst.Raise):
        try:
            self.raise_flag = True
        except Exception as e:
            print(e)


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


def count_if(tokens):
    if_cnt = 0
    for item in tokens:
        if (item == 'if'):
            if_cnt += 1
    return if_cnt


# count raise
def count_raise(tokens):
    if_cnt = 0
    for item in tokens:
        if (item == 'raise'):
            if_cnt += 1
    return if_cnt


# count token==str in tokens
def count_token(tokens, str):
    if_cnt = 0
    for item in tokens:
        if (item == str):
            if_cnt += 1
    return if_cnt


# check if there exist "else/elif"
def has_el(tokens):
    if (("elif" in tokens) or ("else" in tokens)):
        return True
    else:
        return False


# find test class
class FindTest(cst.CSTVisitor):
    tests = []

    def __init__(self):
        self.tests = []

    def visit_If(self, node: cst.If):
        try:
            self.tests.append(node.test)
        except Exception as e:
            print(e)


class GetRaise(cst.CSTVisitor):
    raises = []

    def __init__(self):
        self.raises = []

    def visit_Raise(self, node: cst.Raise):
        try:
            self.raises.append(node)
        except Exception as e:
            print(e)


def extract(code):
    tests = []
    raises = []
    mdl = cst.parse_module(code)

    test_finder = FindTest()
    _test = mdl.visit(test_finder)
    tests = test_finder.tests

    raise_getter = GetRaise()
    _raise = mdl.visit(raise_getter)
    raises = raise_getter.raises

    return [tests, raises]


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


def recreate_test_code(code_src):
    if ("!=" in code_src):

        new_code = code_src.replace("!=", "==")
    elif ("==" in code_src):

        new_code = code_src.replace("==", "!=")
    elif ("<" in code_src):

        new_code = code_src.replace("<", ">")
    elif (">" in code_src):

        new_code = code_src.replace(">", "<")
    elif ("not " in code_src):

        new_code = code_src.replace("not ", "")
    elif ("and " in code_src):
        new_code = code_src.replace("and ", "or ")
    elif ("or " in code_src):
        new_code = code_src.replace("or ", "and ")
    else:  # if no changes made
        return False

    return new_code


def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


setup_seed(333)

# two sub-Bidirectional-LSTM
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
        # h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)
        # c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)
        # out, _ = self.lstm(x, (h0, c0))

        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return out


# for updating learning rate
def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# ensemble model
class MyEnsemble(nn.Module):
    def __init__(self, modelA, modelB, input_len, nb_classes=num_classes):
        super(MyEnsemble, self).__init__()
        # self.ln1 = torch.nn.LayerNorm([sequence_length, input_size], elementwise_affine=False)
        # self.ln2 = torch.nn.LayerNorm([sequence_length, input_size], elementwise_affine=False)

        # two sub-LSTM model---------------------
        self.modelA = modelA
        self.modelB = modelB

        # self.bn3 = nn.BatchNorm1d(nb_classes)

        # linear layers--------------------
        self.classifier1 = nn.Linear(hidden_size * 4, hidden_size)
        self.dropout1 = nn.Dropout(p=0.4)
        # self.relu1 = nn.ReLU(inplace=True)
        self.classifier2 = nn.Linear(hidden_size, nb_classes)
        self.dropout2 = nn.Dropout(p=0.4)
        # self.relu2 = nn.LeakyReLU(inplace=True)
        # self.classifier3 = nn.Linear(int(hidden_size / 4), nb_classes)
        # self.dropout3 = nn.Dropout(p=0.4)

        # initialization-----------------------
        nn.init.xavier_uniform_(self.classifier1.weight, gain=math.sqrt(2.0))
        nn.init.xavier_uniform_(self.classifier2.weight, gain=math.sqrt(2.0))


    def forward(self, xa, xb):
        # sub models ouputs
        x1 = self.modelA(xa)  # condition
        x2 = self.modelB(xb)  # raise

        # concatenation
        x_con = torch.cat((x1, x2), dim=1)

        # linear layers
        x_c1 = self.classifier1(x_con)
        x_c1 = self.dropout1(x_c1)

        x_c2 = self.classifier2(x_c1)
        x_c2 = self.dropout2(x_c2)

        return x_c2


def train_model(source):
    #DATA_PART = "shared_resources/data/functions_list.json"
    with open(source) as dt:
        func_list = json.load(dt)
    len_func_list = len(func_list) # len(func_list) = 100000
    print("read function samples of length:",len_func_list)

    # 1 get all if-raise AST in "all_if_raise"-------------------------
    # without repetition （no child）
    print("---------------------------------------------------")
    print("extracting if-raise AST in all functions...")
    all_if_raise = []
    for i in range(len(func_list)):
        if (i % 1000 == 0):
            print("getting if-raise AST in ", i,"/",len_func_list,"th function")
        code = func_list[i]
        mdl = cst.parse_module(code)

        if_finder = FindIf()
        _ = mdl.visit(if_finder)
        all_if = if_finder.ifs

        if_raises = []
        raise_finder = FindRaise()
        for _if in all_if:
            raise_finder.raise_flag = False
            _ = _if.visit(raise_finder)
            if (raise_finder.raise_flag):
                if_raises.append(_)
        all_if_raise.extend(if_raises)

    # 2 get all if-raise code , store in "if_raise_codes"----------------------------
    if_raise_codes = []
    print("---------------------------------------------------")
    print("trasferring from ast to code for all if-raise pharses:")
    for _ir in all_if_raise:
        _code = cst.Module([_ir]).code
        if_raise_codes.append(_code)
    print("get if_raise_codes, length of ", len(if_raise_codes))

    # 3 get rid of multi-if 2
    # new_simple_codes are if-pharses with only ONE if, still have else!
    read_code = if_raise_codes
    new_simple_codes = []
    print("---------------------------------------------------")
    print("getting rid of complex pharases(multi-if pharases)")
    for i in range(len(read_code)):
        try:
            if (i % 10000 == 0):
                print("processing ", i, "/", len(read_code))
            code = read_code[i]
            tokens = tokenize_python(code)

            if_cnt = count_if(tokens)
            if (if_cnt == 1):
                new_simple_codes.extend([code])
        except Exception as e:
            print("skip code", i, "due to", e)

    # 4 extract test and raises ast
    tests = []
    raises = []
    print("---------------------------------------------------")
    print("extracting test and raises ast")
    for i in range(len(new_simple_codes)):
        if (i % 10000 == 0):
            print("processing ", i, "/", len(new_simple_codes))
        code = new_simple_codes[i]
        tokens = tokenize_python(code)
        if (not has_el(tokens)):
            # print(good_if_raise_code[i])
            t = extract(code)[0]
            r = extract(code)[1]
            if (len(t) > 1 or len(r) > 1):
                continue
            else:
                tests.extend(t)
                raises.extend(r)
    print("get ", len(tests), "test samples")
    # 85819
    print("get ", len(raises), "raise samples")

    # #######write##############################
    # with open('pydata/test_ast.dat', 'wb') as dest1:
    # 	 pickle.dump(tests,dest1)
    # with open('pydata/raise_ast.dat', 'wb') as dest2:
    # 	 pickle.load(raises,dest2)
    # #############################################

    # #######read##############################
    # with open('pydata/test_ast.dat', 'rb') as dest1:
    #     tests = pickle.load(dest1)
    # with open('pydata/raise_ast.dat', 'rb') as dest2:
    #     raises = pickle.load(dest2)
    # #############################################

    # 5 transferring from ast to code
    print("---------------------------------------------------")
    print("transferring from ast to code for test and raises")
    tests_code = []
    raises_code = []
    for x in tests:
        tests_code.append(cst.Module([x]).code)
    for y in raises:
        raises_code.append(cst.Module([y]).code)

    # print("showing several extracted code...")
    # i = 0
    # for x, y in zip(tests_code, raises_code):
    #     print("----------------------")
    #     print(x)
    #     print(y)
    #     i += 1
    #     if (i == 5):
    #         break

    # 6 load the embeded model
    print("loading the FastText model")
    ##try loading the tokenizer
    PATH = 'embed_if_32.mdl'
    embed_model = gensim.models.FastText.load(PATH)

    # 7.1 get testes vectors
    embed_dim = 32
    test_seq_length = 16
    test_vectors = np.empty((0, test_seq_length, embed_dim))
    print("---------------------------------")
    print("creating word embedding vectors for tests code,sequence size=", test_seq_length, ",embed dimension=",
    	  embed_dim)
    for i in range(len(tests_code)):

    	if (i % 10000 == 0):
    		print("processing ", i, "/", len(tests_code))
    	#  then transfer it to code and then tokenize the code
    	t_code = tests_code[i]
    	t = tokenize_python(t_code)
    	test_tokens = np.array(t)[np.newaxis, :]
    	#  get the vector for all item-string in the raises[] and tests[]
    	test_vec = vectorize_trim_pad(test_tokens, embed_model, embed_dim, seq_length=test_seq_length)
    	test_vectors = np.append(test_vectors, test_vec, axis=0)

    print("get test-vectors of size ", test_vectors.shape)

    # #######write##############################
    # with open('pydata/test_vectors.dat', 'wb') as dest:
    # 	pickle.dump(test_vectors, dest)
    # ###########################################

    # 7.2 get raises vectors
    embed_dim = 32
    raise_seq_length = 20
    raise_vectors = a = np.empty((0, raise_seq_length, embed_dim))
    print("---------------------------------")
    print("creating word embedding vectors for raises code,sequence size=", raise_seq_length, ",embed dimension=",
    	  embed_dim)
    for i in range(len(raises_code)):
    	#  then transfer it to code and then tokenize the code
    	if (i % 10000 == 0):
    		print("processing ", i, "/", len(raises_code))
    	r_code = raises_code[i]
    	t = tokenize_python(r_code)
    	raises_tokens = np.array(t)[np.newaxis, :]
    	#  get the vector for all item-string in the raises[] and tests[]
    	raise_vec = vectorize_trim_pad(raises_tokens, embed_model, embed_dim, seq_length=raise_seq_length)
    	raise_vectors = np.append(raise_vectors, raise_vec, axis=0)
    print("get raise-vectors of size ", raise_vectors.shape)  # (85819, 32, 32)

    # #######write##############################
    # with open('pydata/raise_vectors.dat', 'wb') as dest:
    # 	pickle.dump(raise_vectors, dest)
    # ###########################################
    #
    # #######read##############################
    # with open('pydata/test_vectors.dat', 'rb') as dest1:
    #     test_vectors = pickle.load(dest1)
    # with open('pydata/raise_vectors.dat', 'rb') as dest2:
    #     raise_vectors = pickle.load(dest2)
    # with open('pydata/test_vectors.dat', 'rb') as dest3:
    #     test_vectors = pickle.load(dest3)
    # with open('pydata/raise_vectors.dat', 'rb') as dest4:
    #     raise_vectors = pickle.load(dest4)
    # ###########################################

    # 8.1 creating inconsistent inputs for test code
    test_seq_length = 16
    raise_seq_length = 20
    embed_dim = 32
    incon_vectors = np.empty((0, test_seq_length + raise_seq_length, embed_dim))
    nochanges = 0
    change_test = 0
    print("---------------------------------------------------")
    print("creating INconsistent inputs...")
    print("modifying test codes...")
    for i in range(len(tests_code)):
    	if (i % 10000 == 0):
    		print("processing ", i, "/", len(tests_code))

    	new_test_code = recreate_test_code(tests_code[i])

    	if (new_test_code != False):

          t = tokenize_python(new_test_code)
          new_test_tokens = np.array(t)[np.newaxis, :]
          new_test_vector = vectorize_trim_pad(new_test_tokens, embed_model, embed_dim, seq_length=test_seq_length)
          new_vector = np.concatenate((new_test_vector, raise_vectors[i][np.newaxis, :]), axis=1)
          incon_vectors = np.append(incon_vectors, new_vector, axis=0)
    	else:
          nochanges += 1
          new_raise_idx = random.randint(0, len(tests_code)-1)
          while (new_raise_idx == i):
            new_raise_idx = random.randint(0, len(tests_code)-1)
          new_vector = np.concatenate((test_vectors[i][np.newaxis, :], raise_vectors[new_raise_idx][np.newaxis, :]),
          							axis=1)
          incon_vectors = np.append(incon_vectors, new_vector, axis=0)
    print("after modify test code, get new inconsistent vectors of shape ",incon_vectors.shape)
    # #######write##############################
    # with open('pydata/test_incon_vectors.dat', 'wb') as dest:
    # 	pickle.dump(incon_vectors, dest)
    # ###########################################

    # #######read##############################
    # with open('pydata/test_incon_vectors.dat', 'rb') as dest3:
    #     incon_vectors = pickle.load(dest3)
    # print("read incon of shape",incon_vectors.shape)
    # ###########################################
    # # 8.2 creating inconsistent inputs for raise code
    # print("modifying raise codes...")
    # change_raise = 0
    # for i in range(len(raises_code)):
    #     if (i % 10000 == 0):
    #         print("processing ", i, "/", len(raises_code))
    #     new_raise_code = recreate_test_code(raises_code[i])
    #     if (new_raise_code != False):
    #         t = tokenize_python(new_raise_code)
    #         new_raise_tokens = np.array(t)[np.newaxis, :]
    #         new_raise_vector = vectorize_trim_pad(new_raise_tokens, embed_model, embed_dim, seq_length=raise_seq_length)
    #         new_vector = np.concatenate((test_vectors[i][np.newaxis, :], new_raise_vector), axis=1)
    #         incon_vectors = np.append(incon_vectors, new_vector, axis=0)
    #         change_raise += 1

    incon_vectors = np.array(incon_vectors)
    print("after modifying codes, get inconsistent vectors of shape", incon_vectors.shape)
    # print("changed", len(tests_code) - nochanges, " of test/condition codes, changed ", change_raise,
    #       " of raise codes, shuffled ", nochanges, "codes")
    print("changed", len(tests_code) - nochanges, " of test/condition codes, shuffled ", nochanges, "codes")

    # #######write##############################
    # with open('pydata/test_raise_incon_vectors.dat', 'wb') as dest:
    #     pickle.dump(incon_vectors, dest)
    # ###########################################

    # #######read##############################
    # with open('pydata/est_vectors.dat', 'rb') as dest1:
    #     test_vectors = pickle.load(dest1)
    # with open('pydata/raise_vectors.dat', 'rb') as dest2:
    #     raise_vectors = pickle.load(dest2)
    # with open('pydata/incon_vectors.dat', 'rb') as dest3:
    #     incon_vectors = pickle.load(dest3)
    # ###########################################

    # 9 creating inputs for the model
    # 9.1 make consistent vectors [test,raise]
    print("---------------------------------------------------")
    con_vectors = np.concatenate((test_vectors, raise_vectors), axis=1)
    print("concatenate[test_vectors,raise_vectors],get consistet samples of shape", con_vectors.shape)

    # 9.2 put con and incon together
    both_vectors = np.concatenate((con_vectors, incon_vectors), axis=0)
    print("put consistent and inconsistent samples together, get samples of shape", both_vectors.shape)

    # #######write##############################
    # with open('pydata/newnew_both_vectors.dat', 'wb') as dest:
    #     pickle.dump(both_vectors, dest)
    # ###########################################

    # 10 get one-hot labels
    incon_len = incon_vectors.shape[0]
    con_len = con_vectors.shape[0]
    results0 = np.zeros(con_len)
    results1 = np.ones(incon_len)
    results = np.concatenate((results0, results1), axis=0)
    results_oppo = np.concatenate((results1, results0), axis=0)
    results_onehot = np.concatenate((results[:, np.newaxis], results_oppo[:, np.newaxis]), axis=1)
    print("---------------------------------------------------")
    print("made one-hot labels of shape:", results_onehot.shape)

    # 11 shuffle
    inputs_shuffle = both_vectors
    results_onehot_shuffle = results_onehot
    shuffle_state = np.random.get_state()
    np.random.shuffle(inputs_shuffle)
    np.random.set_state(shuffle_state)
    np.random.shuffle(results_onehot_shuffle)

    # 12 make validation data
    batch_size = 32
    val_data = inputs_shuffle[0:2000]
    val_target = results_onehot_shuffle[0:2000]
    print("validation dataset shape:", val_data.shape)
    print("validation target shape:", val_target.shape)
    val_dataset_onehot = Data.TensorDataset(torch.from_numpy(val_data), torch.from_numpy(val_target))
    val_loader_onehot = Data.DataLoader(
        dataset=val_dataset_onehot,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
    )

    # 13 make train data
    batch_size = 32
    train_data = inputs_shuffle[2000:]
    train_target = results_onehot_shuffle[2000:]
    train_dataset_onehot = Data.TensorDataset(torch.from_numpy(train_data),
                                              torch.from_numpy(train_target))
    train_loader_onehot = Data.DataLoader(
        dataset=train_dataset_onehot,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
    )

    # model
    modelA = subBLSTM(input_size, hidden_size, num_layers, num_classes)
    modelA = modelA.to(device)
    modelB = subBLSTM(input_size, hidden_size, num_layers, num_classes)
    modelB = modelB.to(device)
    model = MyEnsemble(modelA, modelB, hidden_size * 4)
    model = model.to(device)

    # 15 Train  Network !!!!!!!!!!!!!!
    val_losses = []
    val_acces = []
    losses = []
    acces = []
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    for epoch in range(num_epochs):
        correct = torch.zeros(1).to(device)
        total = torch.zeros(1).to(device)
        sum_loss = torch.zeros(1).to(device)
        batches = torch.zeros(1).to(device)
        print("epoch ", epoch + 1, "****************************************************")
        model.train()
        for batch_idx, (data, targets) in enumerate(
                train_loader_onehot):

            # forward
            data = data.float().to(device)
            targets = targets.float().to(device)
            scores = model(data[:, 0:test_sequence_length, :],
                           data[:, test_sequence_length:test_sequence_length + raise_sequence_length, :]).to(device)
            optimizer.zero_grad()
            loss = criterion(scores, targets).to(device)

            # backward
            loss.backward()

            # gradient descent or adam step
            optimizer.step()

            if (batch_idx+1) % 256 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, batch_idx + 1, 0, loss.item()))

            prediction = torch.argmax(scores, dim=1).to(device)
            curr_correct = (prediction == targets[:, 1]).sum().float().to(device) #the second column represent consistency

            correct += curr_correct
            total += len(targets)
            sum_loss += loss.item()
            batches += 1


        # validation-------------------------------------------------------------------------------------
        model.eval()
        val_loss_sum = torch.zeros(1).to(device)
        val_batches = torch.zeros(1).to(device)
        val_correct = torch.zeros(1).to(device)
        val_total = torch.zeros(1).to(device)
        # val_acc_sum = 0.0
        for val_step, (val_data, val_targets) in enumerate(val_loader_onehot, 1):
            with torch.no_grad():
                val_data = val_data.float().to(device)
                val_targets = val_targets.float().to(device)
                val_scores = model(val_data[:, 0:test_sequence_length, :],
                                   val_data[:, test_sequence_length:test_sequence_length + raise_sequence_length, :])
                val_loss = criterion(val_scores, val_targets)

                val_prediction = torch.argmax(val_scores, dim=1)
                valcurr_correct = (val_prediction == val_targets[:, 1]).sum().float()

            val_correct += valcurr_correct
            val_loss_sum += val_loss.item()
            val_batches += 1
            val_total += len(val_targets)

        # record--------------------------------------------------------------------------------------
        val_acces.append((val_correct / val_total).cpu().detach().data.numpy())
        val_acc_str = 'Validation Accuracy: %f' % ((val_correct / val_total).cpu().detach().data.numpy())
        print(val_acc_str)
        val_losses.append((val_loss_sum / val_batches).cpu().detach().data.numpy())
        val_loss_str = 'Mean Validation Loss: %f' % ((val_loss_sum / val_batches).cpu().detach().data.numpy())
        print(val_loss_str)
        acces.append((correct / total).cpu().detach().data.numpy())
        acc_str = 'Train Accuracy: %f' % ((correct / total).cpu().detach().data.numpy())
        print(acc_str)
        loss_str = 'Mean Train loss: %f' % ((sum_loss / batches).cpu().detach().data.numpy())
        print(loss_str)
        losses.append((sum_loss / batches).cpu().detach().data.numpy())

    # Decay learning rate
    # learning_rate = (0.95**epoch)*learning_rate
    # update_lr(optimizer, learning_rate)

    return model


def save_model(model, destination):
    torch.save(model, destination)


if __name__ == "__main__":
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    args = parser.parse_args()

    model = train_model(args.source)

    save_model(model, args.destination)
# python Train.py --source "shared_resources/data/functions_list.json" --destination model.pth
