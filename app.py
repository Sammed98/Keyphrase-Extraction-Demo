import requests
import torch
import os
from pytorch_pretrained_bert import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel
import re
import en_core_web_sm
from torchcrf import CRF
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from flask import Flask, render_template, url_for, request
import sys
import random
import time
import glob
import codecs
from nltk.tokenize import word_tokenize, sent_tokenize
from keras.preprocessing import sequence
import nltk.data
import statistics 
import pickle
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)
global model, BERT_FP, bert, tokenizer, nlp




# model = torch.load('model_sciBERT_CRF10.pth')
# BERT_FP = 'scibert_scivocab_uncased'
# bert = BertModel.from_pretrained(BERT_FP)
# tokenizer = BertTokenizer(vocab_file=BERT_FP+'/vocab.txt')
# nlp = en_core_web_sm.load()

@app.route("/", methods=['POST', 'GET'])
def home():
    
    
#   global model, BERT_FP, bert, tokenizer, nlp
    model = torch.load('model_sciBERT_CRF10.pth')
    BERT_FP = 'scibert_scivocab_uncased'
    bert = BertModel.from_pretrained(BERT_FP)
    tokenizer = BertTokenizer(vocab_file=BERT_FP+'/vocab.txt')
    nlp = en_core_web_sm.load()
    datatowrite=[]
    result = ''
    if (request.method == 'POST'):
        token_indices = []
        file_raw = request.form.get('abstract')
        actual_file = open('abstract_str/abstract.txt', 'w')
        actual_file.write(file_raw)
        actual_file.close()
        file = file_raw.lower()
        tokens_list = tokenizer.tokenize(file)
        n = 0
        for i, item in enumerate(tokens_list):
            try:
                start_index = file.index(item.strip('#'))
            except:
                start_index = 100
            if ((start_index < 5 or unk == 1) and item != '[UNK]'):
                token_indices.append((start_index + n, n + start_index+len(item.strip('#'))))

                n = token_indices[-1][-1]
                file = file[start_index+len(item.strip('#')):]
            else:
                token_indices.append((-1, -1))

                if (item != '[UNK]'):
                    n += len(item.strip('#'))
                    file = file[len(item.strip('#')):]

        with torch.no_grad():
            inputs = tokenizer.convert_tokens_to_ids(tokens_list)
            inputs = bert(torch.tensor([inputs]))[0]
            for j in range(len(inputs)):
                inputs[j] = inputs[j].numpy()
            inputs = torch.tensor(np.array(inputs))
            prediction = model(inputs.permute(1, 2, 0, 3).squeeze(0))
            output = prediction[0]

        dic = {}
        dataarr = file_raw
        tagsarr = output
        indicesarr = token_indices
    
        indicesdata = []
        datatowrite = []
        for j in range(len(tagsarr)):
            if (tagsarr[j] == 0 or tagsarr[j] == 4):
                indicesdata.append(list(indicesarr[j]))
            if (tagsarr[j] == 1 or tagsarr[j] == 2):
                indicesdata[-1][1] = indicesarr[j][1]

        indicestowrite = indicesdata

        ind_temp = []
        data_temp = []
        for j in indicestowrite:
            ind_temp.append(j)
            data_temp.append(dataarr[j[0]:j[1]])

        indicestowrite = []
        datatowrite = []
        for j in range(len(ind_temp)):
            temp = nlp(data_temp[j])
            count = 0
            for k in temp:
                count += 1

            if (count == 1):
                ind = [[k.start()+1, k.start()+1+len(data_temp[j])] for k in re.finditer('[^a-z]'+re.escape(data_temp[j].lower())+'[^a-z]', dataarr.lower()) if [k.start()+1, k.start()+1+len(data_temp[j])] not in ind_temp and 
                      [k.start()+1, k.start()+1+len(data_temp[j])] not in indicestowrite]
                temp_ind = []
                dat = []
                for l in ind:
                    if (dataarr[l[0]:l[1]].lower() != dataarr[l[0]:l[1]]):
                        dat.append(dataarr[l[0]:l[1]])
                        temp_ind.append(l)
                indicestowrite += temp_ind
                datatowrite += dat


        ind_temp = ind_temp + indicestowrite
        data_temp = data_temp + datatowrite
        indicestowrite = []
        datatowrite = []

        for j in range(len(data_temp)):
            temp_2 = nlp(data_temp[j])
            temp = []
            for word in temp_2:
                temp.append((len(word.text), word.text))

            if (len(temp) == 1):
                if (str(temp[0][1]).lower()!=str(temp[0][1]) or re.match('^[a-z]+$', temp[0][1])==None or len(temp[0][1])>3):
                    indicestowrite.append(ind_temp[j])
                    datatowrite.append(data_temp[j])
            else:
                indicestowrite.append(ind_temp[j])
                datatowrite.append(data_temp[j])
        indicestowrite = sorted(indicestowrite, key=lambda x:x[0])
        if(len(indicestowrite) == 0):
            return render_template("index.html", keyphrases = file_raw)
        print(indicestowrite)
        annotation_file = open('abstract_str/abstract.ann', 'w')
        for qwe in range(len(indicestowrite)):
            annotation_file.write('T'+str(qwe+1) + '\t' + 'Process ' + str(indicestowrite[qwe][0]) + ' ' + str(indicestowrite[qwe][1]) + '\t' + file_raw[indicestowrite[qwe][0]:indicestowrite[qwe][1]] + '\n')
        annotation_file.close()
        X_test, y_test_gold, _ , test_entities = read_and_map('abstract_str', mapper)
        loaded_model = pickle.load(open('finalized_model_joined.sav' , 'rb'))
        predictions = loaded_model.predict(X_test)
        y_values = ['Process', 'Material', 'Task']
        document_abbr = {}
        asd = os.listdir('abstract_str')
        for i in range(len(asd)):
          document_abbr[asd[i][:-4]] = {}

        for i in range(len(predictions)):
          if(test_entities[i].string == test_entities[i].string.upper() and len(test_entities[i].string) > 1 ):
            if(y_values[predictions[i]] == "Material"):
              predictions[i] = y_values.index("Process")

          if(test_entities[i].string == test_entities[i].string.capitalize() and len(test_entities[i].string) == 2):
            predictions[i] = y_values.index("Material")
          
          tmp = test_entities[i].string.split(" ")
          if(len(tmp) == 1):
            if(test_entities[i].string == test_entities[i].string.upper() and hasNumbers(test_entities[i].string)):
              predictions[i] = y_values.index("Material")
          
          


          if(test_entities[i].string == test_entities[i].string.upper()):
            try:
              predictions[i] = document_abbr[test_entities[i].docid][test_entities[i].string]
            except:
              obracket = test_entities[i].start - 1
              cbracket = test_entities[i].end
              file = open('abstract_str/' + test_entities[i].docid + '.txt', 'r').read()
              if(file[obracket] == '(' and file[cbracket] == ')'):
                if(test_entities[i].start - test_entities[i-1].end == 2):
                  # print(test_entities[i].string, '\t',test_entities[i-1].string ,'\t' ,test_entities[i].start, '\t',test_entities[i-1].end )
                  document_abbr[test_entities[i].docid][test_entities[i].string] = predictions[i-1]
                  predictions[i] = predictions[i-1]

          for j in range(len(tmp)):
            if(len(tmp[j]) == 1 and tmp[j] == tmp[j].upper()):
              predictions[i] = y_values.index("Material")



        # print(predictions)
        

        n = 0
        result = []
        last_closing = 0
        for i in range(len(indicestowrite)):
            qwe_temp = file_raw[n:indicestowrite[i][0]]
            if(qwe_temp != ''):
                result.append(qwe_temp)
            temp = ''
            if(predictions[i] == 0):
                temp = '<span style="background-color:rgba(152, 252, 3, 0.5);"><strong>' + file_raw[indicestowrite[i][0]:indicestowrite[i][1]] + '</strong></span>'
            elif(predictions[i] == 1):
                temp = '<span style="background-color:rgba(252, 152, 3, 0.5);"><strong>' + file_raw[indicestowrite[i][0]:indicestowrite[i][1]] + '</strong></span>'
            elif(predictions[i] == 2):
                temp = '<span style="background-color:rgba(3, 152, 252, 0.5);"><strong>' + file_raw[indicestowrite[i][0]:indicestowrite[i][1]] + '</strong></span>'

            if(indicestowrite[i][1] > last_closing):
                result.append(temp)                
                last_closing = indicestowrite[i][1]
                n = indicestowrite[i][1]
            # else:
            #     ov_string = file_raw[indicestowrite[i][0]:indicestowrite[i][1]]
            #     temp_start = result[-1].index(ov_string)
            #     result[-1] = result[-1][:temp_start] + temp + result[-1][ temp_start+indicestowrite[i][1] - indicestowrite[i][0]:]

            # result += '<span style="background-color:rgba(152, 252, 3, 0.5);"><strong>' +  file_raw[i[0]:i[1]] + '</strong></span>'
            

        result += file_raw[n:]
        # print(result)
        result = "".join(result)
    return render_template("index.html", keyphrases = result)



if __name__ == "__main__":
    def hasNumbers(inputString):
        return any(char.isdigit() for char in inputString)

    def segment_text(text):
        sentence_id = 0
        token_id = 0
        tail = text
        accumulator = 0
        sentences = [sentence for sentence in SentenceSplitter().split(text)]
        sentence_object_array = []
        for sentence in sentences:
            escaped_sentence = re.escape(sentence)
            sentence_occurrence = re.search(escaped_sentence, tail)
            s_start, s_end = sentence_occurrence.span()
            sentence_start = accumulator + s_start
            sentence_end = accumulator + s_end

            tokens = [word for word in word_tokenize(sentence)]
            token_object_array = []
            tail_for_token_search = sentence
            token_accumulator = 0
            for token in tokens:
                escaped_token = re.escape(token)
                token_occurrence = re.search(escaped_token, tail_for_token_search)
                t_start, t_end = token_occurrence.span()
                # global offsets
                token_start = sentence_start + token_accumulator + t_start
                token_end = sentence_start + token_accumulator + t_end
                token_accumulator += t_end

                token_object = Token(token_start, token_end, utf8ify(token), token_id)
                token_object_array.append(token_object)
                # keep searching in the rest
                tail_for_token_search = tail_for_token_search[t_end:]
                token_id += 1

            sentence_object = Sentence(sentence_start, sentence_end, token_object_array, utf8ify(sentence), sentence_id)
            sentence_object_array.append(sentence_object)
            for tok in sentence_object.token_array:
                tok.sentence = sentence_object

            accumulator += s_end
            # keep rest of text for searching
            tail = tail[s_end:]
            sentence_id += 1

        return sentence_object_array

    def utf8ify(obj):
        if sys.version_info < (3,):
            return obj.encode("utf-8")
        else:
            return str(obj)

    def read_and_map(src, mapper, y_values = None):
        r = ScienceIEBratReader(src)
        X = []
        y = []
        entities = []
        for document in r.read():
            for entity in document.entities:
                if entity.uid in document.cover_index:  # only proceed if entity has been successfully mapped to tokens
                    X += [mapper.to_repr(entity, document)]
                    y += [entity.etype]
                    entities += [entity]
        # print(X)
        X = np.vstack(X)

        y_values = y_values if y_values is not None else list(set(y))
        try:
          y = np.array([y_values.index(y_val) for y_val in y])
        except ValueError:
          y = np.array([0 for y_val in y])
        return X, y, y_values, entities


    class LSTMmodel(nn.Module):
        def __init__(self, embedding_dim, hidden_dim):
            super(LSTMmodel, self).__init__()
            self.hidden_dim = hidden_dim
            self.lstm = nn.LSTM(embedding_dim, hidden_dim*4, bidirectional=True)
            self.lstm2 = nn.LSTM(hidden_dim*8, hidden_dim*2, bidirectional=True)
            self.lstm3 = nn.LSTM(hidden_dim*4, hidden_dim, bidirectional=True)
            self.hidden2tag = nn.Linear(hidden_dim*2, 5)
            self.crf = CRF(5)

        def train1(self, sentence, tags):
            tag_space = self.forward_pass(sentence)
            return self.crf(tag_space.view(len(sentence), 1, -1), tags)*-1

        def forward_pass(self, sentence):
            lstm_out, _ = self.lstm(sentence.contiguous().view(len(sentence), 1, -1))
            lstm_out, _ = self.lstm2(lstm_out)
            lstm_out, _ = self.lstm3(lstm_out)
            tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
            return tag_space

        def forward(self, sentence):
            tag_space = self.forward_pass(sentence)
            return self.crf.decode(tag_space.view(len(sentence), 1, -1))


    embedding_src = 'bow2.words'
    class VSM:
        def __init__(self, src):
            self.map = {}
            self.dim = None
            self.source = src.split("/")[-1] if src is not None else "NA"
            if src is not None:
                with open(src) as f:
                    i = 0
                    for line in f:
                        word = line.split()[0]
                        embedding = line.split()[1:]
                        self.map[word] = np.array(embedding, dtype=np.float32)
                        i += 1
                    self.dim = len(embedding)
            else:
                self.dim = 1

        def get(self, word):
            word = word.lower()  
            if word in self.map:
                return self.map[word]
            else: 
                return np.zeros(self.dim)


    class Entity:
        def __init__(self, start, end, etype, string, uid, docid):
            self.start = start
            self.end = end
            self.etype = etype
            self.string = string
            self.uid = uid
            self.docid = docid
    class Token:
        def __init__(self, start, end, string, tid = None, sentence = None):
            self.start = start
            self.end = end
            self.string = string
            self.tid = tid
            self.sentence = sentence
    class Sentence:
        def __init__(self, start, end, token_array, string, sid = None):
            self.start = start
            self.end = end
            self.token_array = token_array
            self.string = string
            self.sid = sid

    class Document:
        def __init__(self, name, entities, text):
            self.name = name
            self.entities = entities
            self.entity_index = {}
            self.token_index = {}
            self.sentence_index = {}
            self.cover_index = {}  # for each entity which tokens it includes
            self.text = text
            self.tokens = []
            self.sentences = []

            self.errors = 0
            self.fixed = 0
            self.index()

        def index_sentences_and_tokens(self):
            segmented_text = segment_text(self.text)
            for sentence in segmented_text:
                self.sentence_index[sentence.sid] = sentence
                self.sentences.append(sentence)
                for token in sentence.token_array:
                    self.token_index[token.tid] = token
                    self.tokens.append(token)

        def index(self):
            for e in self.entities:
                self.entity_index[e.uid] = e

            self.index_sentences_and_tokens()

            for e in self.entities:
                covered_tokens = []
                for sentence in self.sentences:
                    if e.start >= sentence.start and e.end <= sentence.end:
                        for t in sentence.token_array:
                            if t.start >= e.start and t.end <= e.end:
                                covered_tokens += [t]
                        break
                if len(covered_tokens)==0:  # ERR: annotation does not cover a single full token
    #                print(str(self.name), "ERR: Not covering any token", e, " ".join([str(t.string) for t in self.tokens]))
                    self.errors += 1
                    for t in self.tokens:  # FIX: try to expand it to the single covering token
                        if t.start <= e.start and t.end >= e.end:
                            covered_tokens += [t]
                            self.fixed += 1
    #                        print(str(self.name), "FIXED:", e, " ".join([str(t.string) for t in covered_tokens]))
                            break
                else:
                    if covered_tokens[0].start!=e.start:  # ERR: annotation start doesn't match any token's start
    #                    print(str(self.name), "ERR: no matching token start", e, " ".join([str(t.string) for t in covered_tokens]))
                        self.errors += 1
                        e.start = covered_tokens[0].start  # FIX: expand the annotatio to the left
                        self.fixed += 1
    #                    print(str(self.name), "FIXED:", e, " ".join([str(t.string) for t in covered_tokens]))
                    # ERR: annotation end doesn't match any token's end
                    if covered_tokens[-1 if len(covered_tokens)>1 else 0].end!=e.end:
    #                    print(str(self.name), "ERR: no matching token end", e, " ".join([str(t.string) for t in covered_tokens]))
                        self.errors += 1
                        # FIX: move the annotation to the last token's end offset
                        e.end = covered_tokens[-1 if len(covered_tokens)>1 else 0].end
                        self.fixed += 1
    #                    print(str(self.name), "FIXED:", e, " ".join([str(t.string) for t in covered_tokens]))

                # important: we IGNORE entities that don't cover a token due to a data/tokenization glitch
                if len(covered_tokens) > 0:
                    self.cover_index[e.uid] = covered_tokens
    class ScienceIEBratReader:
        def __init__(self, src_folder):
          self.files = []
          for ann_file in glob.glob(os.path.join(src_folder, '*.ann')):
              txt_file = ann_file[:-4]+".txt"
              self.files += [(ann_file, txt_file)]
         
        def is_relation(self, string):
           return string.startswith("T")

        def parse_entity(self, string, docid):
            try:
              uid, ann, string = string.split("\t")
            except:
              print(string, docid)
              uid, ann = string.split("\t")
              ann = ann.split()
              return Entity(int(ann[1]), int(ann[2]) , ann[0] , utf8ify(''), uid, docid)
            if ";" not in ann:
                etype, start, end = ann.split(" ")
                return Entity(int(start), int(end), etype, utf8ify(string), uid, docid)
            else:
                # Multiwords are covered from first token's start to the last token's end
                spans = ann.split(";")
                etype = spans[0].split(" ")[0]
                start = spans[0].split(" ")[1]
                end = spans[-1].split(" ")[1]
                return Entity(int(start), int(end), etype, utf8ify(string), uid, docid)
          
        
        def read(self):
          documents = []
          for (ann_file, txt_file) in self.files:
              entities = []
              with codecs.open(ann_file, "r", "utf-8") as annf:
                  for line in annf.readlines():
                      line = line.strip()
                      if line:
                          docid = txt_file.split("/")[-1].split(".")[0] # Document Name
                          if self.is_relation(line):
                            entities.append(self.parse_entity(line, docid))
              with codecs.open(txt_file, "r", "utf-8") as f:
                  txt = f.read().strip()
              documents.append(Document(os.path.split(ann_file)[1], entities, txt))
          return documents

    class ConcatMapper:
        def __init__(self, vsm, window=2, sentence_boundaries=True):
            self.vsm = vsm
            self.window = window
            self.sentence_boundaries = sentence_boundaries

        def to_repr(self, entity, document):
            covered_tokens = document.cover_index[entity.uid]

            if self.sentence_boundaries:
                span = covered_tokens[0].sentence
                first_token = (span.token_array[0]).tid
                last_token = (span.token_array[-1]).tid
            else:
                span = Token(document.tokens[0].start, document.tokens[-1].end, "")
                first_token = 0
                last_token = len(document.tokens)-1

            left_min_index = max(first_token, covered_tokens[0].tid - self.window)
            left_max_index = covered_tokens[0].tid
            if left_max_index <= left_min_index:
                context_left = []
            else:
                context_left = document.tokens[left_min_index:left_max_index]

            right_min_index = covered_tokens[-1].tid + 1
            right_max_index = min(last_token, covered_tokens[-1].tid + self.window + 1)
            if right_min_index >= right_max_index:
                context_right = []
            else:
                context_right = document.tokens[right_min_index:right_max_index]

            cl = len(context_left)
            cr = len(context_right)
            K=self.vsm.dim
            context_left = [Token(span.start-1, span.start-1, "#BEGIN_OF_SENTENCE#")] * (self.window-cl) + context_left
            context_right = context_right + [Token(span.end+1, span.end+1, "#END_OF_SENTENCE#")]*(self.window-cr)

            # take average embedding as representation
            covered_emb = np.mean([self.vsm.get(t.string) for t in covered_tokens], axis=0)
            # take concatenated embedding as representation 
            # keep only the first m tokens: improve upon this
            m = 4
            if len(covered_tokens) > m:
            #  # simple heuristic: kick out short words
                for t in covered_tokens:
                    if len(t.string) <= 3:
                        covered_tokens.remove(t)
                        if len(covered_tokens) <= m:
                            break
            #  covered_tokens = filter(lambda x: len(t.string)>3,covered_tokens) 
            my_center = np.concatenate([self.vsm.get(t.string) for t in covered_tokens])
            covered_emb = sequence.pad_sequences([my_center],m*K,truncating="post",dtype="float32")[0] 
            context_left_emb = np.concatenate([self.vsm.get(t.string) for t in context_left])
            context_right_emb = np.concatenate([self.vsm.get(t.string) for t in context_right])
            # check if it is alright
            # print([t.string for t in context_left],[t.string for t in covered_tokens],[t.string for t in context_right])

            return np.concatenate((context_left_emb, covered_emb, context_right_emb), axis=0)

    class SentenceSplitter:
        def __init__(self):
            self.splitter = self.__normal_sentence_splitter()

        def __normal_sentence_splitter(self):
    #        print("initializing default nltk sentence splitter")
            return lambda x: sent_tokenize(x)

        def split(self, text):
            return self.splitter(text)
      
    

    vsm = VSM(embedding_src)
    cs  = 4
    mapper = ConcatMapper(vsm,cs)



    # app.run(host= '10.0.62.211', port='5000')
    app.run(debug=True)
