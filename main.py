import nltk
from Sentence import findP3,findP4,findP5,findSentenceScores
from Sentence import Sentence
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
import string
import tkinter as tk
from tkinter import filedialog
from transformers import BertTokenizer, BertModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import matplotlib.pyplot as plt
from nltk import ne_chunk
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from rouge import Rouge


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('maxent_ne_chunker')
nltk.download('words')

G = nx.Graph()

stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer('english')


root = tk.Tk()
root.withdraw()


file_path = filedialog.askopenfilename()


with open(file_path, "r", encoding="utf-8") as f:
    lines = f.readlines()


baslik = lines[0].strip()
text = "".join(lines[1:])


sentences = sent_tokenize(text)
ozelisimler = []
numericVeriler = []


ozetCumleSayisi= int(len(sentences) / 2)


cumleDizisi=[]

for sentence in sentences:
    object = Sentence(sentence)
    cumleDizisi.append(object)


for cumle in cumleDizisi:
   G.add_node(cumle.sentence)

for sentence in sentences:
    sayilar = re.findall(r'\b\d+(?:st|nd|rd|th)?\b', sentence)
    numericVeriler.extend(sayilar)
    tokens = word_tokenize(sentence)
    tagged_tokens = nltk.pos_tag(tokens)
    chunked_tokens = ne_chunk(tagged_tokens, binary=True)
    for subtree in chunked_tokens.subtrees():
        if subtree.label() == 'NE':
            named_entity = ' '.join([token[0] for token in subtree.leaves()])
            ozelisimler.append(named_entity)

filtreliMetinList = []
ozel_isimSkorlari = [] 
numericVeriSkorlari = []  

for sentence in sentences:
    
    sayilar = re.findall(r'\b\d+(?:st|nd|rd|th)?\b', sentence)
    numericVeriler.extend(sayilar)
 
    tokens = word_tokenize(sentence)

    filtered_tokens = [token for token in tokens if token.lower() not in stop_words and token not in string.punctuation]

    filtered_tokens = [token.replace("'s", "") for token in filtered_tokens]

    stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]

    filtreliCumle = ' '.join(stemmed_tokens)
    filtreliMetinList.append(filtreliCumle)


    ozelisimadet = len([token for token in tagged_tokens if token[1] == 'NNP'])
    sentence_length = len(tokens)
    ozelisimoran = ozelisimadet / sentence_length
    ozel_isimSkorlari.append(ozelisimoran)


    numericveriadet = len(re.findall(r'\b\d+(?:st|nd|rd|th)?\b', sentence))
    sentence_length = len(word_tokenize(sentence))
    numericVeriOran = numericveriadet / sentence_length
    numericVeriSkorlari.append(numericVeriOran)


filtreliMetin = ' '.join(filtreliMetinList)


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

sentence_embeddings = []

for sentence in filtreliMetinList:
    tokens = tokenizer.tokenize(sentence)
    tokens = ['[CLS]'] + tokens + ['[SEP]']
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = torch.tensor(input_ids).unsqueeze(0)  

    with torch.no_grad():
        outputs = model(input_ids)
        last_hidden_states = outputs.last_hidden_state

    sentence_embedding = last_hidden_states[0][0].numpy()  
    sentence_embeddings.append(sentence_embedding)

sentence_embeddings = np.array(sentence_embeddings)

benzerlikMatrisi = cosine_similarity(sentence_embeddings)


thresholdsayac=0.0
kenarSayisi=0

for i in range(len(cumleDizisi)):
    for j in range(i + 1, len(cumleDizisi)):
        benzerlikSkoru = benzerlikMatrisi[i, j]
        sentence_i=cumleDizisi[i].sentence
        sentence_j=cumleDizisi[j].sentence
        G.add_edge(sentence_i, sentence_j, weight=benzerlikSkoru)
        thresholdsayac+=benzerlikSkoru
        kenarSayisi+=1
        
kosinusthresold=thresholdsayac/kenarSayisi



p1=ozel_isimSkorlari

p2=numericVeriSkorlari

p3 = findP3(cumleDizisi,G)

p4=findP4(filtreliMetinList,baslik)

p5 = findP5(filtreliMetin,filtreliMetinList)

cumleSkorlari = findSentenceScores(sentences,p1,p2,p3,p4,p5)




for i, p1_value in enumerate(p1):
   sentence=cumleDizisi[i]
   sentence.set_p1(p1_value)
   sentence.set_p2(p2[i])
   sentence.set_p3(p3[i])
   sentence.set_p4(p4[i])
   sentence.set_p5(p5[i])
   sentence.set_totalScore(cumleSkorlari[i])



siraliDizi = [sentence for _, sentence in sorted(zip(cumleSkorlari, sentences), reverse=True)]



guncelCumleler = []
for sentence in siraliDizi:
    sentence = sentence.replace("\n", "")
    if not sentence.strip():
        continue
    guncelCumleler.append(sentence)



ozetCumle = ' '.join(guncelCumleler[:ozetCumleSayisi])

rouge = Rouge()

skor=rouge.get_scores(ozetCumle,text)

rouge_1 = skor[0]['rouge-1']['f']
rouge_2 = skor[0]['rouge-2']['f']
rouge_l = skor[0]['rouge-l']['f']


print("ROUGE-1 Score:", rouge_1)
print("ROUGE-2 Score:", rouge_2)
print("ROUGE-L Score:", rouge_l)


# Grafı çizme
pos = nx.spring_layout(G)

plt.figure(figsize=(8, 8))
plt.axis('off')


for i, node in enumerate(G.nodes):
    total_score = cumleDizisi[i].totalScore
    text_label = f"Sentence {i+1}"
    score_label = f"{total_score:.4f}"
    bbox_props = dict(boxstyle="square,pad=0.3", fc="white", ec="black")
    
  
    plt.text(pos[node][0], pos[node][1] + 0.05, text_label, ha='center', va='center', fontsize=8, bbox=bbox_props)
    
    score_bbox_props = dict(boxstyle="square,pad=0.3", fc="red", ec="black")
    plt.text(pos[node][0], pos[node][1], score_label, ha='center', va='center', fontsize=8, bbox=score_bbox_props)

    kenarSayac = sum(1 for _, _, edge_data in G.edges(node, data=True) if edge_data['weight'] > kosinusthresold)
    
    edge_label = f"{kenarSayac}"
    edge_bbox_props = dict(boxstyle="square,pad=0.3", fc="yellow", ec="black")
    plt.text(pos[node][0], pos[node][1] - 0.05, edge_label, ha='center', va='center', fontsize=8, bbox=edge_bbox_props)



for node in G.nodes:
    neighbors = list(G.neighbors(node))
    for neighbor in neighbors:
        benzerlikSkoru = G.edges[node, neighbor]['weight']
        label = f"{benzerlikSkoru:.6f}"
        x = (pos[node][0] + pos[neighbor][0]) / 2
        y = (pos[node][1] + pos[neighbor][1]) / 2
        plt.text(x, y, label, ha='center', va='center', fontsize=8, color='red')


nx.draw_networkx_edges(G, pos, edge_color='gray')

plt.title("GRAPH")
plt.show()



root = tk.Tk()
root.title("Özet Cümle ve ROUGE Skorları")
root.geometry("400x400")


text_box = tk.Text(root, wrap="word")
text_box.pack(fill="both", expand=True)

text_box.insert("1.0", ozetCumle)


rouge1_label = tk.Label(root, text="ROUGE-1 Skor:")
rouge1_label.pack()

rouge1_score_label = tk.Label(root, text=rouge_1)
rouge1_score_label.pack()


rouge2_label = tk.Label(root, text="ROUGE-2 Skor:")
rouge2_label.pack()

rouge2_score_label = tk.Label(root, text=rouge_2)
rouge2_score_label.pack()

rougel_label = tk.Label(root, text="ROUGE-L Skor:")
rougel_label.pack()

rougel_score_label = tk.Label(root, text=rouge_l)
rougel_score_label.pack()

root.mainloop()
