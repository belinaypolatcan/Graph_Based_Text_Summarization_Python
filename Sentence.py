
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

class Sentence:
    def __init__(self, sentence):
        self.sentence = sentence
        self.p1 = 0.0
        self.p2 = 0.0
        self.p3 = 0.0
        self.p4 = 0.0
        self.p5 = 0.0
        self.totalScore = 0.0


    def set_p1(self, value):
        self.p1 = value

    def set_p2(self, value):
        self.p2 = value

    def set_p3(self, value):
        self.p3 = value

    def set_p4(self, value):
        self.p4 = value

    def set_p5(self, value):
        self.p5 = value

    def set_totalScore(self,value):
        self.totalScore=value


def findP3(cumleDizisi,G):
     threshold = 0.86
     p3_values = []

     for i, sentence in enumerate(cumleDizisi):
        sayac = 0
        toplamKenar = 0
        for edge in G.edges(sentence.sentence, data=True):
            benzerlikSkoru = edge[2]['weight']
            if benzerlikSkoru > threshold:
                sayac += 1
            toplamKenar += 1
        benzerlikOrani = sayac / toplamKenar if toplamKenar > 0 else 0
        print(f"Cümle {i + 1} için Benzerlik Skoru Eşiği Üzerindeki Kenar Sayısı: {sayac}")
        p3 = sayac / toplamKenar if toplamKenar > 0 else 0
        p3_values.append(p3)

     return p3_values


def findP4(filtreliMetinList,baslik):
  
  p4 = []

  for sentence in filtreliMetinList:

    tokens = nltk.word_tokenize(sentence)

    basliktokens = nltk.word_tokenize(baslik)

    baslikKelimeSayisi = len(basliktokens)
    
    cumleKelimeSayisi = len(tokens)

    Oran = baslikKelimeSayisi / cumleKelimeSayisi if cumleKelimeSayisi > 0 else 0

    p4.append(Oran)
  
  return p4      




def findP5(filtreliMetin,filtreliMetinList):
  

    tfidf = TfidfVectorizer()
  
    X = tfidf.fit_transform(filtreliMetinList)  
  
    tfidf_dict = dict(zip(tfidf.get_feature_names_out(), X.toarray().sum(axis=0)))
 
    tfidf_dict = dict(sorted(tfidf_dict.items(), key=lambda x: x[1], reverse=True))


    tfidf_df = pd.DataFrame.from_dict(tfidf_dict, orient='index', columns=['tfidf'])
 
    tfidf_df.index = tfidf_df.index.rename('token')
  
    filtreKelime = filtreliMetin.split()

    toplamkelime = len(filtreKelime)

    threshold = int(toplamkelime * 0.1)

    top_tokens = tfidf_df.head(threshold).index.tolist()

    selected_tokens_df = tfidf_df.loc[top_tokens].copy()

    print(selected_tokens_df)

    p5 = []

    for sentence in filtreliMetinList:

        tokens = word_tokenize(sentence)

        cumleKelimeSayisi = len(tokens)
        ortakKelimeSayisi = len(set(tokens).intersection(set(selected_tokens_df.index.tolist())))
        ortakKelimeOrani = ortakKelimeSayisi / cumleKelimeSayisi if cumleKelimeSayisi > 0 else 0

        p5.append(ortakKelimeOrani)

    return p5




def findSentenceScores(sentences,p1,p2,p3,p4,p5):

    cumleSkorlari = []

    for i in range(len(sentences)):
        score = (p1[i] + p2[i] + 3*p3[i] + 2* p4[i] + 5*p5[i]) / 5
        cumleSkorlari.append(score)

    return cumleSkorlari

 
    