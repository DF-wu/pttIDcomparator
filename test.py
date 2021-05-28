import json 
import math
import nltk 
import time
from nltk.stem.porter import *
from os import path

nltk.download('punkt')
stemmer = PorterStemmer()

dataset = []
datasetlength = 0 

def readFile(filename):
    global datasetlength
    # author = dict()
    with open(filename,"r",encoding="utf-8") as f:
        doc = json.load(f)
        for file in doc: 
            if file["article_id"] == "":
                continue 
           
            # if file["author"] not in author:
            #     author[file["author"]] = 1
            # else:
            #     author[file["author"]] += 1

            datasetlength += 1  
            dataset.append(file)
    # print(dict((a,b) for a,b in author.items() if b>12))


# def segFile(filename):
#     with open(filename,"r",encoding="utf-8") as f:
#         w = open("gossiping_dataset_seg4000.json","w",encoding="utf-8")
#         doc = json.load(f)
#         t = []
#         for i in range(4000): 
#              t.append(doc[i])
#         json.dump(t,w,ensure_ascii=False)



wordListDict = dict()        
docWordDict = dict()        
normDocWordDict = dict()     

def createDicts():
    start = time.time()
    for data in dataset:
        tokens = nltk.word_tokenize(data["content_seg"])
        curDocDict = dict()  

        for word in tokens: 
            wordIndex = stemmer.stem(word.lower())
            if wordIndex not in curDocDict:
                curDocDict[wordIndex] = 0
            curDocDict[wordIndex] += 1

        docWordDict[data["article_id"]] = curDocDict

        for word in curDocDict:
            if word not in wordListDict:
                wordListDict[word] = []
            if len(wordListDict[word]) == 0 or wordListDict[word][len(wordListDict[word])-1]["article_id"] != data["article_id"]:
                wordListDict[word].append({
                    "article_id":data["article_id"],
                    "tf":curDocDict[word]
                })

        for article_id in docWordDict:
             for word in docWordDict[article_id]:
                tf = docWordDict[article_id][word]
                df = len(wordListDict[word])
                N = len(docWordDict)
                score = (1+ math.log10(tf) * math.log10(N / df))
                docWordDict[article_id][word] = score

        for article_id in docWordDict:
            normDocWordDict[article_id] = dict()
            totalTf = 0
            for word in docWordDict[article_id]:
                totalTf += math.pow(docWordDict[article_id][word],2)
            
            base = math.sqrt(totalTf)        
            for word in docWordDict[article_id]:
                normDocWordDict[article_id][word] = docWordDict[article_id][word]/base
    end = time.time()
    print("Runtime of the create dicts is {:.2f}s".format(end-start))

def cosSim(id1, id2):
    score = 0
    for word in normDocWordDict[id1]:
        if word in normDocWordDict[id2]:
            score += normDocWordDict[id1][word] * normDocWordDict[id2][word]
    return score

def formatOutput(score):
    return "{:.5f}".format(score)


def output(result):
    with open("output.txt","w",encoding="utf-8") as f:        
        length = 10 if len(result)>=10 else len(result)
        for count in range(length):
            json.dump(result[count], f,ensure_ascii=False,indent=2)
        print("輸出 {} 個結果".format(length))


def inputId():
    # 500 hikku
    # 1000 eten
    # 4000 Kay731
    id = input("Input an ID : ")
    idDocDict = []
    for data in dataset:
        if data["author"].split(" ")[0] == id:
            idDocDict.append((data["article_id"],data["content"]))

    count = 0
    res = []
    for docId,content in idDocDict:
        for data in dataset:
            if id != data["author"].split(" ")[0]:
                count += 1 
                res.append(("作者文章 : {} , 比對文章 : {} ".format(docId,data["article_id"])
                            ,"分數 : {} ".format(formatOutput(cosSim(docId,data["article_id"])))
                            ,"原文章內容 : {} ".format(content)
                            ,"比對文章內容 : {} ".format(data["content"])))
 
    output(sorted(res, key=lambda x:x[1],reverse=True))


def saveDicts(fileSlice):
    with open("gossiping_dataset_wordListDict"+fileSlice + ".json", "w",encoding="utf-8") as f:
        json.dump(wordListDict,f,ensure_ascii=False)

    with open("gossiping_dataset_docWordDict"+fileSlice + ".json", "w",encoding="utf-8") as f:
        json.dump(docWordDict,f,ensure_ascii=False)

    with open("gossiping_dataset_normDocWordDict"+fileSlice + ".json", "w",encoding="utf-8") as f:
        json.dump(normDocWordDict,f,ensure_ascii=False)


def loadDicts(fileSlice):
    global wordListDict,docWordDict,normDocWordDict
    with open("gossiping_dataset_wordListDict"+fileSlice + ".json", "r",encoding="utf-8") as f:
        wordListDict =json.load(f)

    with open("gossiping_dataset_docWordDict"+fileSlice + ".json" , "r",encoding="utf-8") as f:
        docWordDict =json.load(f) 

    with open("gossiping_dataset_normDocWordDict"+fileSlice + ".json" , "r",encoding="utf-8") as f:
        normDocWordDict =json.load(f)


def main():     
    fileSlice = "4000"
    readFile("gossiping_dataset_seg" + fileSlice + ".json")
    if (path.exists("gossiping_dataset_wordListDict"+fileSlice+ ".json") and path.exists("gossiping_dataset_docWordDict"+fileSlice+ ".json") and path.exists("gossiping_dataset_normDocWordDict"+fileSlice+ ".json")):
        loadDicts(fileSlice)
    else:
        createDicts()
        saveDicts(fileSlice)
    inputId()

    # segFile("gossiping_dataset_seg.json")

main()