import math
from collections import Counter

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
from nltk.corpus import stopwords
import nltk
# nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, LancasterStemmer
from nltk.stem import PorterStemmer, LancasterStemmer
from src.UtilityClasses import *

fileInfoArr=[]
file = "AISampleText.txt"
files=["SampleText.txt","DNSampleText.txt","Area51SampleText.txt","AISampleText.txt"]
sampleDir="SampleTexts"
# with open(filename, 'r', encoding="utf8") as f:
#     farr = f.readlines()
#     totalWordCount = sum([len(x.split(' ')) for x in farr])
#     print("total word count: ", totalWordCount)


def IDF(corpus, unique_words):
    idf_dict = {}
    N = len(corpus)
    for i in unique_words:
        count = 0
        for sen in corpus:
            if i in sen.split():
                count = count + 1
            idf_dict[i] = (math.log((1 + N) / (count + 1))) + 1
    return idf_dict


def fit(whole_data):
    unique_words = set()
    if isinstance(whole_data, (list,)):
        for x in whole_data:
            for y in x.split():
                if len(y) < 2:
                    continue
                unique_words.add(y)
        unique_words = sorted(list(unique_words))
        vocab = {j: i for i, j in enumerate(unique_words)}
        #       print(vocab)
        Idf_values_of_all_unique_words = IDF(whole_data, unique_words)
    #       print(Idf_values_of_all_unique_words)
    return vocab, Idf_values_of_all_unique_words

for filename in files:
    newInfo=fileInfo()
    newInfo.filename=filename

    print("Processing File "+filename)
    f = open(sampleDir+"/"+filename, 'r', encoding="utf8")
    arr = f.readlines()
    arr = [''.join(arr)]
    totalWordCount = sum([len(x.split(' ')) for x in arr])
    print("total word count: ", totalWordCount)
    print(len(arr))
    print(arr)

    # lower case conversion
    PrcsLines = []
    for x in arr:
        x = x.lower()
        #     print(len(x))
        symbols = "!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n,1234567890"
        newstr = ""
        for letter in x:
            if letter not in symbols:
                newstr += letter
        PrcsLines.append(newstr)

    # print(PrcsLines[0])

    corpus = PrcsLines
    # print("Original"+"\n"+corpus[0]+"*"*100+"count :"+str(len(corpus[0].split(' '))))

    # this variable will hold all tokenized text with stopwords to use for Noun verb idetification
    saveTokens = []

    # stop word removal
    for sentInd in range(len(corpus)):
        print(sentInd)
        para = corpus[sentInd]
        text_tokens = word_tokenize(para)
        saveTokens.append(text_tokens)
        tokensWsw = [word for word in text_tokens if not word in stopwords.words('english')]
        corpus[sentInd] = ' '.join(tokensWsw)
    #     print("Processed"+"\n"+para+"*"*100+"count :"+str(len(para.split(' '))))

    Vocabulary, idf_of_vocabulary = fit(corpus)
    print("Vocabulary Fit done ")
    # print(stopwords.words())
    # print(stopwords.words('english'))

    # print(list(Vocabulary.keys()))
    print("key word count: ", len(Vocabulary.keys()))
    print("Compression rate", len(Vocabulary.keys()) / totalWordCount)
    # 687 words reduced to 50! woah...good results already!!

    # lemmatazation

    from nltk.stem import WordNetLemmatizer

    lemmatizer = WordNetLemmatizer()
    newSet = []
    nonNewSet = []
    for x in Vocabulary.keys():
        nonNewSet.append(x)
        x = lemmatizer.lemmatize(x)
        if len(x) > 4:
            newSet.append(x)
    print("After lammentization :", newSet)
    print("count :", len(newSet))
    print(Vocabulary.keys() - nonNewSet)

    # stemming

    ps = PorterStemmer()
    ls = LancasterStemmer()
    stemSet = {}
    for w in newSet:
        #     print(w, " : ", ps.stem(w))
        stemSet[ps.stem(w)] = 1

    # print("After Stemming : ",stemSet)
    print("Count : ", len(stemSet))
    stemSet = {}

    for w in newSet:
        #     print(w, " : ", ps.stem(w))
        stemSet[ls.stem(w)] = 1
    # print("After Stemming : ",stemSet)
    print("Count : ", len(stemSet))

    print(idf_of_vocabulary.keys())

    # Vocabulary dict has to switched with a dict that has preprocessed keys...
    # do preprocessing sepratley instead of with lammentization!
    print(len(Vocabulary))
    Vocabulary = {x: Vocabulary[x] for x in nonNewSet}
    print(len(Vocabulary))


    def transform(dataset, vocabulary, idf_values):
        sparse_matrix = csr_matrix((len(dataset), len(vocabulary)), dtype=np.float64)
        for row in range(0, len(dataset)):
            number_of_words_in_sentence = Counter(dataset[row].split())
            for word in dataset[row].split():
                if word in list(vocabulary.keys()):
                    tf_idf_value = (number_of_words_in_sentence[word] / len(dataset[row].split())) * (idf_values[word])
                    sparse_matrix[row, vocabulary[word]] = tf_idf_value
        print("NORM FORM\n", normalize(sparse_matrix, norm='l2', axis=1, copy=True, return_norm=False))
        output = normalize(sparse_matrix, norm='l2', axis=1, copy=True, return_norm=False)
        return output


    final_output = transform(corpus, Vocabulary, idf_of_vocabulary)
    print(final_output.shape)

    # print(final_output.data)

    print(len(final_output.data))

    # enumerating TFIDF value to Vocab words
    # this zipping loops assumes final_output.data values correspond to order in Vocabulary.keys()
    # BUT DOES IT???
    newArr = []
    for x, y in zip(list(final_output.data), list(Vocabulary.keys())):
        newArr.append((x, y))
    newArr.sort()

    s = set(newArr)
    print(len(newArr))

    resArr = filter(lambda x: len(x[1]) > 3, newArr[::-1][:])
    resAsString = '\n'.join([str(x[1]) + str(x[0]) for x in resArr])
    # print("top 50 words : \n"+resAsString)
    print(len(newArr))
    # this result is accurate to a satisfactory level!

    # lammentization
    finalisedWords = [x[1] for x in newArr[::-1][:]]
    # save to fileInfo object
    newInfo.fileKeywords=finalisedWords

    from nltk.stem import WordNetLemmatizer

    lemmatizer = WordNetLemmatizer()
    newSet = []
    for x in finalisedWords:
        x = lemmatizer.lemmatize(x)
        if len(x) > 4:
            newSet.append(x)
    print("After lammentization :", newSet)
    print("count :", len(newSet))
    finalisedWords = (newSet)
    # 800-> 600 words

    # stemming

    ps = PorterStemmer()
    ls = LancasterStemmer()
    stemSet = {}
    for w in newSet:
        #     print(w, " : ", ps.stem(w))
        stemSet[ps.stem(w)] = 1
    print("After Stemming : ", stemSet)
    print("Count : ", len(stemSet))
    stemSet = {}
    for w in newSet:
        #     print(w, " : ", ps.stem(w))
        stemSet[ls.stem(w)] = 1
    print("After Stemming : ", stemSet)
    print("Count : ", len(stemSet))

    stemmedFinalWords = stemSet.keys()
    print(stemmedFinalWords)
    print(len(stemmedFinalWords))

    print(finalisedWords)
    print("Final Tag word count: ", len(finalisedWords))

    # manual count Check
    for x in finalisedWords:
        cnt = 0
        print(len(x), end=" ")
        for sentence in PrcsLines[0].split():
            if sentence.find(x) >= 0:
                cnt += 1
        #             print(x)
        #             print(sentence.find(x))
        #             print(sentence)
        print("count ", x, ":", cnt)
    # try sorting these words according to some parameter lineary dependent on the frequency
    # maybe (( TFIDF value )* freq/100)??
    print()

    # use this list to check important keywords
    checkWords = ["area 51", "consipiracy", "alien", "military", 'installation']
    for x in checkWords:
        try:
            print(x, "found @", finalisedWords.index(x))
        except:
            print(x, "not found")

    # removing verbs
    nouns = nltk.pos_tag(saveTokens[0])
    # # print(saveTokens)
    # for tknSent in saveTokens:
    # #     print(tknSent)
    #     if nltk.pos_tag(tknSent)=="NN":
    #         nouns.append(tknSent)
    #         print("ff")
    #     else:
    #         print(nltk.pos_tag(tknSent))
    #         pass

    typeDict = {x: y for x, y in nouns}
    print(typeDict)
    newInfo.fileTypeDict=typeDict
    print("finished File " + filename)
    fileInfoArr.append(newInfo)


infoDict={}
for info in fileInfoArr:
    print("File Name",info.filename)
    print("File Keywords",info.fileKeywords)
    print("File TagDict",info.fileTypeDict)
    infoDict[info.filename]={'keys':info.fileKeywords,'tags':info.fileTypeDict}

import pickle
outFile=open('fileTags.txt','wb')
pickle.dump(infoDict,outFile)
outFile.close()



# use this text to filter out all the verbs in the finalisedwords

# Observations:
# this algo considers each paragraph as a single separate document
# as a default all the important words as a whole for the document as thrown away as unique word for every
# the result words are all words that occur only once in the damn file !!
# basically one word inside a file will be super unique then it would be the top word

# the same file was modified to go through the document as a whole
# output was satisfactory and relevant
# it omits one word noun like 'L' as in L lawliet
# it produces all important verbs also( som of which wont be searched as keyword for sure)
# need to implement noun seperation and word to word correlation
# nouns only need to be in the keywords and verbs that are strongly realted to them can be then found and added manually
# 2 word nouns are also needed to be found ('light yagami')
# alternative is to perform the same algo for a modified arr with 2 words for as combined word with a ' ' in between

