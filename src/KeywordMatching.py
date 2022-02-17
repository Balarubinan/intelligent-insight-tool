import pickle
from src.UtilityClasses import *
infoDict=pickle.load(open('fileTags.txt','rb'))
# print(infoDict)
# print(infoDict["SampleText.txt"]['keys'])
fileInfo=fileInfo()
fileInfo.loadInfo("SampleText.txt",infoDict["SampleText.txt"])
# works as expected!!
print(fileInfo.fileKeywords)
# get search keywords
# sort them out as nouns ans verbs
# finish bigram search
# load model and try to keep it at bay in the server!!
# try out a simple GUI - pySimpleGUI


