class fileInfo:
    filename = ""
    fileTypeDict = {}
    fileKeywords = []

    def toInfoDict(self):
        infoDict = {}
        infoDict[self.filename] = {'keys': self.fileKeywords, 'tags': self.fileTypeDict}

    def loadInfo(self, name, info):
        self.fileKeywords = info['keys']
        self.fileTypeDict = info['tags']
        self.filename = name

