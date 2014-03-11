
def posmax(seq, key=lambda x:x): 
    return max(enumerate(seq), key=lambda k: key(k[1]))[0] 

class SymbolTable:

    def __init__(self):
        self.items = []
        self.ids = {}

    def size(self):
        return len(self.items)

    '''
    add item to symbol table if it doesnt exist and return index
    '''
    def addID(self,item):
        if item in self.ids:
            return self.ids[item]
        else:
            idx = len(self.items)
            self.ids[item] = idx
            self.items.append(item)
            return idx

    '''
    get the index for an item, will fail if not present
    '''
    def getID(self,item):
        return self.items[item]

    
    def getItem(self,id):
        return items[id]
