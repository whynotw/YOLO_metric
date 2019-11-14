from __future__ import print_function

class PrinterWithSaving():

    def __init__(self,filename):
        self.file = open(filename,"w")

    def prints(self,content):
        print(content)
        print(content, file=self.file)

if __name__ == "__main__":
    # settings
    
    prints = PrinterWithSaving(filename="test123.txt").prints
    
    for i in range(5):
        prints(i)
