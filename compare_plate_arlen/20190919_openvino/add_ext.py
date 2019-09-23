import os

os.chdir("labels_prediction")
filenames = os.listdir(".")
#print(len(filenames))
for filename in filenames:
    print(filename)
    filename2 = filename+".txt"
    os.rename(filename,filename2)
    #quit()
