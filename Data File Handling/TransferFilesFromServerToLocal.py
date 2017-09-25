
# coding: utf-8

# ### Import libs

# In[9]:

import os
import shutil

from IPython.display import Markdown, display
def printmd(string):
    display(Markdown(string))


# ### Set the source folder (make sure you're connected to SOD, if this is where the folder is located)

# In[2]:

rootDir = 'T:/ACC/Tayfuntumkaya/DATA/WALi2.0/AsgharScreen/20170417'


# ### Set the destination folder

# In[3]:

destDir = 'C:/Users/tumkayat/Desktop/CodeRep/WALiSAR/BehaviroalDataAnalyses'


# In[12]:

## Make a new local folder in the destination directory with a name same as the root directory has (i.e 20170417)
newFolderName = destDir + '/' + rootDir.strip().split('/')[-1]
if not os.path.exists(newFolderName):
    os.makedirs(newFolderName)

## Crawl thru the root directory
for path, dirs, files in os.walk(rootDir):
    ## loop thru the files
    for f in files:
    ## find the tdms files
        if f[-5:] == '.tdms':
            tdmsFileName = f[:-5]
    ## get the genotype and experimental conditions from the path of this tdms file
            pathInfo = path.strip()
            pathInfo = pathInfo.split('\\')  
            genotype = pathInfo[-2]
            experimentalConditions = pathInfo[-1][3:]
            
    ## generate a unique file name 'tdms name + gentoype + exp conditions'
            uniqueFileName = tdmsFileName + '_' + genotype + '_' + experimentalConditions
        
    ## get the full name and path of the tdms and corresponding pattern csv files for copying purposes   
            oldPathTdmsFile = path + '/' + f
            oldPathPattern01 = path + '/' + 'Pattern 01.csv'
            oldPathPattern10 = path + '/' + 'Pattern 10.csv'
            
    ## prepare new names for tdms file and corresponding Pattern csv files 
            newNameTdmsFile = newFolderName + '/' + uniqueFileName + '.tdms'
            newNamePattern01 = newFolderName + '/' + uniqueFileName + '_Pattern01.csv'
            newNamePattern10 = newFolderName + '/' + uniqueFileName + '_Pattern10.csv'
    
    ## print progress in the terminal
            printmd("**Copying** %s **to** %s" %(oldPathTdmsFile, newNameTdmsFile))
        
    ## copy and rename the files
            shutil.copy2(oldPathTdmsFile,newNameTdmsFile)
            shutil.copy2(oldPathPattern01,newNamePattern01)
            shutil.copy2(oldPathPattern10,newNamePattern10)


# In[ ]:



