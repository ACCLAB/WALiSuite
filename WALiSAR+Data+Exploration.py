
# coding: utf-8

# ## Import libs

# In[676]:

import os
import seaborn as sns
import pandas as pd
from scipy import stats
import scipy as sp
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
from matplotlib import gridspec
from itertools import groupby
from operator import itemgetter
import matplotlib as mpl
import bootstrap_contrast as bs
from nptdms import *
pd.set_option('display.max_colwidth', -1)
pd.set_option('display.max_rows', 1)
mpl.rcParams['pdf.fonttype'] = 42


# ## Read tdms files

# In[256]:

# TdmsFile > Groups > Channels > Data 
f = TdmsFile("LOG_2017-04-18_16-22-19.tdms")


# ## Inspect the file

# In[257]:

## Print group names
groupNames = f.groups()
print 'Group Names:', groupNames


# In[258]:

## Print channel names of a particular group
channelNames = f.group_channels('Tracker')
print 'Trackers channels:', channelNames


# In[259]:

for group in groupNames:
    channelNames = f.group_channels(group)
    for item in channelNames:
        print 'Group Name/Channel Name:',item


# ## Access to the data alternative 1

# In[15]:

## Get the data from a channel
Genotype = f.channel_data('ExperimentInfo','Genotype')
Tracking = f.channel_data('Tracker','Untitled 1')

print 'Genotype:', Genotype
print 'Tracker(cXmm001):', Tracking
print 'Length of the cXmm data:', len(Tracking)


# ## Access to the data alternative 2

# In[260]:

## Get the data from a channel 2
channel = f.object('Tracker','Speed_Av_mm_per_sec001')
data = channel.data
#time = channel.time_track()


# In[264]:

channel1 = f.object('Count','PatternState')
data1 = channel1.data  


# In[276]:

channel1 = f.object('Count','Timestamp')
data1 = channel1.data 


# ## Transfer the data into a Pandas df

# In[403]:

## Loading data into a Pandas Df
df = f.as_dataframe()
#df
#df.to_csv('tdmsfile2.csv')


# ## Get the light ON/OFF information

# In[577]:

metaData = f.object().properties


# In[400]:

data1 = df["/\'Count\'/\'Obj1_cX\'"]


# In[384]:

data3 = df["/'Tracker'/'HeadX_pix001'"]


# In[401]:

#data1.isnull().sum()
data1


# In[380]:

t = "/'Tracker'/'HeadX_pix001'"
"/'Tracker'/'HeadX_pix001"


# In[381]:

data2 = df[t]


# In[360]:

patterns = df["/\'Count\'/\'PatternState'"]


# In[371]:

pat01 = patterns[patterns == 'Pattern 01'].index
pat10 = patterns[patterns == 'Pattern 10'].index

p01_min = min(pat01)
p01_max = max(pat01)
p10_min = min(pat10)
p10_max = max(pat10)


# In[433]:

p01_min


# In[379]:

pat01_first_light_exposure = min(df["/\'Count\'/\'Obj1_InLight'"][p01_min:p01_max][df["/\'Count\'/\'Obj1_InLight'"] == '1'].index) if not df["/\'Count\'/\'Obj1_InLight'"][p01_min:p01_max][df["/\'Count\'/\'Obj1_InLight'"] == '1'].empty else 0
pat10_first_light_exposure = min(df["/\'Count\'/\'Obj1_InLight'"][p10_min:p10_max][df["/\'Count\'/\'Obj1_InLight'"] == '1'].index) if not df["/\'Count\'/\'Obj1_InLight'"][p10_min:p10_max][df["/\'Count\'/\'Obj1_InLight'"] == '1'].empty else 0


# In[598]:

df


# In[382]:

fig1 = plt.figure()
ax1 = plt.subplot(111)

sns.set(style="ticks", palette="bright", color_codes=True)
sns.despine()
ax1.set_ylabel('Diff')
ax1.set_xlabel('Time')

ax1.plot(range(len(data2)), data2)
ax1.axvspan(p10_min,p10_max,color='red',alpha=0.3)
ax1.axvspan(p01_min,p01_max,color='red',alpha=0.3)
ax1.axvline(pat01_first_light_exposure,color='black')
ax1.axvline(pat10_first_light_exposure,color='black')
#ax1.plot(range(len(data2)), data2,color='red')
#plt.savefig('HeadX_vs_Cx.pdf',dpi=1000,bbox_inches='tight')
plt.show()


# In[392]:

diff = [abs(float(x) - float(y)) for x, y in zip(data1, data3)]


# In[599]:

df_pat01 = df[df["/\'Count\'/\'PatternState'"] == 'Pattern 01']    
df_pat10 = df[df["/\'Count\'/\'PatternState'"] == 'Pattern 10'] 


# In[605]:

a = min(df_pat01.index),max(df_pat01.index)


# In[515]:


##Get the chunks where the light is ON   
df_pat01 = df[df["/\'Count\'/\'PatternState'"] == 'Pattern 01']    
df_pat10 = df[df["/\'Count\'/\'PatternState'"] == 'Pattern 10'] 

##Check number of flies in the df
num_of_flies = sum(df.columns.str.contains("/'Tracker'/'HeadX_pix"))

temp = {'FlyID':[],'Pattern 01 First light contact index':[],'Pattern 10 First light contact index':[]}
for fly in range(1,num_of_flies+1):
## format the fly index into 3 digits number,i.e '5' >> '005' 
flyID = format(str(fly).zfill(3))

## generate IDs for the data need to be accessed from the df
fly_headX_pix_ID = "/'Tracker'/'HeadX_pix" + str(flyID) + "'"
fly_inLight_ID = "/\'Count\'/\'Obj%s_InLight'" % fly

## find the index where the fly first contacted with light in each pattern
pat01_first_light_contact = df_pat01.index[df_pat01[fly_inLight_ID] == '1'][0] if not df_pat01.index[df_pat01[fly_inLight_ID] == '1'].empty else None 
pat10_first_light_contact = df_pat10.index[df_pat10[fly_inLight_ID] == '1'][0] if not df_pat10.index[df_pat10[fly_inLight_ID] == '1'].empty else None 

temp['FlyID'].append(flyID)
temp['Pattern 01 First light contact index'].append(pat01_first_light_contact)
temp['Pattern 10 First light contact index'].append(pat10_first_light_contact)

results = pd.DataFrame(temp)


# ## Read the tdms and pattern border data into a df

# In[607]:

fileName = "LOG_2017-04-18_16-22-19.tdms"


# In[694]:

## Generate a single dataframe from the .tdms and pattern files 
temp = {'Tdms file name':[],'Date':[],'Time':[],'Light type':[],'Light Intensity(uW/mm2)':[],'Wind status':[],
        'Satiety':[],'Genotype':[],'Status':[],'Fly ID':[],'cX(pix)':[],'HeadX(pix)':[],'HeadY(pix)':[],
        'InLight':[],'First light contact index|P01':[],'First light contact index|P10':[],'LightON index|P01':[],
        'LightON index|P10':[],'Border|P01':[],'Border|P10':[]}

## Open the tdms file
f = TdmsFile(fileName) 

## Get exp info from the tdms filename
fname = fileName.split('_')
date = fname[1]
time = fname[2][:-5]
genotype = 'dummy'
intensity = 'dummy'
lightType = 'Constant'
windState = 'dummy'
satiety = 'dummy'

## Get status info
if ('w1118' in genotype) | ('W1118' in genotype):
    status = 'Parent'
elif (('Gal4' in genotype) | ('GAL4' in genotype)) & ('UAS' in genotype):
    status = 'Offspring'
else:
    status = 'Unknown: check your genotype names'

## Load the tdms into a pandas df
TDMSdf = f.as_dataframe()

## Get number of flies in the exp
num_of_flies = sum(TDMSdf.columns.str.contains("/'Tracker'/'HeadX_pix"))

for fly in range(1,num_of_flies+1):
    ## format the fly index into 3 digits number,i.e '5' >> '005' 
    flyID = format(str(fly).zfill(3))
    
    ## generate column names for the data need to be pulled from the df
    fly_cX_pix_ID = "/\'Count\'/\'Obj%s_cX'" % fly 
    fly_inLight_ID = "/\'Count\'/\'Obj%s_InLight'" % fly
    fly_headX_pix_ID = "/'Tracker'/'HeadX_pix" + str(flyID) + "'"
    fly_headY_pix_ID = "/'Tracker'/'HeadY_pix" + str(flyID) + "'"
   
    temp['Fly ID'].append(flyID)
    temp['cX(pix)'].append(TDMSdf[fly_cX_pix_ID].values.astype(float))
    temp['InLight'].append(TDMSdf[fly_inLight_ID].values.astype(float))
    temp['HeadX(pix)'].append(TDMSdf[fly_headX_pix_ID].values.astype(float))
    temp['HeadY(pix)'].append(TDMSdf[fly_headY_pix_ID].values.astype(float))
    
## Get the chunks where the light was ON   
TDMSdf_pat01 = TDMSdf[TDMSdf["/\'Count\'/\'PatternState'"] == 'Pattern 01']    
TDMSdf_pat10 = TDMSdf[TDMSdf["/\'Count\'/\'PatternState'"] == 'Pattern 10'] 

LightOnP01 = min(TDMSdf_pat01.index),max(TDMSdf_pat01.index)
LightOnP10 = min(TDMSdf_pat10.index),max(TDMSdf_pat10.index)

## Open the pattern csv files to extract light border info per fly
P01_df = pd.read_csv('Pattern 01.csv')
P10_df = pd.read_csv('Pattern 10.csv')

for fly in range(1,num_of_flies+1):
    flyID = format(str(fly).zfill(3))
    
    ## generate column names for the data need to be pulled from the df
    fly_inLight_ID = "/\'Count\'/\'Obj%s_InLight'" % fly
    
    ## find the index where the fly first contacted with light in each pattern
    P01_first_light_contact = TDMSdf_pat01.index[TDMSdf_pat01[fly_inLight_ID] == '1'][0] if not TDMSdf_pat01.index[TDMSdf_pat01[fly_inLight_ID] == '1'].empty else None 
    P10_first_light_contact = TDMSdf_pat10.index[TDMSdf_pat10[fly_inLight_ID] == '1'][0] if not TDMSdf_pat10.index[TDMSdf_pat10[fly_inLight_ID] == '1'].empty else None 
    
    temp['First light contact index|P01'].append(P01_first_light_contact)
    temp['First light contact index|P10'].append(P10_first_light_contact)

    ## Append the info to temp dict
    temp['Tdms file name'].append(fileName)
    temp['Date'].append(date)
    temp['Time'].append(time)
    temp['Light type'].append(lightType)
    temp['Light Intensity(uW/mm2)'].append(intensity)
    temp['Wind status'].append(windState)
    temp['Satiety'].append(satiety)
    temp['Genotype'].append(genotype)
    temp['Status'].append(status)
    temp['LightON index|P01'].append(LightOnP01)
    temp['LightON index|P10'].append(LightOnP10)
    temp['Border|P01'].append(P01_df.filter(regex='pix').iloc[1].values[fly-1])
    temp['Border|P10'].append(P10_df.filter(regex='pix').iloc[1].values[fly-1])

colOrder = ['Tdms file name','Date','Time','Light type','Light Intensity(uW/mm2)','Wind status',
        'Satiety','Genotype','Status','Fly ID','cX(pix)','HeadX(pix)','HeadY(pix)',
        'InLight','First light contact index|P01','First light contact index|P10','LightON index|P01',
        'LightON index|P10','Border|P01','Border|P10']

## Convert temp into a df
results = pd.DataFrame(temp,columns=colOrder)
#results.to_csv('TrialDF.csv',index=False)


# ## Analyze the df

# ### Percentage time spent after light contact

# In[696]:

def PercentageTimeSpentAfterLightContact(df):
    
    
    return


# ### Preference index in the choice zone

# In[697]:

def PreferenceIndexinTheChoiceZone(df):
    
    
    return


# ### Fractional time in the odorized zone

# In[698]:

def FractionalTimeinTheOdorizedZone(df):
    
    
    return


# In[574]:

## Plotting first time contacts of the flies in a given experiment

## Number of flies that never exposed to the light

no_contact_w_light_pat01 = sum(results['Pattern 01 First light contact index'].isnull())
no_contact_w_light_pat10 =sum(results['Pattern 10 First light contact index'].isnull())

## Light start-end indices
pat01_start = df_pat01.index[0]
pat01_end = df_pat01.index[-1]
pat10_start = df_pat10.index[0]
pat10_end = df_pat10.index[-1]

fig1 = plt.figure()
ax1 = plt.subplot(131)
ax2 = plt.subplot(133)

sns.set(style="ticks", palette="bright", color_codes=True)
sns.despine()
#ax1.set_ylabel('First light contact')
#ax1.set_xlabel('Time')

sns.swarmplot(y = 'Pattern 01 First light contact index', data=results,ax = ax1)
sns.swarmplot(y = 'Pattern 10 First light contact index', data=results,ax = ax2)
#ax1.boxplot(results['Pattern 01 First light contact index'])
#ax1.axvspan(p10_min,p10_max,color='red',alpha=0.3)
#ax1.axvspan(p01_min,p01_max,color='red',alpha=0.3)
ax1.axhline(pat01_start,color='red')
ax1.axhline(pat01_end,color='red')
ax2.axhline(pat10_start,color='red')
ax2.axhline(pat10_end,color='red')
ax1.set_title('Pattern 01')
ax2.set_title('Pattern 10')
#ax1.plot(range(len(data2)), data2,color='red')
#plt.savefig('HeadX_vs_Cx.pdf',dpi=1000,bbox_inches='tight')
plt.show()
print 'Never seen the light (Pat01) =',no_contact_w_light_pat01, 'Never seen the light (Pat10) =',no_contact_w_light_pat10 


# In[575]:

results


# ### Set your data folder 

# In[40]:

folder = "C:/Users/tumkayat/Desktop/ORScreening/OSAR/Orco-ACR1-Male-starved/"
os.chdir(folder)


# ### Can check how many .csv files is located under the folder

# In[41]:

dataFiles = os.listdir(folder)
len(dataFiles)


# ### Set how many last seconds you want to use for the analysis

# In[42]:

lastXseconds = 30


# ### Run this cell to do the analysis

# In[43]:

def calculatePI(data):
    numofTimePoints = len(data)
    totalTimeinLight = sum(data)
    totalTimeinDark = numofTimePoints - totalTimeinLight
    
    PI = float(totalTimeinLight - totalTimeinDark)/float(numofTimePoints)
    return PI

def splitDataintoEpochChunks(df):
    return np.split(lightON, np.where(np.diff(lightON.index.second) > 1)[0]+1)

temp = {'FileName':[], 'Pattern':[], 'FileName_pattern':[],'Date':[], 'StartTime':[], 'Genotype': [],'Status':[],'Status_Intensity': [], 'Intensity':[], 'LightType':[],
        'LightType_Intensity':[] ,'Intensity_Wind': [],'Wind': [],'LightType_Intensity_Wind':[],'Status_LightType_Intensity_Wind': [], 'SingleFlyPI':[], 'Half PI': []}


for f in dataFiles:
    if f[-4:] == '.csv':
        print f
        expInfo = f.split('_')
        #print expInfo
        date = expInfo[1]
        #startTime = expInfo[-5]
        #genotype = expInfo[-4]
        #intensity = expInfo[-3]
        #lightType = expInfo[-2]
        #windState = expInfo[-1][:-4]
        
        startTime = expInfo[2]
        genotype = expInfo[3]
        intensity = expInfo[4][3:]
        lightType = "Constant"
        windState = "NoAir"
        
        
        if 'w1118' in genotype:
            status = 'Parent'
        else:
            status = 'Offspring'
            
        df = pd.read_csv(f)
        timeIndex = pd.to_datetime(df['Time'])
        df = df.drop(['Time'], axis=1)
        df.index = (timeIndex)
        
        lightON = df[df['PatternState'] != 'Pattern 00']
        chunks = splitDataintoEpochChunks(lightON)
        
        for chunk in chunks:
            pattern = chunk['PatternState'][0]        
            
            lastTimePointofTheChunk = chunk.index[-1]            
            epoch = lastTimePointofTheChunk - dt.timedelta(seconds=lastXseconds)
            
            dataOfInterest = chunk.between_time(start_time=epoch.time(), end_time=lastTimePointofTheChunk.time())
            halfPI = dataOfInterest['LightPI'].mean()
            
            singleFlyData = dataOfInterest.filter(regex='_InLight')
            
            for col in singleFlyData:
                sfd = singleFlyData[col]
                sfPI = calculatePI(sfd)          
                
                temp['FileName'].append(f)
                temp['Pattern'].append(pattern)
                temp['FileName_pattern'].append(f + '_' + pattern)
                temp['Date'].append(date)
                temp['StartTime'].append(startTime)
                temp['Genotype'].append(genotype)
                temp['Status'].append(status)
                temp['Intensity'].append(intensity)
                temp['Status_Intensity'].append(status + '_' + intensity)
                temp['LightType'].append(lightType)
                temp['Wind'].append(windState)
                temp['Half PI'].append(halfPI)
                temp['SingleFlyPI'].append(sfPI)
                temp['LightType_Intensity'].append(lightType + '_' + intensity)
                temp['Intensity_Wind'].append(intensity + '_' + windState)
                temp['LightType_Intensity_Wind'].append(lightType + '_' + intensity + '_' + windState)
                temp['Status_LightType_Intensity_Wind'].append(status + '_' + lightType + '_' + intensity + '_' + windState)

results = pd.DataFrame(temp, columns=['FileName','Pattern', 'FileName_pattern', 'Date', 'StartTime', 'Genotype','Status', 'Intensity','LightType', 'Wind','LightType_Intensity','Intensity_Wind','Status_Intensity','LightType_Intensity_Wind','Status_LightType_Intensity_Wind','SingleFlyPI', 'Half PI'])
#upWind = results[results['Pattern'] == 'Pattern 10']
#downWind = results[results['Pattern'] == 'Pattern 01']



# ### The analysis (Single fly PI etc.) will be saved as a df called "results". You can display it here.

# In[37]:

results


# ### To plot the half PIs instead of Single Fly PI, run this cell. Otherwise NOT neccessary.

# In[16]:

halfPI = results.drop_duplicates('FileName_pattern')
halfPI = halfPI.drop('SingleFlyPI',1)


# ### Can see which genotypes there are in the analysis, and assign color to them for the plots

# In[44]:

print results['Genotype'].unique()
myPal = {#results['Genotype'].unique()[0] : 'cyan',
        results['Genotype'].unique()[0] : 'lightgreen',
        results['Genotype'].unique()[1] : 'red'}


# ### Can check the experimental conditions and use the ones you want to compare as "idx" in the next cells

# In[16]:

results['Status_LightType_Intensity_Wind'].unique()


# In[45]:

fig, contrastHalfPI = bs.contrastplot(data = results, 
                                x = 'Status_LightType_Intensity_Wind', y = 'SingleFlyPI', hue = 'Genotype',
                                 idx = (('Parent_Constant_EighthGreen_NoAir','Offspring_Constant_EighthGreen_NoAir'),                                        
                                        ('Parent_Constant_QuarterGreen_NoAir','Offspring_Constant_QuarterGreen_NoAir'),
                                        ('Parent_Constant_HalfGreen_NoAir','Offspring_Constant_HalfGreen_NoAir'),
                                        ('Parent_Constant_FullGreen_NoAir','Offspring_Constant_FullGreen_NoAir'),
                                        
                                      
                                        )                                   
                                ,
                                 pal = myPal,
                                 swarmYlim = (-1,1),
                                 contrastYlim = (-1,1),
                                 size = 6,
                                 figsize=(10,10),
                                 floatContrast = False,
                                 summaryBar = True,
                                 showAllYAxes = False)

for j in range(0,6,2):
    fig.axes[j].legend_.remove()

for k in range(0,8,1):
    labels = fig.axes[k].get_xticklabels()
    plt.setp(labels, rotation=45, horizontalalignment = 'right')

title = 'Constant light - No wind - Last 15 sec'
fig.suptitle(title,fontsize = 20)
plt.savefig(str(title) +'.pdf',dpi=1000,bbox_inches='tight')
contrastHalfPI.to_csv(str(title) +'.csv')


# ### Constant Light - No Wind - Upwind and Downwind combined - Last 15 sec

# In[10]:

fig, contrastHalfPI = bs.contrastplot(data = results, 
                                x = 'Status_LightType_Intensity_Wind', y = 'SingleFlyPI', hue = 'Genotype',
                                 idx = (('Parent_Constant_14uW_NoAir','Offspring_Constant_14uW_NoAir'),                                        
                                        
                                        
                                        ('Parent_Constant_42uW_NoAir', 'Offspring_Constant_42uW_NoAir'),                                       
                                        
                                        
                                        ('Parent_Constant_70uW_NoAir', 'Offspring_Constant_70uW_NoAir')
                                        
                                        )                                   
                                ,
                                 pal = myPal,
                                 swarmYlim = (-1,1),
                                 contrastYlim = (-1,1),
                                 size = 6,
                                 figsize=(10,10),
                                 floatContrast = False,
                                 summaryBar = True,
                                 showAllYAxes = False)

for j in range(0,4,2):
    fig.axes[j].legend_.remove()

for k in range(0,6,1):
    labels = fig.axes[k].get_xticklabels()
    plt.setp(labels, rotation=45, horizontalalignment = 'right')

title = 'Constant light - No wind - Last 15 sec'
fig.suptitle(title,fontsize = 20)
plt.savefig(str(title) +'.pdf',dpi=1000,bbox_inches='tight')
contrastHalfPI.to_csv(str(title) +'.csv')


# ### Constant Light - No Wind - UpWind - Last 15 sec

# In[11]:

fig, contrastHalfPI = bs.contrastplot(data = upWind, 
                                x = 'Status_LightType_Intensity_Wind', y = 'SingleFlyPI', hue = 'Genotype',
                                 idx = (('Parent_Constant_14uW_NoAir','Offspring_Constant_14uW_NoAir'),
                                        
                                        
                                        
                                        ('Parent_Constant_42uW_NoAir', 'Offspring_Constant_42uW_NoAir'),
                                        
                                        
                                        
                                        ('Parent_Constant_70uW_NoAir', 'Offspring_Constant_70uW_NoAir')
                                        
                                        )                                   
                                ,
                                 pal = myPal,
                                 swarmYlim = (-1,1),
                                 contrastYlim = (-1,1),
                                 size = 6,
                                 figsize=(10,10),
                                 floatContrast = False,
                                 summaryBar = True,
                                 showAllYAxes = False)

for j in range(0,4,2):
    fig.axes[j].legend_.remove()

for k in range(0,6,1):
    labels = fig.axes[k].get_xticklabels()
    plt.setp(labels, rotation=45, horizontalalignment = 'right')

title = 'Constant light - No wind (Upwind half) Last 15 sec'
fig.suptitle(title,fontsize = 20)
plt.savefig(str(title) +'.pdf',dpi=1000,bbox_inches='tight')
contrastHalfPI.to_csv(str(title) +'.csv')


# ### Constant Light - No Wind - DownWind - Last 15 sec

# In[12]:

fig, contrastHalfPI = bs.contrastplot(data = downWind, 
                                x = 'Status_LightType_Intensity_Wind', y = 'SingleFlyPI', hue = 'Genotype',
                                 idx = (('Parent_Constant_14uW_NoAir','Offspring_Constant_14uW_NoAir'),
                                        
                                        
                                        
                                        ('Parent_Constant_42uW_NoAir', 'Offspring_Constant_42uW_NoAir'),
                                        
                                        
                                        
                                        ('Parent_Constant_70uW_NoAir', 'Offspring_Constant_70uW_NoAir')
                                        
                                        )                                   
                                ,
                                 pal = myPal,
                                 swarmYlim = (-1,1),
                                 contrastYlim = (-1,1),
                                 size = 6,
                                 figsize=(10,10),
                                 floatContrast = False,
                                 summaryBar = True,
                                 showAllYAxes = False)

for j in range(0,4,2):
    fig.axes[j].legend_.remove()

for k in range(0,6,1):
    labels = fig.axes[k].get_xticklabels()
    plt.setp(labels, rotation=45, horizontalalignment = 'right')

title = 'Constant light - No wind(Downwind half) Last 15 sec'
fig.suptitle(title,fontsize = 20)
plt.savefig(str(title) +'.pdf',dpi=1000,bbox_inches='tight')
contrastHalfPI.to_csv(str(title) +'.csv')


# ### Constant Light - Wind - Upwind and Downwind combined - Last 15 sec

# In[13]:

fig, contrastHalfPI = bs.contrastplot(data = results, 
                                x = 'Status_LightType_Intensity_Wind', y = 'SingleFlyPI', hue = 'Genotype',
                                 idx = (('Parent_Constant_14uW_Air','Offspring_Constant_14uW_Air'),
                                                                                                                       
                                        ('Parent_Constant_42uW_Air', 'Offspring_Constant_42uW_Air'),                                                                              
                                        
                                        ('Parent_Constant_70uW_Air', 'Offspring_Constant_70uW_Air')
                                        
                                        )                                   
                                ,
                                 pal = myPal,
                                 swarmYlim = (-1,1),
                                 contrastYlim = (-1,1),
                                 size = 6,
                                 figsize=(10,10),
                                 floatContrast = False,
                                 summaryBar = True,
                                 showAllYAxes = False)

for j in range(0,4,2):
    fig.axes[j].legend_.remove()

for k in range(0,6,1):
    labels = fig.axes[k].get_xticklabels()
    plt.setp(labels, rotation=45, horizontalalignment = 'right')

title = 'Constant light - Wind (Upwind & Downwind) Last 15 sec'
fig.suptitle(title,fontsize = 20)
plt.savefig(str(title) +'.pdf',dpi=1000,bbox_inches='tight')
contrastHalfPI.to_csv(str(title) +'.csv')


# ### Constant Light - Wind - Upwind - Last 15 sec

# In[14]:

fig, contrastHalfPI = bs.contrastplot(data = upWind, 
                                x = 'Status_LightType_Intensity_Wind', y = 'SingleFlyPI', hue = 'Genotype',
                                 idx = (('Parent_Constant_14uW_Air','Offspring_Constant_14uW_Air'),
                                                                                                                        
                                        ('Parent_Constant_42uW_Air', 'Offspring_Constant_42uW_Air'),                                    
                                                                               
                                        ('Parent_Constant_70uW_Air', 'Offspring_Constant_70uW_Air')
                                        
                                        )                                   
                                ,
                                 pal = myPal,
                                 swarmYlim = (-1,1),
                                 contrastYlim = (-1,1.5),
                                 size = 6,
                                 figsize=(10,10),
                                 floatContrast = False,
                                 summaryBar = True,
                                 showAllYAxes = False)

for j in range(0,4,2):
    fig.axes[j].legend_.remove()

for k in range(0,6,1):
    labels = fig.axes[k].get_xticklabels()
    plt.setp(labels, rotation=45, horizontalalignment = 'right')

title = 'Constant light - Wind(Upwind half) Last 15 sec'
fig.suptitle(title,fontsize = 20)
plt.savefig(str(title) +'.pdf',dpi=1000,bbox_inches='tight')
contrastHalfPI.to_csv(str(title) +'.csv')


# ### Constant Light - Wind - Downwind - Last 15 sec

# In[15]:

fig, contrastHalfPI = bs.contrastplot(data = downWind, 
                                x = 'Status_LightType_Intensity_Wind', y = 'SingleFlyPI', hue = 'Genotype',
                                 idx = (('Parent_Constant_14uW_Air','Offspring_Constant_14uW_Air'),                                                                               
                                        
                                        ('Parent_Constant_42uW_Air', 'Offspring_Constant_42uW_Air'),                                       
                                                                                
                                        ('Parent_Constant_70uW_Air', 'Offspring_Constant_70uW_Air')
                                        
                                        )                                   
                                ,
                                 pal = myPal,
                                 swarmYlim = (-1,1),
                                 contrastYlim = (-1,1),
                                 size = 6,
                                 figsize=(10,10),
                                 floatContrast = False,
                                 summaryBar = True,
                                 showAllYAxes = False)

for j in range(0,4,2):
    fig.axes[j].legend_.remove()

for k in range(0,6,1):
    labels = fig.axes[k].get_xticklabels()
    plt.setp(labels, rotation=45, horizontalalignment = 'right')

title = 'Constant light - Wind(Downwind half) Last 15 sec'
fig.suptitle(title,fontsize = 20)
plt.savefig(str(title) +'.pdf',dpi=1000,bbox_inches='tight')
contrastHalfPI.to_csv(str(title) +'.csv')


# ## Warning: you have to set the "lastXsec" variable to 30, and run the analysis cell again before starting to plot 30sec data!

# ### Constant Light - No Wind (Downwind and Upwind) Last 30 sec

# In[18]:

fig, contrastHalfPI = bs.contrastplot(data = results, 
                                x = 'Status_LightType_Intensity_Wind', y = 'SingleFlyPI', hue = 'Genotype',
                                 idx = (('Parent_Constant_14uW_NoAir','Offspring_Constant_14uW_NoAir'),
                                                                              
                                        ('Parent_Constant_42uW_NoAir', 'Offspring_Constant_42uW_NoAir'),
                                                                            
                                        ('Parent_Constant_70uW_NoAir', 'Offspring_Constant_70uW_NoAir')
                                        
                                        )                                   
                                ,
                                 pal = myPal,
                                 swarmYlim = (-1,1),
                                 contrastYlim = (-1,1),
                                 size = 6,
                                 figsize=(10,10),
                                 floatContrast = False,
                                 summaryBar = True,
                                 showAllYAxes = False)

for j in range(0,4,2):
    fig.axes[j].legend_.remove()

for k in range(0,6,1):
    labels = fig.axes[k].get_xticklabels()
    plt.setp(labels, rotation=45, horizontalalignment = 'right')

title = 'Constant light - No wind Last 30 sec'
fig.suptitle(title,fontsize = 20)
plt.savefig(str(title) +'.pdf',dpi=1000,bbox_inches='tight')
contrastHalfPI.to_csv(str(title) +'.csv')


# ### Constant Light - No Wind (Downwind half) Last 30 sec

# In[19]:

fig, contrastHalfPI = bs.contrastplot(data = downWind, 
                                x = 'Status_LightType_Intensity_Wind', y = 'SingleFlyPI', hue = 'Genotype',
                                 idx = (('Parent_Constant_14uW_NoAir','Offspring_Constant_14uW_NoAir'),
                                                                              
                                        ('Parent_Constant_42uW_NoAir', 'Offspring_Constant_42uW_NoAir'),
                                                                            
                                        ('Parent_Constant_70uW_NoAir', 'Offspring_Constant_70uW_NoAir')
                                        
                                        )                                   
                                ,
                                 pal = myPal,
                                 swarmYlim = (-1,1),
                                 contrastYlim = (-1,1),
                                 size = 6,
                                 figsize=(10,10),
                                 floatContrast = False,
                                 summaryBar = True,
                                 showAllYAxes = False)

for j in range(0,4,2):
    fig.axes[j].legend_.remove()

for k in range(0,6,1):
    labels = fig.axes[k].get_xticklabels()
    plt.setp(labels, rotation=45, horizontalalignment = 'right')

title = 'Constant light - No wind (Downwind half) Last 30 sec'
fig.suptitle(title,fontsize = 20)
plt.savefig(str(title) +'.pdf',dpi=1000,bbox_inches='tight')
contrastHalfPI.to_csv(str(title) +'.csv')


# ### Constant Light - No Wind (Upwind half) Last 30 sec

# In[20]:

fig, contrastHalfPI = bs.contrastplot(data = upWind, 
                                x = 'Status_LightType_Intensity_Wind', y = 'SingleFlyPI', hue = 'Genotype',
                                 idx = (('Parent_Constant_14uW_NoAir','Offspring_Constant_14uW_NoAir'),
                                                                              
                                        ('Parent_Constant_42uW_NoAir', 'Offspring_Constant_42uW_NoAir'),
                                                                            
                                        ('Parent_Constant_70uW_NoAir', 'Offspring_Constant_70uW_NoAir')
                                        
                                        )                                   
                                ,
                                 pal = myPal,
                                 swarmYlim = (-1,1),
                                 contrastYlim = (-1,1),
                                 size = 6,
                                 figsize=(10,10),
                                 floatContrast = False,
                                 summaryBar = True,
                                 showAllYAxes = False)

for j in range(0,4,2):
    fig.axes[j].legend_.remove()

for k in range(0,6,1):
    labels = fig.axes[k].get_xticklabels()
    plt.setp(labels, rotation=45, horizontalalignment = 'right')

title = 'Constant light - No wind (Upwind half) Last 30 sec'
fig.suptitle(title,fontsize = 20)
plt.savefig(str(title) +'.pdf',dpi=1000,bbox_inches='tight')
contrastHalfPI.to_csv(str(title) +'.csv')


# ### Constant Light - Wind (Downwind and Upwind) Last 30 sec

# In[21]:

fig, contrastHalfPI = bs.contrastplot(data = results, 
                                x = 'Status_LightType_Intensity_Wind', y = 'SingleFlyPI', hue = 'Genotype',
                                 idx = (('Parent_Constant_14uW_Air','Offspring_Constant_14uW_Air'),
                                        
                                       
                                        
                                        ('Parent_Constant_42uW_Air', 'Offspring_Constant_42uW_Air'),
                                        
                                       
                                        
                                        ('Parent_Constant_70uW_Air', 'Offspring_Constant_70uW_Air')
                                        
                                        )                                   
                                ,
                                 pal = myPal,
                                 swarmYlim = (-1,1),
                                 contrastYlim = (-1,1),
                                 size = 6,
                                 figsize=(10,10),
                                 floatContrast = False,
                                 summaryBar = True,
                                 showAllYAxes = False)

for j in range(0,4,2):
    fig.axes[j].legend_.remove()

for k in range(0,6,1):
    labels = fig.axes[k].get_xticklabels()
    plt.setp(labels, rotation=45, horizontalalignment = 'right')

title = 'Constant light - Wind(Downwind and Upwind) Last 30 sec'
fig.suptitle(title,fontsize = 20)
plt.savefig(str(title) +'.pdf',dpi=1000,bbox_inches='tight')
contrastHalfPI.to_csv(str(title) +'.csv')


# ### Constant Light - Wind (Downwind half) Last 30 sec

# In[22]:

fig, contrastHalfPI = bs.contrastplot(data = downWind, 
                                x = 'Status_LightType_Intensity_Wind', y = 'SingleFlyPI', hue = 'Genotype',
                                 idx = (('Parent_Constant_14uW_Air','Offspring_Constant_14uW_Air'),
                                        
                                       
                                        
                                        ('Parent_Constant_42uW_Air', 'Offspring_Constant_42uW_Air'),
                                        
                                       
                                        
                                        ('Parent_Constant_70uW_Air', 'Offspring_Constant_70uW_Air')
                                        
                                        )                                   
                                ,
                                 pal = myPal,
                                 swarmYlim = (-1,1),
                                 contrastYlim = (-1,1),
                                 size = 6,
                                 figsize=(10,10),
                                 floatContrast = False,
                                 summaryBar = True,
                                 showAllYAxes = False)

for j in range(0,4,2):
    fig.axes[j].legend_.remove()

for k in range(0,6,1):
    labels = fig.axes[k].get_xticklabels()
    plt.setp(labels, rotation=45, horizontalalignment = 'right')

title = 'Constant light - Wind(Downwind half) Last 30 sec'
fig.suptitle(title,fontsize = 20)
plt.savefig(str(title) +'.pdf',dpi=1000,bbox_inches='tight')
contrastHalfPI.to_csv(str(title) +'.csv')


# ### Constant Light - Wind (Upwind half) Last 30 sec

# In[23]:

fig, contrastHalfPI = bs.contrastplot(data = upWind, 
                                x = 'Status_LightType_Intensity_Wind', y = 'SingleFlyPI', hue = 'Genotype',
                                 idx = (('Parent_Constant_14uW_Air','Offspring_Constant_14uW_Air'),
                                        
                                       
                                        
                                        ('Parent_Constant_42uW_Air', 'Offspring_Constant_42uW_Air'),
                                        
                                       
                                        
                                        ('Parent_Constant_70uW_Air', 'Offspring_Constant_70uW_Air')
                                        
                                        )                                   
                                ,
                                 pal = myPal,
                                 swarmYlim = (-1,1),
                                 contrastYlim = (-1,1),
                                 size = 6,
                                 figsize=(10,10),
                                 floatContrast = False,
                                 summaryBar = True,
                                 showAllYAxes = False)

for j in range(0,4,2):
    fig.axes[j].legend_.remove()

for k in range(0,6,1):
    labels = fig.axes[k].get_xticklabels()
    plt.setp(labels, rotation=45, horizontalalignment = 'right')

title = 'Constant light - Wind(Upwind half) Last 30 sec'
fig.suptitle(title,fontsize = 20)
plt.savefig(str(title) +'.pdf',dpi=1000,bbox_inches='tight')
contrastHalfPI.to_csv(str(title) +'.csv')


# In[39]:

os.close()


# In[ ]:


