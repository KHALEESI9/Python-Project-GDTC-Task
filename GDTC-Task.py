#!/usr/bin/env python
# coding: utf-8

# In[44]:


import pandas as pd


# In[45]:


import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.pyplot import pie,axis,show
import os


# In[3]:


working_directory=os.getcwd()
print(working_directory)


# In[8]:


path=working_directory+'/Documents/insurance_data.csv'


# In[9]:


ins = pd.read_csv(path)


# In[10]:


ins.head()


# In[11]:


working_directory=os.getcwd()
print(working_directory)


# In[12]:


path=working_directory+'/Documents/employee_data.csv'


# In[13]:


emp = pd.read_csv(path)


# In[14]:


emp.head()


# In[15]:


working_directory=os.getcwd()
print(working_directory)


# In[16]:


path=working_directory+'/Documents/vendor_data.csv'


# In[17]:


ven = pd.read_csv(path)
ven.head()


# In[18]:


##1-Merging all Three Dataset And Creating 1 view


# In[19]:


doc1 = pd.merge(ins,emp,on="AGENT_ID",how = "outer")
doc1


# In[20]:


## All 3 Dataset merged As doc2


doc2 = pd.merge(doc1,ven,on= 'VENDOR_ID',how="outer")
doc2


# In[21]:


## 2-: Top 3 Insurance Type where we are getting most insurance claims.


# In[22]:


ins.head()


# In[23]:


sns.countplot(doc2.INSURANCE_TYPE)	


# In[24]:


ins['INSURANCE_TYPE'].value_counts()[:3]


# In[25]:


##Top-3 Insurance Type where we are getting most insurance claims - 
 ##                       1-: Property
 ##                      2-: Mobile
 ##                      3-:Health 


# In[26]:


## 3-: Top 5 States where we are getting most insurance claims


# In[27]:


sns.countplot(ins.STATE)


# In[28]:


ins.STATE.value_counts()


# In[29]:


h_risk_state=ins[['STATE','RISK_SEGMENTATION']] [ins['RISK_SEGMENTATION'] == 'H']
h_risk_state.value_counts()[:5]


# In[30]:


sns.countplot(h_risk_state.STATE)


# In[31]:


## Tp 5 state where we are getting most insurance claims for customer belonging to HIGH(H) risk segment are
# 1-CA     
# 2-AZ
# 3-FL
# 4-TN
# 5-AR


# In[32]:


## 4-: “COLOCATION"


# In[33]:


def colocation(row):
    if (row['STATE_y'] == row['INCIDENT_STATE']) & (row['STATE_y'] == row['STATE_x']):
        val = 1
    else:
        val = 0
    return val

doc2['Colocation'] = doc2.apply(colocation, axis= 1)


# In[34]:


doc2.Colocation.describe()


# In[35]:


doc2.Colocation.mean()


# In[36]:


## mean of this new column - 0.00431668792308447


# In[37]:


## 5 -: Data entry error

doc2.info()


# In[38]:


doc2.AUTHORITY_CONTACTED.head()


# In[39]:


## Updating “AUTHORITY_CONTACTED” to Police

doc2.loc[(doc2['POLICE_REPORT_AVAILABLE'] == '1','AUTHORITY_CONTACTED')] = 'police'


# In[40]:


doc2[['AUTHORITY_CONTACTED', 'POLICE_REPORT_AVAILABLE']].head()


# In[41]:


doc2.AUTHORITY_CONTACTED.value_counts()


# In[42]:


## 6-: CLAIM_DEVIATION

doc2['TXN_DATE_TIME'] = pd.to_datetime(doc2['TXN_DATE_TIME']).dt.date
print(doc2)


# In[47]:


doc2['TXN_DATE_TIME'] = pd.to_datetime(doc2['TXN_DATE_TIME'])
doc2['TXN_DATE_TIME'].max()


# In[48]:


doc2['TXN_DATE_TIME'].min()


# In[49]:


avg_claim_amt = doc2.loc[(doc2['TXN_DATE_TIME'] > '2021-05-31') & (doc2['TXN_DATE_TIME'] < '2021-06-30')]
avg_claim_amt['CLAIM_AMOUNT'].mean()


# In[50]:


## Average Claim Amount = 16667.852437417656


# In[51]:


doc2['CLAIM_DEVIATION'] =avg_claim_amt['CLAIM_AMOUNT'].mean()/doc2['CLAIM_AMOUNT']
doc2


# In[52]:


doc2['CLAIM_DEVIATION'] = doc2['CLAIM_DEVIATION'].astype(float)
doc2['claim_Deviation_value'] = doc2['CLAIM_DEVIATION'].apply(lambda x: '1' if x < 0.5 else '0')
doc2


# In[53]:


## If the value Less Than 0.5 THEN CLAIM_DEVIATION = 1 else '0'
doc2[['CLAIM_DEVIATION', 'claim_Deviation_value']].tail()


# In[54]:


## 7-: Agents who have worked on more than 2 types of Insurance Claims
doc2.info()


# In[55]:


agent=doc2.groupby(['AGENT_ID', 'AGENT_NAME', 'CLAIM_AMOUNT'])['INSURANCE_TYPE'].count().reset_index().sort_values('INSURANCE_TYPE', ascending= False,axis=0)
agent


# In[59]:


agent['INSURANCE_CLAIMED'] = agent['INSURANCE_TYPE'].astype(int)


# In[61]:


agent_insurance_claim = agent[agent['INSURANCE_CLAIMED']>2].sort_values('CLAIM_AMOUNT', ascending= False)
agent_insurance_claim.head()
## All Agents who have worked on more than 2 types of Insurance Claims


# In[63]:


## 08 -: overall change in % of the Premium Amount

overall_change = pd.DataFrame(doc2.groupby(['INSURANCE_TYPE'])['PREMIUM_AMOUNT'].sum())
overall_change['INSURANCE_TYPE'] = overall_change.index
overall_change.reset_index(drop= True, inplace= True)
overall_change


# In[64]:


overall_change['PREMIUM_AMOUNT'].sum()


# In[67]:


overall_change['PERCENTAGE'] = (overall_change['PREMIUM_AMOUNT']/overall_change['PREMIUM_AMOUNT'].sum())*100
overall_change


# In[68]:


def new_premium(row):
    if ((row['INSURANCE_TYPE']=='Health') | (row['INSURANCE_TYPE']=='Property')):
        val = row['PREMIUM_AMOUNT']+(0.07*row['PREMIUM_AMOUNT'])
    elif ((row['INSURANCE_TYPE']=='Life') | (row['INSURANCE_TYPE']=='Motor')):
        val = row['PREMIUM_AMOUNT']+(0.02*row['PREMIUM_AMOUNT'])
    elif ((row['INSURANCE_TYPE']=='Mobile') | (row['INSURANCE_TYPE']=='Travel')):
        val = row['PREMIUM_AMOUNT']-(0.10*row['PREMIUM_AMOUNT'])
    return val

overall_change['NEW_PREMIUM'] = overall_change.apply(new_premium, axis= 1)


# In[69]:


overall_change['NEW_PREMIUM_percentage'] = overall_change['NEW_PREMIUM']/(overall_change['NEW_PREMIUM'].sum())*100


# In[70]:


## Mobile & Travel Insurance premium are discounted by 10%
## Health and Property Insurance premium are increased by 7%
## Life and Motor Insurance premium are marginally increased by 2%

overall_change


# In[71]:


## 9-: ELIGIBLE_FOR_DISCOUNT
doc2.info()


# In[72]:


doc2[['TENURE','EMPLOYMENT_STATUS','NO_OF_FAMILY_MEMBERS']].head()


# In[73]:


def business_disc(doc2):
    if((doc2['TENURE']>60) & (doc2['EMPLOYMENT_STATUS']=='N') &(doc2['NO_OF_FAMILY_MEMBERS']>=4)):
        val = 1
    else:
        val = 0
    return val

doc2['ELIGIBLE_FOR_DISCOUNT'] = doc2.apply(business_disc, axis=1)


# In[74]:


doc2['ELIGIBLE_FOR_DISCOUNT'].mean()


# In[76]:


## mean “ELIGIBLE_FOR_DISCOUNT” is -: 0.0293338565682331


# In[77]:


## 10-: Claim Velocity

doc2['TXN_DATE_TIME'].max()


# In[78]:


Num_of_claim_in_last_30_days = doc2.loc[(doc2['TXN_DATE_TIME'] > '2021-05-31') & (doc2['TXN_DATE_TIME'] < '2021-06-30')]
Num_of_claim_in_last_03_days = doc2.loc[(doc2['TXN_DATE_TIME'] > '2021-06-27') & (doc2['TXN_DATE_TIME'] < '2021-06-30')]


# In[79]:


Num_of_claim_in_last_30_days.groupby('INSURANCE_TYPE').count()


# In[80]:


Num_of_claim_in_last_03_days.groupby('INSURANCE_TYPE').count()


# In[82]:


claims = pd.DataFrame()
claims= {
          "INSURANCE_TYPE":["Health","Life","Mobile","Motor","Property","Travel"],
          "Num_of_claim_in_last_30_days":[131,137,129,119,122,121],
          "Num_of_claim_in_last_03_days":[9,9,2,10,9,12]
}
claims = pd.DataFrame(claims)


# In[83]:


claims.head()


# In[84]:


claims["CLAIM_VELOCITY"] = claims['Num_of_claim_in_last_30_days']/claims['Num_of_claim_in_last_03_days']


# In[85]:


claims

##claim Velocity


# In[86]:


##total Number of claims=
doc2['TXN_DATE_TIME'].max()
doc2['TXN_DATE_TIME'].min()
total_Number_of_claim = Num_of_claim_in_last_30_days = doc2.loc[(doc2['TXN_DATE_TIME'] > '2020-06-01') & (doc2['TXN_DATE_TIME'] < '2021-06-30')]
total_Number_of_claim.groupby('INSURANCE_TYPE').count()


# In[87]:


## 11 -: Low Performing Agents 


doc2_subset = doc2.filter(['AGENT_ID','INSURANCE_TYPE'])


# In[88]:


doc2_subset.value_counts(sort = False).head(20)


# In[89]:


doc2_subset.AGENT_ID.value_counts(ascending = True)


# In[90]:


low_performance = doc2_subset.value_counts()


# In[91]:


low_performance


# In[92]:


print("5th percentile of low_performace : ",
       np.percentile(low_performance, 5))


# In[93]:


## All low performing agents are -: AGENT00252     1
##                                  AGENT01161


# In[94]:


doc2.info()


# In[95]:


#12 -: Suspicious Agents


doc2[['CLAIM_STATUS','RISK_SEGMENTATION','INCIDENT_SEVERITY']].head()


# In[98]:


suspicious = doc2[(doc2['CLAIM_STATUS']=='A') & (doc2['RISK_SEGMENTATION']=='H') & (doc2['INCIDENT_SEVERITY']=='Major Loss')]


# In[99]:


def suspicious_agent(suspicious):
    if(suspicious['CLAIM_AMOUNT']>=15000):
        val=1
    else:
        val=0
    return val
suspicious['SUSPICIOUS_Agent'] = suspicious.apply(suspicious_agent, axis= 1)


# In[101]:


suspicious[['AGENT_ID', 'AGENT_NAME', 'CLAIM_AMOUNT', 'SUSPICIOUS_Agent']].head()


# In[103]:


suspicious['SUSPICIOUS_Agent'].mean()


# In[104]:


## mean of SUSPICIOUS_Agent column is -: 0.3462414578587699


# In[ ]:




