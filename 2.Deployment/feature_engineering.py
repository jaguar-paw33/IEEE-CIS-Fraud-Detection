import gc
import numpy as np
import pandas as pd
import pickle
import datetime

input_path = './Input/'
mapping_path = './Mappings/'

def parent_device_name(df):
  '''
    Utility Function to map deice name to its parent company.
  '''

  if(df['device_name'].isna().all()):
        return df

  df.loc[df['device_name'].str.contains('SM', na=False), 'device_name'] = 'Samsung'
  df.loc[df['device_name'].str.contains('SAMSUNG', na=False), 'device_name'] = 'Samsung'
  df.loc[df['device_name'].str.contains('GT-', na=False), 'device_name'] = 'Samsung'
  df.loc[df['device_name'].str.contains('Moto G', na=False), 'device_name'] = 'Motorola'
  df.loc[df['device_name'].str.contains('Moto', na=False), 'device_name'] = 'Motorola'
  df.loc[df['device_name'].str.contains('moto', na=False), 'device_name'] = 'Motorola'
  df.loc[df['device_name'].str.contains('LG-', na=False), 'device_name'] = 'LG'
  df.loc[df['device_name'].str.contains('rv:', na=False), 'device_name'] = 'RV'
  df.loc[df['device_name'].str.contains('HUAWEI', na=False), 'device_name'] = 'Huawei'
  df.loc[df['device_name'].str.contains('ALE-', na=False), 'device_name'] = 'Huawei'
  df.loc[df['device_name'].str.contains('-L', na=False), 'device_name'] = 'Huawei'
  df.loc[df['device_name'].str.contains('Blade', na=False), 'device_name'] = 'ZTE'
  df.loc[df['device_name'].str.contains('BLADE', na=False), 'device_name'] = 'ZTE'
  df.loc[df['device_name'].str.contains('Linux', na=False), 'device_name'] = 'Linux'
  df.loc[df['device_name'].str.contains('XT', na=False), 'device_name'] = 'Sony'
  df.loc[df['device_name'].str.contains('HTC', na=False), 'device_name'] = 'HTC'
  df.loc[df['device_name'].str.contains('ASUS', na=False), 'device_name'] = 'Asus'
  df.loc[df.device_name.isin(df.device_name.value_counts()[df.device_name.value_counts() < 200].index), 'device_name'] = "Others"

  return df




def basic_feature_engineering(testdata):
    
    testdata['TransactionMT'] = (testdata['TransactionDT']//60)%60
    testdata['TransactionMT_X'] = np.sin(2.*np.pi*testdata['TransactionMT']/60.)
    testdata['TransactionMT_Y'] = np.cos(2.*np.pi*testdata['TransactionMT']/60.)
    
    testdata['TransactionHR'] = (testdata['TransactionDT']//3600)%24
    testdata['TransactionHR_X'] = np.sin(2.*np.pi*testdata['TransactionHR']/24.)
    testdata['TransactionHR_Y'] = np.cos(2.*np.pi*testdata['TransactionHR']/24.)
    
    testdata['TransactionDay'] = testdata['TransactionDT']//(24*3600)
    
    testdata['TransactionWD'] = (testdata['TransactionDT']//(24*3600))%7
    
    testdata_amt_whole = [int(str(a).split('.')[0]) for a in testdata['TransactionAmt'].values]
    testdata_amt_decimal = [int(str(a).split('.')[1]) for a in testdata['TransactionAmt'].values]
    testdata['dollars'] = testdata_amt_whole
    testdata['cents'] = testdata_amt_decimal
    
    testdata['TransactionAmt_log'] = np.log(testdata['TransactionAmt'])

    testdata['card1_div_1000'] = testdata['card1']//1000

    testdata['card2_div_10'] = testdata['card2']//10


    parent_domain = {'gmail.com':'gmail', 'outlook.com':'microsoft', 
                     'yahoo.com':'yahoo', 'mail.com':'mail', 'anonymous.com':'anonymous', 
                     'hotmail.com':'microsoft', 'verizon.net':'verizon', 'aol.com':'aol', 
                     'me.com':'apple', 'comcast.net':'comcast', 'optonline.net':'optimum', 
                     'cox.net':'cox', 'charter.net':'spectrum', 'rocketmail.com':'yahoo', 
                     'prodigy.net.mx':'AT&T', 'embarqmail.com':'century_link', 'icloud.com':'apple', 
                     'live.com.mx':'microsoft', 'gmail':'gmail', 'live.com':'microsoft', 
                     'att.net':'AT&T', 'juno.com':'juno', 'ymail.com':'yahoo', 
                     'sbcglobal.net':'sbcglobal', 'bellsouth.net':'AT&T', 'msn.com':'microsoft', 
                     'q.com':'century_link','yahoo.com.mx':'yahoo', 'centurylink.net':'century_link',  
                     'servicios-ta.com':'asur','earthlink.net':'earthlink', 'hotmail.es':'microsoft', 
                     'cfl.rr.com':'spectrum', 'roadrunner.com':'spectrum','netzero.net':'netzero', 
                     'gmx.de':'gmx','suddenlink.net':'suddenlink','frontiernet.net':'frontier', 
                     'windstream.net':'windstream','frontier.com':'frontier','outlook.es':'microsoft', 
                     'mac.com':'apple','netzero.com':'netzero','aim.com':'aol', 
                     'web.de':'web_de','twc.com':'whois','cableone.net':'sparklight', 
                     'yahoo.fr':'yahoo','yahoo.de':'yahoo','yahoo.es':'yahoo', 'scranton.edu':'scranton', 
                     'sc.rr.com':'sc_rr','ptd.net':'ptd','live.fr':'microsoft', 
                     'yahoo.co.uk':'yahoo','hotmail.fr':'microsoft','hotmail.de':'microsoft', 
                     'hotmail.co.uk':'microsoft','protonmail.com':'protonmail','yahoo.co.jp':'yahoo'}

    
    testdata_P_emaildomain = testdata['P_emaildomain']

    testdata['P_parent_domain'] = [np.nan if pd.isna(domain) else parent_domain[domain] for domain in testdata_P_emaildomain] 

    testdata['P_domain_name'] = [np.nan if pd.isna(addrs) else addrs.split('.')[0] for addrs in testdata_P_emaildomain]

    testdata['P_top_level_domain'] = [np.nan if (pd.isna(addrs)) or (len(addrs.split('.'))<=1) else '.'.join(addrs.split('.')[1:]) for addrs in testdata_P_emaildomain]

    
    
    testdata_R_emaildomain = testdata['R_emaildomain']

    testdata['R_parent_domain'] = [np.nan if pd.isna(domain) else parent_domain[domain] for domain in testdata_R_emaildomain] 

    testdata['R_domain_name'] = [np.nan if pd.isna(addrs) else addrs.split('.')[0] for addrs in testdata_R_emaildomain]

    testdata['R_top_level_domain'] = [np.nan if (pd.isna(addrs)) or (len(addrs.split('.'))<=1) else '.'.join(addrs.split('.')[1:]) for addrs in testdata_R_emaildomain]
    
    testdata['device_name'] = [np.nan if pd.isna(v) else v.split('/')[0] for v in testdata['DeviceInfo'].values]

    testdata['device_version'] = [np.nan if (pd.isna(v)) or (len(v.split('/'))<=1) else v.split('/')[1] for v in testdata['DeviceInfo'].values]

    testdata = parent_device_name(testdata)
    
    testdata['os_name'] = [info if (pd.isna(info)) or (len(info.split())<=1) else ' '.join(info.split()[:-1]) for info in testdata['id_30']]

    testdata['os_version'] = [np.nan if (pd.isna(info)) or (len(info.split())<=1) else info.split()[-1] for info in testdata['id_30']]
    
    testdata['screen_width'] = [np.nan if pd.isna(v) else v.split('x')[0] for v in testdata['id_33'].values]

    testdata['screen_height'] = [np.nan if (pd.isna(v)) or len(v.split('x'))<=1 else v.split('x')[1] for v in testdata['id_33'].values]
    
    
    testdata['card_intr1'] = testdata['card1_div_1000'].astype(str) + " " + \
                           testdata['card2_div_10'].astype(str) + " " + \
                           testdata['card3'].astype(str) + " " + \
                           testdata['card5'].astype(str) + " " + \
                           testdata['card6'].astype(str)


    testdata['card_intr2'] = testdata['card1'].astype(str) + " " + \
                               testdata['card2'].astype(str) + " " + \
                               testdata['card3'].astype(str) + " " + \
                               testdata['card5'].astype(str) + " " + \
                               testdata['card6'].astype(str)
    
    
    testdata['card1_addr1'] = testdata['card1'].astype(str)+testdata['addr1'].astype(str)

    testdata['card1_addr2'] = testdata['card1'].astype(str)+testdata['addr2'].astype(str)

    
    testdata['card2_addr1'] = testdata['card2'].astype(str)+testdata['addr1'].astype(str)

    testdata['card2_addr2'] = testdata['card2'].astype(str)+testdata['addr2'].astype(str)

    
    testdata['card3_addr1'] = testdata['card3'].astype(str)+testdata['addr1'].astype(str)

    testdata['card3_addr2'] = testdata['card3'].astype(str)+testdata['addr2'].astype(str)

    
    testdata['card5_addr1'] = testdata['card5'].astype(str)+testdata['addr1'].astype(str)

    testdata['card5_addr2'] = testdata['card5'].astype(str)+testdata['addr2'].astype(str)

    
    testdata['card6_addr1'] = testdata['card6'].astype(str)+testdata['addr1'].astype(str)

    testdata['card6_addr2'] = testdata['card6'].astype(str)+testdata['addr2'].astype(str)

    
    testdata['ProductCD_addr1'] = testdata['ProductCD'].astype(str)+testdata['addr1'].astype(str)

    testdata['ProductCD_addr2'] = testdata['ProductCD'].astype(str)+testdata['addr2'].astype(str)

    
    testdata['card1_ProductCD'] =testdata['card1'].astype(str)+testdata['ProductCD'].astype(str)

    testdata['card2_ProductCD'] =testdata['card2'].astype(str)+testdata['ProductCD'].astype(str)

    testdata['card5_ProductCD'] =testdata['card5'].astype(str)+testdata['ProductCD'].astype(str)

    testdata['card6_ProductCD'] = testdata['card6'].astype(str)+testdata['ProductCD'].astype(str)
    
    
    testdata['addr1_P_emaildomain'] = testdata['addr1'].astype(str)+testdata['P_emaildomain'].astype(str)

    testdata['card1_P_emaildoman'] = testdata['card1'].astype(str)+testdata['P_emaildomain'].astype(str)

    testdata['card1_addr1_P_emaildomain'] = testdata['card1'].astype(str)+testdata['addr1_P_emaildomain'].astype(str)

    
    d_features = ["D"+str(i) for i in range(1,16) if "D"+str(i) in testdata.columns]

    for f in d_features:
        testdata[f] =  testdata[f] - testdata['TransactionDay']

    del d_features
    gc.collect()

    testdata['uid1'] = testdata['card1'].astype(str)+testdata['card2'].astype(str)+\
                         testdata['card3'].astype(str)+testdata['card5'].astype(str)+\
                         testdata['card6'].astype(str)+testdata['addr1'].astype(str)+\
                         testdata['P_emaildomain'].astype(str)


    testdata['uid2'] = testdata['card1'].astype(str)+testdata['addr1_P_emaildomain'].astype(str)
    
    return testdata



def frequency_encode(testdata, frequency_encoder_dict, features):

    '''
    Utility Function to perform frequency encoding for a feature.
    '''

    for f in features:
        value_count_dict = frequency_encoder_dict[f]
        name = f+'_FE'        
        testdata[name] = [value_count_dict.get(val, -1) for val in testdata[f].values]
        del value_count_dict
        gc.collect()

    return testdata
        
    
        
def feature_aggregation1(features, uids, testdata, feature_aggregation1_dict, aggregations=['mean']):
    
    '''
      Utility Function to perform aggregation of a given feature with uid for given statistic.
    '''

    for f in features:  
        for uid in uids:
            for agg_type in aggregations:
                
                temp_df = feature_aggregation1_dict[f][uid][agg_type]
                
                name = f+'_'+uid+'_'+agg_type

                testdata[name] = [temp_df.get(uid, -1) for uid in testdata[uid].values]
                
                del temp_df

                gc.collect()


    return testdata
                
def feature_aggregation2(features, uids, testdata, feature_aggregation2_dict):
    
    '''
    Utility Function to perform Aggregation based on the number of unique values present in a feature.
    '''

    for f in features:  
        for uid in uids:

            mp = feature_aggregation2_dict[f][uid]

            name = uid+'_'+f+'_ct'

            testdata[name] = [mp.get(uid, -1) for uid in testdata[uid].values]

            del mp

            gc.collect()

    return testdata
        

def advanced_feature_engineering(testdata):

    with open(mapping_path+'frequency_encoder_dict.pkl', 'rb') as handle:
        frequency_encoder_dict = pickle.load(handle)

    with open(mapping_path+'feature_aggregation1_dict.pkl', 'rb') as handle:
        feature_aggregation1_dict = pickle.load(handle)

    with open(mapping_path+'feature_aggregation2_dict.pkl', 'rb') as handle:
        feature_aggregation2_dict = pickle.load(handle)
        
    testdata = frequency_encode(testdata,frequency_encoder_dict,['addr1','card1','card2','card3','P_emaildomain'])
    testdata = frequency_encode(testdata,frequency_encoder_dict,['card1_addr1','card1_addr1_P_emaildomain'])

    testdata = feature_aggregation1(['TransactionAmt','D9','D11'],['card1','card1_addr1','card1_addr1_P_emaildomain'],testdata, feature_aggregation1_dict, ['mean','std'])

    START_DATE = datetime.datetime.strptime('2017-11-30', '%Y-%m-%d')
    testdata['DT_M'] = testdata['TransactionDT'].apply(lambda x: (START_DATE + datetime.timedelta(seconds = x)))
    
    testdata = frequency_encode(testdata,frequency_encoder_dict,['uid1', 'uid2'])

    testdata = feature_aggregation1(['TransactionAmt','D4','D9','D10','D15'],['uid1', 'uid2'],testdata, feature_aggregation1_dict,['mean','std'])
    testdata = feature_aggregation1(['C'+str(i) for i in range(1,15) if 'C'+str(i) in testdata.columns],['uid1', 'uid2'],testdata,feature_aggregation1_dict,['mean'])
    testdata = feature_aggregation1(['M'+str(i) for i in range(1,10) if 'M'+str(i) in testdata.columns],['uid1', 'uid2'], testdata, feature_aggregation1_dict,['mean'])
    testdata = feature_aggregation1(['C14'],['uid1', 'uid2'],testdata,feature_aggregation1_dict,['std'])

    testdata = feature_aggregation2(['P_emaildomain','dist1','DT_M','id_02','cents'], ['uid1', 'uid2'],testdata,feature_aggregation2_dict)
    testdata = feature_aggregation2(['V127','V307'],['uid1', 'uid2'],testdata,feature_aggregation2_dict)

    testdata['outsider15'] = (np.abs(testdata.D1-testdata.D15)>3).astype('int8')
    
    testdata.drop(['uid1', 'uid2'], axis=1, inplace=True)
    
    del frequency_encoder_dict, feature_aggregation1_dict, feature_aggregation2_dict
    gc.collect()
    
    return testdata

