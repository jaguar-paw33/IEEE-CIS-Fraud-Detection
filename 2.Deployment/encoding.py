import gc
import pickle

mapping_path = './Mappings/'

def label_encode(testdata):

    with open(mapping_path+'catf.pkl', 'rb') as handle:
        catf = pickle.load(handle)
        
    with open(mapping_path+'label_encoder_dict.pkl', 'rb') as handle:
        label_encoder_mapping_dict = pickle.load(handle)
    
    catf = [f for f in testdata.columns if f in catf]

    testdata[catf]=testdata[catf].fillna('missing')
    
    for f in catf:
        testdata[f] = testdata[f].astype(str)
        mapping = label_encoder_mapping_dict[f]
        testdata[f] = [-1 if mapping.get(v, -1)==-1 else mapping[v] for v in testdata[f].values]
        del mapping
        
    del catf, label_encoder_mapping_dict
    gc.collect()

    return testdata


    