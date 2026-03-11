import numpy as np 
import pandas as pd 
from sklearn import preprocessing
import os 
import ast


valid_entities = {
"smap" : ["P-1","S-1","E-1","E-2","E-3","E-4","E-5","E-6","E-7","E-8","E-9","E-10","E-11","E-12","E-13","A-1","D-1","P-2","P-3","D-2","D-3","D-4","A-2","A-3","A-4","G-1","G-2","D-5","D-6","D-7","F-1","P-4","G-3","T-1","T-2","D-8","D-9","F-2","G-4","T-3","D-11","D-12","B-1","G-6","G-7","P-7","R-1","A-5","A-6","A-7","D-13","P-2","A-8","A-9"],
"msl" : ["M-6","M-1","M-2","S-2","P-10","T-4","T-5","F-7","M-3","M-4","M-5","P-15","C-1","C-2","T-12","T-13","F-4","F-5","D-14","T-9","P-14","T-8","P-11","D-15","D-16","M-7","F-8"],
"smd" : ['1-1','1-2','1-3','1-4','1-5','1-6','1-7','1-8', '2-1', '2-2', '2-3', '2-4', '2-5', '2-6', '2-7', '2-8', '2-9', '3-1', '3-2', '3-3', '3-4', '3-5', '3-6', '3-7', '3-8', '3-9', '3-10', '3-11'], 
"ucr" : [109, 110, 111, 112, 119, 120, 121, 122, 123, 124, 125, 126, 163, 164, 165, 166, 178, 179, 180, 182, 183, 192, 193, 194, 195, 196, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 197, 198, 199, 200, 242, 243, 244, 245, 246, 247, 145, 146, 147, 148, 149, 150, 173, 174, 175, 176, 177, 167, 168, 169, 170, 171, 172, 181, 239, 240, 241, 108, 184, 185, 186, 187, 188, 189, 190, 191, 113, 114, 115, 116, 117, 118, 156, 157, 158, 159, 160, 127, 128, 129, 130, 131, 152, 153, 154, 155, 210, 211, 212, 151, 161, 162, 201, 202, 203, 204, 205, 206, 207, 208, 209, 248, 249, 250]
}

ucr_skip = [238, 239, 240, 241, 213, 214,215,216,217,218,219,220,221,222,223,224,225]

ucr_groups = {
1 : [109,110,111,112,119,120,121,122,123,124,125,126,163,164,165,166,178,179,180,182,183,192,193,194,195,196,213,214,215,216,217,218,219,220,221,222,223,224,225,226,227,228,229,230,231,232,233,234,235,236,237,238],
2 : [132,133,134,135,136,137,138,139,140,141,142,143,144,197,198,199,200,242,243,244,245,246,247],
3 : [145,146,147,148,149,150,173,174,175,176,177], 
4 : [167,168,169,170,171,172,181,239,240,241],
5 : [108,184,185,186,187,188,189,190,191],
6 : [113,114,115,116,117,118], 
7 : [156,157,158,159,160], 
8 : [127,128,129,130,131], 
9 : [152,153,154,155,210,211,212], 
10: [151,161,162], 
11: [201,202,203,204,205,206,207,208,209,248,249,250]
}

def read_swat(*args):
    normal = pd.read_csv("datasets/swat/SWaT_Dataset_Normal_v1.csv", header = 1)#, nrows=1000)
    normal = normal.iloc[21600:]
    normal = normal.drop([' Timestamp', "Normal/Attack"], axis = 1)
    normal = normal.astype(float)
    
    attack = pd.read_csv("datasets/swat/SWaT_Dataset_Attack_v0.csv")#, nrows=1000)
    # attack.head(5)
    labels = attack[attack.columns[-1]]
    labels = labels.replace(["Normal", "Attack", "A ttack"], [0,1,1]).values

    attack = attack.drop([' Timestamp', "Normal/Attack"] , axis = 1)
    attack = attack.astype(float)
    
    return normal, attack, labels
    
def read_wadi(*args):
    normal = pd.read_csv("datasets/wadi/WADI_14days.csv", sep=',', skip_blank_lines=True)#, nrows=1000)  
    #remove sensor warm-up period 
    normal = normal.iloc[21600:]
    
    # drop the empty columns and the date/time columns
    normal = normal.dropna(axis='columns', how='all').dropna()
    normal = normal.drop(normal.columns[[0,1,2]],axis=1)    
    
    attack = pd.read_csv("datasets/wadi/WADI_attackdataLABLE.csv", sep=',', skip_blank_lines=True, header=1)#, nrows=1000)
    # attack = attack.iloc[2160:]
    
    #replace normal=1, attack=-1 with 0, 1 
    labels = attack[attack.columns[-1]]
    labels = labels.replace([1, -1], [0,1]).values
    
    attack = attack.dropna(axis='columns', how='all').dropna()
    attack = attack.drop(attack.columns[[0,1,2,-1]],axis=1) 
    attack = attack.astype(float)
    
    return normal, attack, labels      

def read_smap(entity = "P-1"):
    if entity == None:
        entity = "P-1"
    print("Getting entity " + entity)
    labels_df = pd.read_csv("datasets/smap-msl/labeled_anomalies.csv", header=0, sep=",")

    #sanity check 
    assert entity in labels_df['chan_id'].values, "Entity " + entity + " not found" 
    assert entity in valid_entities['smap'], entity + " is not part of the SMAP dataset" 

    normal = np.load("datasets/smap-msl/train/" + entity + ".npy")    
    attack = np.load("datasets/smap-msl/test/" + entity + ".npy")
    
    labels = np.zeros(attack.shape[0])
    attack_labels = labels_df[labels_df["chan_id"] == entity]["anomaly_sequences"].values[0]
    attack_labels = ast.literal_eval(attack_labels)
    
    for a in attack_labels:
        labels[a[0]:(a[1] + 1)] = 1

    return pd.DataFrame(normal), pd.DataFrame(attack), labels

def read_msl(entity = "M-6"):
    if entity == None:
        entity = "M-6"
        
    print("Getting entity " + entity)
    labels_df = pd.read_csv("datasets/smap-msl/labeled_anomalies.csv", header=0, sep=",")

    #sanity check 
    assert entity in labels_df['chan_id'].values, "Entity " + entity + " not found" 
    assert entity in valid_entities['msl'], entity + " is not part of the MSL dataset" 

    normal = np.load("datasets/smap-msl/train/" + entity + ".npy")    
    attack = np.load("datasets/smap-msl/test/" + entity + ".npy")
    
    labels = np.zeros(attack.shape[0])
    attack_labels = labels_df[labels_df["chan_id"] == entity]["anomaly_sequences"].values[0]
    attack_labels = ast.literal_eval(attack_labels)
    
    for a in attack_labels:
        labels[a[0]:(a[1] + 1)] = 1

    return pd.DataFrame(normal), pd.DataFrame(attack), labels


def read_ucr(entity = 1):
    if entity == None:
        entity = 1
        
    print("Getting entity " + str(entity))
    
    #sanity check 
    assert entity in valid_entities['ucr'], entity + " is not part of the UCR dataset" 
    
    all_files = os.listdir("datasets/ucr/")
    all_files.sort()
    dataset = "datasets/ucr/" + all_files[entity]

    # 012_UCR_Anomaly_tiltAPB1_100000_114283_114350.txt
    # [0] [1] [2]     [3]      [4]    [5]    [6]   .txt
    split = all_files[entity].replace(".txt", "").split("_")

    normal_end = int(split[4])
    anom_start = int(split[5]) - int(split[4])
    anom_end = int(split[6]) - int(split[4])

    special_files =  ["204", "205", "206", "207", "208", "225", "226", "242", "243"]
    
    with open(dataset) as file:             
        if split[0] in special_files:
            print("special")
            data = [float(x) for x in filter(None, file.read().split("  "))]
        else:
            data = [float(x) for x in filter(None, file.read().split("\n"))]  
        # if len(data[0].split("  ")) > 1:
        #     print("Uncaught Special File: " + entity)
        #     assert False
            
    data = pd.DataFrame(data)
            
    normal = data[0:normal_end].astype(float)
    attack = data[normal_end:].astype(float)

    labels = np.zeros(attack.shape[0])
    if anom_start == anom_end:
        labels[anom_start] = 1 
    
    labels[anom_start:anom_end] = 1
    
    return pd.DataFrame(normal), pd.DataFrame(attack), labels

def read_smd(entity = "1-1"):
    if entity == None:
        entity = '1-1'
        
    print("Getting entity " + entity)
    #sanity check 
    assert entity in valid_entities['smd'], entity + " is not part of the SMD dataset" 

    normal = pd.read_csv("datasets/smd/train/machine-" + entity + ".txt", header=None)
    attack = pd.read_csv("datasets/smd/test/machine-" + entity + ".txt", header=None)
    labels = pd.read_csv("datasets/smd/test_label/machine-" + entity + ".txt", header=None)
    
    labels = labels.values.squeeze(1)
    
    return normal, attack, labels

get_datasets = {
"swat": read_swat,
"wadi": read_wadi,
"smap": read_smap,
"msl" : read_msl,
"ucr" : read_ucr,
"smd" : read_smd,
}

def get_data(dataset, val_split=0.2, seq_len=12, down_rate=5, entity = None, verbose = False):
    print("Reading " + dataset)
    if entity: 
        normal, attack, labels = get_datasets[dataset](entity)
    else:
        normal, attack, labels = get_datasets[dataset]()           
  
    if down_rate > 1:
        #Downsampling
        normal=normal.groupby(np.arange(len(normal.index)) // down_rate).mean()
        attack=attack.groupby(np.arange(len(attack.index)) // down_rate).mean()
        labels = pd.DataFrame(labels)    
        labels = labels.groupby(np.arange(len(labels.index)) // down_rate).max()   
        labels = labels.values.flatten()
    else:        
        labels = np.array(labels)
        

    #Normalizing     
    min_max_scaler = preprocessing.MinMaxScaler()
    normal = min_max_scaler.fit_transform(normal.values)
    attack = min_max_scaler.transform(attack.values)   
    

    split = int(len(normal) * (1-val_split))
    validate = normal[split:]
    normal = normal[:split]
    
    if verbose:
        print(normal.shape)
        print(validate.shape)
        print(attack.shape)
        print(labels.shape)

    return normal, validate, attack, labels
