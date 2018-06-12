import sqlite3
import numpy as np
import random
import pickle

#Normally, if we have more than 10k training examples for neural networks, we would use a training, test, and validation set.
#We only have 612 examples here, so we will split the database into 20% validation and 80% training
fraction_split=0.2

# Connect to the database
conn = sqlite3.connect('mpdata.sqlite')#connect to db or create
cur = conn.cursor()#database handle

#Grab materials handles, densities, compositions, and space groups of binary alloys from database
cur.execute('''SELECT Material.material_id,Material.density,Material.element1,Material.element1_num,Material.element2,Material.element2_num,SpaceGroup.crystal_system 
FROM Material JOIN SpaceGroup ON SpaceGroup.id = Material.spacegroup_id''')
results=cur.fetchall()

#Change list of tuples into list of lists
tmp=list()
for result in results:
    tmp.append(list(result))
results=tmp

#Numeralize element labels into integers
alloyelements=["Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Hf","Ta","W","Re","Os","Ir","Pt","Au"]
for i in range(len(results)):
    results[i][2]=alloyelements.index(results[i][2])
    results[i][4]=alloyelements.index(results[i][4])

labels=list()
y=np.array([])
X=np.zeros([len(results),len(alloyelements)+7])#Each material represented by a fingerprint of its elements and crystal system

m=0
for result in results:
    #Formalize independent (X) and dependent (y) variables, as well as training labels (labels)
    labels.append(result[0])
    y=np.append(y,result[1])

    #Rather than having two input nodes representing atomic numbers, it is  more advantageous to use several input nodes indicating the presence of these elements
    #For example, for the reduced set Sc, Ti, V, Cr, and the hypothetical alloy ScV, it may be less effective to numeralize this as [0 2]
    #It is better to instead one-hot encode the alloy as [1 0 1 0] in order to keep each node in the network more significant.
    X[m,result[2]-1]=result[3]
    X[m,result[4]-1]=result[5]
    if result[6]=="tetragonal":
        X[m,len(alloyelements)]=1
    elif result[6]=="cubic":
        X[m,len(alloyelements)+1]=1
    elif result[6]=="hexagonal":
        X[m,len(alloyelements)+2]=1
    elif result[6]=="monoclinic":
        X[m,len(alloyelements)+3]=1
    elif result[6]=="orthorhombic":
        X[m,len(alloyelements)+4]=1
    elif result[6]=="trigonal":
        X[m,len(alloyelements)+5]=1
    elif result[6]=="triclinic":
        X[m,len(alloyelements)+6]=1
    print(X[m])
    m+=1
    
#Split data into training set and validation set.
labels_train=list()
labels_valid=list()
y_train=np.array([])
y_valid=np.array([])
X_train=np.array([])
X_valid=np.array([])

for i in range(m):
    if random.random()<fraction_split: #put into the validation set
        labels_valid.append(labels[i])
        y_valid=np.append(y_valid,y[i])
        X_valid=np.append(X_valid,X[i])
    else:
        labels_train.append(labels[i])
        y_train=np.append(y_train,y[i])
        X_train=np.append(X_train,X[i])

X_train=X_train.reshape([len(labels_train),len(alloyelements)+7])
X_valid=X_valid.reshape([len(labels_valid),len(alloyelements)+7])

#Store data in pkl file
data_store=[X_train,y_train,labels_train,X_valid,y_valid,labels_valid,alloyelements]
pickle.dump(data_store, open("data.pkl",'wb') )

