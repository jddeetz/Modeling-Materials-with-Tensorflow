#Import MPRester
from pymatgen import MPRester
import sqlite3

#Private key for Materials Project database (MP-DB)
mpr = MPRester("")

#Open database
conn = sqlite3.connect('mpdata.sqlite')#connect to db or create
cur = conn.cursor()#database handle

#Create SQL tables from outermost to innermost
#Set up tables from outermost to innermost
cur.executescript('''
CREATE TABLE IF NOT EXISTS SpaceGroup (
    id     INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE,
    symbol   TEXT UNIQUE,
    number   TEXT,
    point_group   TEXT,
    crystal_system TEXT,
    hall   TEXT
);

CREATE TABLE IF NOT EXISTS Material (
    id     INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE,
    material_id   TEXT UNIQUE,
    density  FLOAT,
    element1  TEXT,
    element1_num  INTEGER,
    element2  TEXT,
    element2_num  INTEGER,
    spacegroup_id    INTEGER   
);''')

#List of elements to be considered for alloys
alloyelements=["Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Hf","Ta","W","Re","Os","Ir","Pt","Au"]

#Query MP-DB for all materials ids of binary alloys of transitional metals. Returns a list of dictionaries.
entries = mpr.query({"elements":{"$in":alloyelements}, "nelements":2}, ["material_id","density","unit_cell_formula","pretty_formula","spacegroup","warnings"])

#Print number of entries in resulting from query 
print('The query returned',len(entries),'entries')

#Store data in SQL database
num_sql_entries=0
for entry in entries:
    #If there is a warning, do not store the data.
    if len(entry["warnings"])>0: continue
    
    #Convert unit_cell_formula dictionary into list of items
    elements=[]
    for key, value in entry["unit_cell_formula"].items():
        temp = [key,value]
        elements.append(temp)

    #Skip entries where both items are not from elements list
    if elements[0][0] in alloyelements and elements[1][0] in alloyelements:

        #Write SpaceGroup to the database
        space=entry["spacegroup"]
        #Insert space group data
        cur.execute('''INSERT OR IGNORE INTO SpaceGroup (symbol,number,point_group,crystal_system,hall)
        VALUES ( ?,?,?,?,? )''', (space["symbol"],space["number"],space["point_group"],space["crystal_system"],space["hall"]) )#
        #Fetch spacegroup_id
        cur.execute('SELECT id FROM SpaceGroup WHERE symbol = ? ', (space["symbol"], ))
        spacegroup_id = cur.fetchone()[0]

        #Write mp entry into database
        cur.execute('''INSERT OR REPLACE INTO Material
        (material_id, density, element1, element1_num,element2, element2_num,spacegroup_id) VALUES (?,?,?,?,?,?,?)''',
        ( entry["material_id"], entry["density"], elements[0][0], elements[0][1], elements[1][0], elements[1][1],spacegroup_id) )
        
        num_sql_entries+=1

conn.commit()
print(num_sql_entries,'records were added to','mpdata.sqlite')



