"""
Adapted from https://github.com/kxhit/pointnetvlad/blob/master/submap_generation/KITTI/gen_gt.py
Generate triplets (query, positive, negative) from KITTI dataset
Author: Xin Kong
Contact: xinkong@zju.edu.cn
Date: Oct 2019

and https://github.com/mikacuy/pointnetvlad/blob/master/generating_queries/generate_test_sets.py
for Oxford Dataset
"""

import os
import numpy as np
from tqdm import tqdm
import random
import pickle
import pandas as pd
from sklearn.neighbors import KDTree


def get_dataset(sequence_id='02', basedir= '/home/cel/data/kitti'):
    return pykitti.odometry(basedir, sequence_id)


def p_dist(pose1, pose2, threshold=3, print_bool=False):
    dist = np.linalg.norm(pose1[:,-1]-pose2[:,-1])    # xyz
    if print_bool==True:
        print(dist)
    if abs(dist) <= threshold:
        return True
    else:
        return False


def t_dist(t1, t2, threshold=10):
    if abs((t1-t2).total_seconds()) >= threshold:
        return True
    else:
        return False


def get_triplets_dict(seqs, output_dir, d_thresh, t_thresh):
    triplets = {}
    triplets_all = {}

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for seq in tqdm(seqs):
        dataset = get_dataset(seq)

        if seq not in triplets:
            triplets[seq] = {}
            triplets_all[seq] = {}

        #  we do not search for loop candidates in the past 50 scans to avoid detecting loops in nearby scans.
        for t1 in range(len(dataset.timestamps)):
            positives = []
            negatives = []
            distances = []
            for t2 in range(max(t1 - t_thresh, 0)):
                if p_dist(dataset.poses[t1], dataset.poses[t2], d_thresh):
                    positives.append(t2)
                    distances.append(abs(np.linalg.norm(dataset.poses[t1][:,-1]-dataset.poses[t2][:,-1])))
                elif not p_dist(dataset.poses[t1], dataset.poses[t2], 20): # with stricker distance threshold for negatives, TODO: 20?
                    negatives.append(t2)
            if len(positives) > 0:
                # triplets stores the closest frame from positive list, and randomly chooses frame from negative list
                # triplets_all stores all positive frames and all negative frames 
                triplets[seq][t1] = {'anchor': t1, 'positives': positives[distances.index(min(distances))], 'negatives': random.choice(negatives)}
                triplets_all[seq][t1] = {'anchor': t1, 'positives': positives, 'negatives': negatives}

    with open('{}/triplet_D-{}_T-{}.json'.format(output_dir, d_thresh, t_thresh), 'w') as f:
        json.dump(triplets, f)
    with open('{}/triplet_all_D-{}_T-{}.json'.format(output_dir, d_thresh, t_thresh), 'w') as f:
        json.dump(triplets_all, f)

    return triplets


def generate_training_tuples_baseline(base_path):
    """
    Code taken from https://github.com/mikacuy/pointnetvlad/blob/master/generating_quesries/generate_training_tuples_baseline.py
    """
    
    env = ['abandonedfactory']
    train_seq = 
    environment_list = [env_name for env_name in os.listdir(dataset_root_folder) if os.path.isdir(os.path.join(dataset_root_folder, env_name))]
    


    all_folders=sorted(os.listdir(os.path.join(base_path, runs_folder)))
    folders = []

    #All runs are used for training (both full and partial)
    index_list=range(len(all_folders)-1)
    print("Number of runs: "+str(len(index_list)))
    for index in index_list[:3]:
        folders.append(all_folders[index])
    print(folders)

    #####For training and test data split#####
    x_width=150
    y_width=150
    p1=[5735712.768124,620084.402381]
    p2=[5735611.299219,620540.270327]
    p3=[5735237.358209,620543.094379]
    p4=[5734749.303802,619932.693364]   
    p=[p1,p2,p3,p4]


    def check_in_test_set(northing, easting, points, x_width, y_width):
        in_test_set=False
        for point in points:
                if(point[0]-x_width<northing and northing< point[0]+x_width and point[1]-y_width<easting and easting<point[1]+y_width):
                        in_test_set=True
                        break
        return in_test_set
    ##########################################


    def construct_query_dict(df_centroids, filename):
        tree = KDTree(df_centroids[['northing','easting']])
        ind_nn = tree.query_radius(df_centroids[['northing','easting']],r=10)
        ind_r = tree.query_radius(df_centroids[['northing','easting']], r=50)
        queries={}
        for i in range(len(ind_nn)):
                query=df_centroids.iloc[i]["file"]
                positives=np.setdiff1d(ind_nn[i],[i]).tolist()
                negatives=np.setdiff1d(df_centroids.index.values.tolist(),ind_r[i]).tolist()
                random.shuffle(negatives)
                queries[i]={"query":query,"positives":positives,"negatives":negatives}

        with open(filename, 'wb') as handle:
            pickle.dump(queries, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print("Done ", filename)


    ####Initialize pandas DataFrame
    df_train= pd.DataFrame(columns=['file','northing','easting'])
    df_test= pd.DataFrame(columns=['file','northing','easting'])

    for folder in folders:
        df_locations= pd.read_csv(os.path.join(base_path,runs_folder,folder,filename),sep=',')
        df_locations['timestamp']=runs_folder+folder+pointcloud_fols+df_locations['timestamp'].astype(str)+'.bin'
        df_locations=df_locations.rename(columns={'timestamp':'file'})
        
        for index, row in df_locations.iterrows():
                if(check_in_test_set(row['northing'], row['easting'], p, x_width, y_width)):
                        df_test=df_test.append(row, ignore_index=True)
                else:
                        df_train=df_train.append(row, ignore_index=True)

    print("Number of training submaps: "+str(len(df_train['file'])))
    print("Number of non-disjoint test submaps: "+str(len(df_test['file'])))
    construct_query_dict(df_train, os.path.join(base_path, "training_queries_baseline_3seq.pickle"))
    construct_query_dict(df_test, os.path.join(base_path, "test_queries_baseline_3seq.pickle"))


def generate_test_sets():
    """
    Code taken from https://github.com/mikacuy/pointnetvlad/blob/master/generating_queries/generate_test_sets.py
    """
    #####For training and test data split#####
    x_width=150
    y_width=150

    #For Oxford
    p1=[5735712.768124,620084.402381]
    p2=[5735611.299219,620540.270327]
    p3=[5735237.358209,620543.094379]
    p4=[5734749.303802,619932.693364]   

    #For University Sector
    p5=[363621.292362,142864.19756]
    p6=[364788.795462,143125.746609]
    p7=[363597.507711,144011.414174]

    #For Residential Area
    p8=[360895.486453,144999.915143]
    p9=[362357.024536,144894.825301]
    p10=[361368.907155,145209.663042]

    p_dict={"oxford":[p1,p2,p3,p4], "university":[p5,p6,p7], "residential": [p8,p9,p10], "business":[]}

    def check_in_test_set(northing, easting, points, x_width, y_width):
        in_test_set=False
        for point in points:
            if(point[0]-x_width<northing and northing< point[0]+x_width and point[1]-y_width<easting and easting<point[1]+y_width):
                in_test_set=True
                break
        return in_test_set
    ##########################################

    def output_to_file(output, filename):
        with open(filename, 'wb') as handle:
            pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("Done ", filename)


    def construct_query_and_database_sets(base_path, runs_folder, folders, pointcloud_fols, filename, p, output_name):
        database_trees=[]
        test_trees=[]
        for folder in folders:
            print(folder)
            df_database= pd.DataFrame(columns=['file','northing','easting'])
            df_test= pd.DataFrame(columns=['file','northing','easting'])
            
            df_locations= pd.read_csv(os.path.join(base_path,runs_folder,folder,filename),sep=',')
            # df_locations['timestamp']=runs_folder+folder+pointcloud_fols+df_locations['timestamp'].astype(str)+'.bin'
            # df_locations=df_locations.rename(columns={'timestamp':'file'})
            for index, row in df_locations.iterrows():
                #entire business district is in the test set
                if(output_name=="business"):
                    df_test=df_test.append(row, ignore_index=True)
                elif(check_in_test_set(row['northing'], row['easting'], p, x_width, y_width)):
                    df_test=df_test.append(row, ignore_index=True)
                df_database=df_database.append(row, ignore_index=True)

            database_tree = KDTree(df_database[['northing','easting']])
            test_tree = KDTree(df_test[['northing','easting']])
            database_trees.append(database_tree)
            test_trees.append(test_tree)

        test_sets=[]
        database_sets=[]
        for folder in folders:
            database={}
            test={} 
            df_locations= pd.read_csv(os.path.join(base_path,runs_folder,folder,filename),sep=',')
            df_locations['timestamp']=runs_folder+folder+pointcloud_fols+df_locations['timestamp'].astype(str)+'.bin'
            df_locations=df_locations.rename(columns={'timestamp':'file'})
            for index,row in df_locations.iterrows():				
                #entire business district is in the test set
                if(output_name=="business"):
                    test[len(test.keys())]={'query':row['file'],'northing':row['northing'],'easting':row['easting']}
                elif(check_in_test_set(row['northing'], row['easting'], p, x_width, y_width)):
                    test[len(test.keys())]={'query':row['file'],'northing':row['northing'],'easting':row['easting']}
                database[len(database.keys())]={'query':row['file'],'northing':row['northing'],'easting':row['easting']}
            database_sets.append(database)
            test_sets.append(test)		

        for i in range(len(database_sets)):
            tree=database_trees[i]
            for j in range(len(test_sets)):
                if(i==j):
                    continue
                for key in range(len(test_sets[j].keys())):
                    coor=np.array([[test_sets[j][key]["northing"],test_sets[j][key]["easting"]]])
                    index = tree.query_radius(coor, r=25)
                    #indices of the positive matches in database i of each query (key) in test set j
                    test_sets[j][key][i]=index[0].tolist()

        output_to_file(database_sets, os.path.join(base_path, output_name+'_evaluation_database.pickle'))
        output_to_file(test_sets, os.path.join(base_path, output_name+'_evaluation_query.pickle'))

    ###Building database and query files for evaluation
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))+"/../"
    base_path= "../../data/"

    #For Oxford
    folders=[]
    runs_folder = "oxford/"
    all_folders=sorted(os.listdir(os.path.join(BASE_DIR,base_path,runs_folder)))
    index_list=[5,6,7,9,10,11,12,13,14,15,16,17,18,19,22,24,31,32,33,38,39,43,44]
    print(len(index_list))
    for index in index_list:
        folders.append(all_folders[index])

    print(folders)
    construct_query_and_database_sets(base_path, runs_folder, folders, "/pointcloud_20m/", "pointcloud_locations_20m.csv", p_dict["oxford"], "oxford")


def main():
    dataset_root_folder = '/home/cel/DockerFolder/data/tartanair/'

    # generate tuplets for training
    generate_training_tuples_baseline(dataset_root_folder)
    # generate tuplets for testing
    generate_test_sets(dataset_root_folder)

if __name__ == '__main__':
    main()
    

