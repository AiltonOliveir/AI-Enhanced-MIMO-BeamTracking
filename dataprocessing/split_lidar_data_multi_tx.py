"""
Splits Lidar and generate datastructure for datagenerators
"""

import os
import numpy as np
import pandas as pd
from termcolor import colored

def save_npz_from_lidar_df(folder_name, dic, output_dict, processed_path, raw_path, tx, channel_dict):
    lidar_path_save = os.path.join(processed_path, 'lidar', folder_name)
    if not os.path.exists(lidar_path_save):
        os.makedirs(lidar_path_save)
    lidar_path_list = []
    # Iterates over all Valid Channels
    for ep in dic.keys():
        print('Processing ', colored(folder_name, 'green'),' Ep: ', colored(f'{ep:5d}', 'red'), ' Tx: ', colored(f'{tx:2d}', 'red'))
        data = np.load(raw_path.replace('EP', str(ep)))['obstacles_matrix_array']

        # Reads the specific lidar data
        Vehs = dic[ep]
        for veh in Vehs:
            # Get the lidar from ep and Veh
            lidar_path = os.path.join(lidar_path_save, f"tx_{tx}_ep_{ep}_veh_{veh}.npz")
            lidar_path_list.append(lidar_path)
            
            lidar_data = data[:, veh, :]
            lidar_data[lidar_data>1] = 1 # make it binary
            lidar_data = lidar_data.astype(np.int8)
            
            beam = output_dict[ep][:,veh]
            channel = channel_dict[ep][:,veh,:]

            # Save Lidar as an Input
            np.savez_compressed(lidar_path, spatial_input=lidar_data, beam_index=beam, channel_matrix=channel)
            
    # Gen csv with the path to npz input and the output label
    output_df = pd.DataFrame({'input_path':lidar_path_list})
    
    datagenerator_csv = os.path.join(processed_path, f"{folder_name}_lidar.csv")
    if os.path.exists(datagenerator_csv):
        datagenerator_df = pd.read_csv(datagenerator_csv)
        output_df = pd.concat([datagenerator_df, output_df], axis=0, ignore_index=True)
    
    output_df.to_csv(datagenerator_csv, index=False)

def read_and_split_data(path_file, dic, split, tx, array_name, output_type = np.int8):
    all_data = np.load(path_file.replace('TX', str(tx)))[array_name]
    output = {}
    for ep in dic.keys():
        if (ep<split[0]) and (ep>=split[1]):
            continue
        output[ep] = all_data[ep, :].astype(dtype = output_type)
    return output

if __name__ == "__main__":
    ####################################
    # Files Name, change the values here
    csv_file = '/mnt/data/Datasets/t001/CoordVehiclesRxPerScene_txTX.csv'
    splits_pc = [0.7, 0.2, 0.1]
    raw_data_path = '/mnt/data/Datasets/t001/Lidar/obstacles_txTX_3D/'
    processed_path = '/mnt/data/Datasets/t001/baseline/'
    beam_output_path = '/mnt/data/Datasets/t001/Beams/64tx/beam_output_txTX_t001.npz'
    channel_output_path = '/mnt/data/Datasets/t001/Beams/Channel_array_beam_output_txTX_t001.npz'
    tx_antennas = 2
    ####################################
    if not os.path.exists(processed_path):
        os.makedirs(processed_path)
        
    for tx in range(1, tx_antennas+1):
        print(colored(f"Processing Transmitter {tx}", color='green'))
        raw_path = os.path.join(raw_data_path.replace('TX', str(tx)), "obstacles_e_EP.npz")

        # Read csv
        coord_df = pd.read_csv(csv_file.replace('TX', str(tx)))

        # Filter valid Episodes
        coord_df = coord_df[coord_df['Val']=='V']

        # Sort values to Follow one receiver per time
        coord_df = coord_df.sort_values(by=['EpisodeID', 'VehicleArrayID', 'SceneID'], ascending=[True, True, True])

        # Filter only the complete 10 scenes vehicles
        df_episodes = coord_df.groupby('EpisodeID')
        # Iterate over each group
        trk_eps = {} # keys(episode): veh_id | to read the valid veh scenes
        for episode_id, group in df_episodes:
            # Check if the episode has at least 10 scenes
            if len(group['SceneID'].unique()) >= 10:
                # Get the vehicle_ids that remain in the episode for all scenes
                valid_vehicle_ids = group.groupby('VehicleArrayID').filter(lambda x: len(x['SceneID'].unique()) == len(group['SceneID'].unique()))
                
                # If there are valid vehicles for this episode, add the episode_id to the list
                if not valid_vehicle_ids.empty:
                    veh_id = valid_vehicle_ids['VehicleArrayID'].unique()
                    trk_eps[episode_id] = veh_id

        # Generate Splits for train, val, test
        n_scenes = np.max(coord_df['SceneID'])+1
        episodes = np.array(list(trk_eps.keys()))
        n_ep = len(episodes)
        ep_split = episodes[(np.cumsum(splits_pc)*n_ep).astype(int)]
        
        del coord_df # Delete to save memory

        # Split the csv data
        train_dict = dict(filter(lambda item: item[0] < ep_split[0], trk_eps.items()))
        val_dict = dict(filter(lambda item: (item[0] >= ep_split[0]) & (item[0] < ep_split[1]), trk_eps.items()))
        test_dict = dict(filter(lambda item: (item[0] >= ep_split[1]) & (item[0] <= ep_split[2]), trk_eps.items()))

        train_output = read_and_split_data(beam_output_path, train_dict, [0,ep_split[0]], tx, 'beam_index_array', np.uint8)
        val_output = read_and_split_data(beam_output_path, val_dict, ep_split[0:2], tx, 'beam_index_array', np.uint8)
        test_output = read_and_split_data(beam_output_path, test_dict, [ep_split[1], ep_split[2]+1], tx, 'beam_index_array', np.uint8)

        train_channel = read_and_split_data(channel_output_path, train_dict, [0,ep_split[0]], tx, 'channel_array', np.float32)
        val_channel = read_and_split_data(channel_output_path, val_dict, ep_split[0:2], tx, 'channel_array', np.float32)
        test_channel = read_and_split_data(channel_output_path, test_dict, [ep_split[1], ep_split[2]+1], tx, 'channel_array', np.float32)
        
        # Gen splitted data structure
        save_npz_from_lidar_df('train', train_dict, train_output, processed_path, raw_path, tx, train_channel)
        save_npz_from_lidar_df('val', val_dict, val_output, processed_path, raw_path, tx, val_channel)
        save_npz_from_lidar_df('test', test_dict, test_output, processed_path, raw_path, tx, test_channel)
