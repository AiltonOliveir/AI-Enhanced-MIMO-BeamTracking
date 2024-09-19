"""
Splits Position Matrix data and generate datastructure for datagenerators
"""

import os
import numpy as np
import pandas as pd
from termcolor import colored

def save_npz_from_coord_df(folder_name, dic, output_dict, processed_path, raw_path, tx_id, channel_dict):
    coord_path_save = os.path.join(processed_path, 'coord_matrix', folder_name)
    if not os.path.exists(coord_path_save):
        os.makedirs(coord_path_save)
    coord_path_list = []
    data = np.load(raw_path + f'{tx_id}.npz')['position_matrix_array']
    # Iterates over all Valid Channels
    start = 0
    end = start + 10
    for ep in dic.keys():
        print('Processing ', colored(folder_name, 'green'),' Ep: ', colored(f'{ep:5d}', 'red'), ' Tx: ', colored(f'{tx:2d}', 'red'))

        # Reads the specific coord data
        Vehs = dic[ep]
        for veh in Vehs:
            # Get the coord from ep and Veh
            coord_data = data[start:end, 0, veh, :]
            coord_data[coord_data>1] = 1 # make it binary
            coord_data = coord_data.astype(np.int8)
            file_path = os.path.join(coord_path_save, f"tx_{tx_id}_ep_{ep}_veh_{veh}.npz")
            coord_path_list.append(file_path)
            
            beam = output_dict[ep][:,veh]
            channel = channel_dict[ep][:,veh,:]

            # Save coord as an Input
            np.savez_compressed(file_path, spatial_input=coord_data, beam_index=beam, channel_matrix=channel)
        start += 10
        end = start + 10
    
    return coord_path_list

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
    csv_file = '/mnt/data/Datasets/t001/CoordVehiclesRxPerScene_tx'
    splits_pc = [0.7, 0.2, 0.1]
    raw_path = '/mnt/data/Datasets/t001/Coord/gradient_matrix_positions_t001_tx'
    processed_path = '/mnt/data/Datasets/t001/baseline/'
    beam_output_path = '/mnt/data/Datasets/t001/Beams/beam_output_txTX_t001.npz'
    channel_output_path = '/mnt/data/Datasets/t001/Beams/Channel_array_beam_output_txTX_t001.npz'
    ####################################
    if not os.path.exists(processed_path):
        os.makedirs(processed_path)

    df_list_train = []
    df_list_val = []
    df_list_test = []
    for tx in range(1,3):
        # # Read csv
        coord_df = pd.read_csv(csv_file +f'{tx}.csv')
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
    
        # Adds output splitted
        train_output = read_and_split_data(beam_output_path, train_dict, [0,ep_split[0]], tx, 'beam_index_array', np.uint8)
        val_output = read_and_split_data(beam_output_path, val_dict, ep_split[0:2], tx, 'beam_index_array', np.uint8)
        test_output = read_and_split_data(beam_output_path, test_dict, [ep_split[1], ep_split[2]+1], tx, 'beam_index_array', np.uint8)

        train_channel = read_and_split_data(channel_output_path, train_dict, [0,ep_split[0]], tx, 'channel_array', np.float32)
        val_channel = read_and_split_data(channel_output_path, val_dict, ep_split[0:2], tx, 'channel_array', np.float32)
        test_channel = read_and_split_data(channel_output_path, test_dict, [ep_split[1], ep_split[2]+1], tx, 'channel_array', np.float32)
        
        # Gen splitted data structure
        coord_path_list = save_npz_from_coord_df('train', train_dict, train_output, processed_path, raw_path, tx, train_channel)
        # Gen csv with the path to npz input and the output label
        data_structure_df = pd.DataFrame()
        data_structure_df['input_path'] = coord_path_list
        data_structure_df['TxId'] = tx
        df_list_train.append(data_structure_df)
        
        coord_path_list = save_npz_from_coord_df('val', val_dict, val_output, processed_path, raw_path, tx, val_channel)
        data_structure_df = pd.DataFrame()
        data_structure_df['input_path'] = coord_path_list
        data_structure_df['TxId'] = tx
        df_list_val.append(data_structure_df)

        coord_path_list = save_npz_from_coord_df('test', test_dict, test_output, processed_path, raw_path, tx, test_channel)
        data_structure_df = pd.DataFrame()
        data_structure_df['input_path'] = coord_path_list
        data_structure_df['TxId'] = tx
        df_list_test.append(data_structure_df)
    final_df = pd.concat(df_list_train, ignore_index=True)
    final_df.to_csv(os.path.join(processed_path, "train_coord.csv"), index=False)

    final_df = pd.concat(df_list_val, ignore_index=True)
    final_df.to_csv(os.path.join(processed_path, "val_coord.csv"), index=False)
    
    final_df = pd.concat(df_list_test, ignore_index=True)
    final_df.to_csv(os.path.join(processed_path, "test_coord.csv"), index=False)
