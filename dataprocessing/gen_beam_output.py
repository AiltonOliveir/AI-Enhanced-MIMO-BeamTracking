import numpy as np
import h5py
from utils import getNarrowBandULAMIMOChannel, getDFTOperatedChannel

def process_ep(path):
    number_Tx_antennas = 1
    number_Rx_antennas = 64
    normalizedAntDistance = 0.5
    angleWithArrayNormal = 0 
    numOfInvalidChannels = 0
    beam_index = number_Rx_antennas * number_Tx_antennas
    h5_data = h5py.File(path)
    ray_data = np.array(h5_data.get('allEpisodeData'))
    numScenes = ray_data.shape[0]
    numReceivers = ray_data.shape[1]
    channelOutputs = np.nan * np.ones((numScenes, numReceivers,number_Rx_antennas,number_Tx_antennas), np.complex128)
    beamIndexOutputs = np.nan * np.ones((numScenes, numReceivers),
							np.int8)
    for s in range(numScenes):  # 10
        for r in range(numReceivers):  # 2
            insiteData = ray_data[s, r, :, :]
            numNaNsInThisChannel = sum(np.isnan(insiteData.flatten()))
            if numNaNsInThisChannel == np.prod(insiteData.shape):
                numOfInvalidChannels += 1
                continue  # next Tx / Rx pair
            if numNaNsInThisChannel > 0:
                numMaxRays = insiteData.shape[0]
                for itemp in range(numMaxRays):
                    if sum(np.isnan(insiteData[itemp].flatten())) > 0:
                        insiteData = insiteData[:itemp-1,:] #replace by smaller array without NaN
                        break
            gain_in_dB = insiteData[:, 0]
            #timeOfArrival = insiteData[:, 1]
            #AoD_el = insiteData[:, 2]
            AoD_az = insiteData[:, 3]
            #AoA_el = insiteData[:, 4]
            AoA_az = insiteData[:, 5]
            #isLOSperRay = insiteData[:, 6]
            pathPhases = insiteData[:, 7]
            mimoChannel = getNarrowBandULAMIMOChannel(AoD_az, AoA_az, gain_in_dB, number_Tx_antennas,
                                                            number_Rx_antennas, normalizedAntDistance,
                                                            angleWithArrayNormal,pathPhases)
            equivalentChannel = getDFTOperatedChannel(mimoChannel, number_Tx_antennas, number_Rx_antennas)
            #if UPA:
            '''mimoChannel = getNarrowBandUPAMIMOChannel(AoD_el,AoD_az,AoA_el,AoA_az,
                                                        gain_in_dB,pathPhases,number_Tx_antennasX,
                                                        number_Tx_antennasY, number_Rx_antennasX,
                                                        number_Rx_antennasY,normalizedAntDistance)
            equivalentChannel = getCodebookOperatedChannel(mimoChannel, Wt, Wr)'''
            equivalentChannelMagnitude = np.abs(equivalentChannel)
            beamIndexOutputs[s,r] = int(np.argmax(equivalentChannelMagnitude, axis=None))
            channelOutputs[s,r]=np.abs(equivalentChannel)
    return beamIndexOutputs, channelOutputs
                
if __name__ == '__main__':
    episodes = 250
    output_beam_list = []
    output_channel_list = []

    for ep in range(episodes):
        print("Episode # ", ep)
        beamIndex, channel = process_ep(f'/mnt/data/Datasets/t002/ray_tracing_data_t002/roundbout_mobile_28GHz_e{ep}.hdf5')
        
        output_beam_list.append(beamIndex)
        output_channel_list.append(channel)

    # Convert lists to numpy arrays
    output_beam_matrix = np.stack(output_beam_list,axis=0)
    output_channel_matrix = np.stack(output_channel_list,axis=0)

    # Save the output
    outputFileName = 'beam_output_t002'
    np.savez(f'Channel_array_{outputFileName}.npz', channel_array=output_channel_matrix)
    np.savez(f'{outputFileName}.npz', beam_index_array=output_beam_matrix)

    print('==> Wrote file ' + outputFileName)