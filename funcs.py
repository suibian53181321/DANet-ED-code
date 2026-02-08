import numpy as np
from scipy.signal import welch
import networkx as nx
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import re
from scipy.signal import spectrogram, butter, filtfilt
import scipy.io as scio
import lmdb
from scipy.stats import differential_entropy
import pandas as pd
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split




def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b,a = butter(order, [low, high], btype='band')
    return b,a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b,a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b,a, data)
    return y


def filter_eeg_data_single_band(load_dir):            #return filtered eeg data and corresponding labels
    EEG_REGULAR_CHANNELS = 68
    EEG_CHANNEL_INDICES = np.arange(0,EEG_REGULAR_CHANNELS)
    fs = 250.0  # 采样频率 (Hz)
    lowcut = 4.0
    highcut = 30.0

    subject_number = 0
    files = os.listdir(load_dir)
    for file in files:
        file_path = os.path.join(load_dir, file)
        if os.path.isfile(file_path) and file.endswith('.mat'):
            subject_number += 1

    filtered_eeg_data_and_labels_and_type = []
    eeg_data_subject_ids = []

    for file in files:
        file_path = os.path.join(load_dir, file)
        if os.path.isfile(file_path) and file.endswith('.mat'):
            data = scio.loadmat(file_path)
            # eeg_key = [key for key in data.keys() if re.search(r'\d{8}', key)][0]
            # print(eeg_key)
            # sub_id = eeg_key
            # eeg_signals_raw = data[sub_id]
            eeg_signals_raw = data['eeg_data']
            channel_num = data['channel_numbers']
            if channel_num == EEG_REGULAR_CHANNELS:
                eeg_data_selected = eeg_signals_raw[EEG_CHANNEL_INDICES, :]
            else :
                channel_append_number = EEG_REGULAR_CHANNELS - int(channel_num)
                original_eeg_data = eeg_signals_raw
                append_channel_data = np.zeros((channel_append_number, eeg_signals_raw.shape[1]))
                eeg_data_selected = np.concatenate([original_eeg_data, append_channel_data])

            filtered_signals = np.zeros_like(eeg_data_selected)
            print(channel_num)
            print(filtered_signals.shape)
            n_eeg_signals = eeg_data_selected.shape[0]
            print("开始滤波...")
            # for j in range(3):
            for i in range(n_eeg_signals):
                # filtered_signals[i, :] = butter_bandpass_filter(eeg_data_selected[i, :], lowcut[0],highcut[2], fs,order=5)
                filtered_signals[i, :] = butter_bandpass_filter(eeg_data_selected[i, :], lowcut, highcut, fs,order=5)
            print("滤波完成.")
            print(f"滤波后信号维度: {filtered_signals.shape}")

            sub_data_and_label_and_type = {}
            sub_data_and_label_and_type['eeg_signals'] = filtered_signals
            sub_id = file[:8]
            sub_data_and_label_and_type['sub_id'] = sub_id

            eeg_data_subject_ids.append(sub_id)

            if (sub_id.startswith('0201')):
                sub_data_and_label_and_type['sub_type'] = 'MD'  # 以0201开头的是MD
            else:
                sub_data_and_label_and_type['sub_type'] = 'HC'  # 否则是HC

            filtered_eeg_data_and_labels_and_type.append(sub_data_and_label_and_type)

    print(filtered_eeg_data_and_labels_and_type[subject_number-1]['sub_id'])
    print(filtered_eeg_data_and_labels_and_type[subject_number-1]['eeg_signals'].shape)
    print(filtered_eeg_data_and_labels_and_type[subject_number-1]['sub_type'])
    print(eeg_data_subject_ids)
    return filtered_eeg_data_and_labels_and_type, eeg_data_subject_ids


def filter_eeg_data_multi_band(load_dir):            #return filtered eeg data and corresponding labels

    EEG_REGULAR_CHANNELS = 68
    EEG_CHANNEL_INDICES = np.arange(0,EEG_REGULAR_CHANNELS)
    fs = 250.0  # 采样频率 (Hz)
    lowcut = [4.0, 8.0, 12.0]  # Hz (调整低频截止，EOG信号可能包含更低频成分)
    highcut = [8.0, 12.0, 30.0]  # Hz (调整高频截止)


    subject_number = 0
    files = os.listdir(load_dir)
    for file in files:
        file_path = os.path.join(load_dir, file)
        if os.path.isfile(file_path) and file.endswith('.mat'):
            subject_number += 1

    filtered_eeg_data_and_labels_and_type = []
    eeg_data_subject_ids = []

    for file in files:
        file_path = os.path.join(load_dir, file)
        if os.path.isfile(file_path) and file.endswith('.mat'):
            data = scio.loadmat(file_path)
            # eeg_key = [key for key in data.keys() if re.search(r'\d{8}', key)][0]
            # print(eeg_key)
            # sub_id = eeg_key
            # eeg_signals_raw = data[sub_id]
            eeg_signals_raw = data['eeg_data']
            channel_num = data['channel_numbers']
            if channel_num == EEG_REGULAR_CHANNELS:
                eeg_data_selected = eeg_signals_raw[EEG_CHANNEL_INDICES, :]
            else :
                channel_append_number = EEG_REGULAR_CHANNELS - int(channel_num)
                original_eeg_data = eeg_signals_raw
                append_channel_data = np.zeros((channel_append_number, eeg_signals_raw.shape[1]))
                eeg_data_selected = np.concatenate([original_eeg_data, append_channel_data])

            filtered_signals = np.zeros((eeg_data_selected.shape[0], eeg_data_selected.shape[1], 3))
            print(channel_num)
            print(filtered_signals.shape)
            n_eeg_signals = eeg_data_selected.shape[0]
            print("开始滤波...")
            for j in range(3):
                for i in range(n_eeg_signals):
                    filtered_signals[i, :, j] = butter_bandpass_filter(eeg_data_selected[i, :], lowcut[j],highcut[j], fs,order=5)
            print("滤波完成.")
            print(f"滤波后信号维度: {filtered_signals.shape}")

            sub_data_and_label_and_type = {}
            sub_data_and_label_and_type['eeg_signals'] = filtered_signals
            sub_id = file[:8]
            sub_data_and_label_and_type['sub_id'] = sub_id

            eeg_data_subject_ids.append(sub_id)

            if (sub_id.startswith('0201')):
                sub_data_and_label_and_type['sub_type'] = 'MD'  # 以0201开头的是MD
            else:
                sub_data_and_label_and_type['sub_type'] = 'HC'  # 否则是HC

            filtered_eeg_data_and_labels_and_type.append(sub_data_and_label_and_type)

    print(filtered_eeg_data_and_labels_and_type[subject_number-1]['sub_id'])
    print(filtered_eeg_data_and_labels_and_type[subject_number-1]['eeg_signals'].shape)
    print(filtered_eeg_data_and_labels_and_type[subject_number-1]['sub_type'])
    print(eeg_data_subject_ids)
    return filtered_eeg_data_and_labels_and_type, eeg_data_subject_ids

def load_embedding_and_matrix(file_dir,lmdb_dir):
    EEG_REGULAR_CHANNELS = 68
    # EEG_REGULAR_CHANNELS = 60
    EEG_CHANNEL_INDICES = np.arange(0, EEG_REGULAR_CHANNELS)
    fs = 250.0  # 采样频率 (Hz)
    # fs = 500.0  # 采样频率 (Hz)
    lowcut = 4.0
    highcut = 30.0

    # subject_number = 0
    # subject_labels = []

    subject_id_list = []

    subject_segments_id_list = []
    graph_data_list = []
    labels = []
    assistance_information_list = []

    with open(file_dir, 'r') as f:
        subject_id_str = f.read()
        subject_id_list = subject_id_str.split(' ')
        subject_id_list = subject_id_list[:-1]

    # with open(file_dir, 'r') as f:
    #     subject_id_str = f.read()
    #     subject_id_list = subject_id_str.split(' ')
    #     label = 1
    #     for i in range(len(subject_id_list)-1):
    #         if int(subject_id_list[i]) > int(subject_id_list[i+1]):
    #             label = 0
    #         subject_labels.append(label)
    #     print(subject_labels)

    print(subject_id_list)



    lmdb_env = lmdb.open(lmdb_dir, readonly=True)

    with (lmdb_env.begin() as txn):
        for subject_id in subject_id_list:
            print(subject_id)
            segment_number_key_str = subject_id + "total_segment_number"
            subject_segments = txn.get(segment_number_key_str.encode())
            subject_segments = np.frombuffer(subject_segments, dtype=np.int64)[0]
            print(subject_segments)

            if (subject_id.startswith('0201')):
                    label = 1  # 以0201开头的是MD
            else:
                label = 0  # 否则是HC

            # label = subject_labels[subject_number]
            # subject_number = subject_number + 1

            for i in range(subject_segments):
                subject_segments_id = subject_id + "segment_" + str(i+1)
                adj_matrix_alpha_str = subject_segments_id + "_alpha_PCC_"
                adj_matrix_beta_str = subject_segments_id + "_beta_PCC_"
                adj_matrix_theta_str = subject_segments_id + "_theta_PCC_"

                embedding_alpha_str = subject_segments_id + "_alpha_embedding_"
                embedding_beta_str = subject_segments_id + "_beta_embedding_"
                embedding_theta_str = subject_segments_id + "_theta_embedding_"

                freqs_str = subject_segments_id + "_freqs_"

                matrix = txn.get(adj_matrix_alpha_str.encode())
                matrix = np.frombuffer(matrix, dtype=np.float64)
                matrix = np.reshape(matrix, (EEG_REGULAR_CHANNELS, -1))
                adj_matrix_alpha = np.copy(matrix)
                adj_matrix_alpha = filter_matrix(adj_matrix_alpha)

                matrix = txn.get(adj_matrix_beta_str.encode())
                matrix = np.frombuffer(matrix, dtype=np.float64)
                matrix = np.reshape(matrix, (EEG_REGULAR_CHANNELS, -1))
                adj_matrix_beta = np.copy(matrix)
                adj_matrix_beta = filter_matrix(adj_matrix_beta)

                matrix = txn.get(adj_matrix_theta_str.encode())
                matrix = np.frombuffer(matrix, dtype=np.float64)
                matrix = np.reshape(matrix, (EEG_REGULAR_CHANNELS, -1))
                adj_matrix_theta = np.copy(matrix)
                adj_matrix_theta = filter_matrix(adj_matrix_theta)

                matrix = txn.get(embedding_alpha_str.encode())
                matrix = np.frombuffer(matrix, dtype=np.float64)
                embedding_alpha = np.reshape(matrix, (EEG_REGULAR_CHANNELS, -1))

                matrix = txn.get(embedding_beta_str.encode())
                matrix = np.frombuffer(matrix, dtype=np.float64)
                embedding_beta = np.reshape(matrix, (EEG_REGULAR_CHANNELS, -1))

                matrix = txn.get(embedding_theta_str.encode())
                matrix = np.frombuffer(matrix, dtype=np.float64)
                embedding_theta = np.reshape(matrix, (EEG_REGULAR_CHANNELS, -1))

                matrix = txn.get(freqs_str.encode())
                matrix = np.frombuffer(matrix, dtype=np.float64)
                freqs = np.reshape(matrix, (257,))

                # 仅保留前20%的连接                                                 #保留20%的连接可能太少，尝试保留所有连接，但效果不佳，注意到大部分节点之间相似度高于0.9甚至0.95，仅有极少节点之间相似度为负，需保留这些节点数
                # percentile_80 = np.percentile(abs(hc_md_difference), 80)
                percentile_80 = np.percentile(abs(adj_matrix_alpha), 80)

                # 找出大于等于第80百分位数的所有元素的索引
                condition = abs(adj_matrix_alpha) >= percentile_80
                indices = np.argwhere(condition)

                mask = np.zeros(adj_matrix_alpha.shape, dtype=bool)
                for index in indices:
                    mask[tuple(index)] = True

                adj_matrix_alpha = np.where(mask, adj_matrix_alpha, 0)

                percentile_80 = np.percentile(abs(adj_matrix_beta), 80)

                # 找出大于等于第80百分位数的所有元素的索引
                condition = abs(adj_matrix_beta) >= percentile_80
                indices = np.argwhere(condition)

                mask = np.zeros(adj_matrix_beta.shape, dtype=bool)
                for index in indices:
                    mask[tuple(index)] = True

                adj_matrix_beta = np.where(mask, adj_matrix_beta, 0)

                percentile_80 = np.percentile(abs(adj_matrix_theta), 80)

                # 找出大于等于第80百分位数的所有元素的索引
                condition = abs(adj_matrix_theta) >= percentile_80
                indices = np.argwhere(condition)

                mask = np.zeros(adj_matrix_theta.shape, dtype=bool)
                for index in indices:
                    mask[tuple(index)] = True

                adj_matrix_theta = np.where(mask, adj_matrix_theta, 0)


                # 转换邻接矩阵为边索引和边特征
                edge_index_alpha, edge_attr_alpha = adj_to_edge_index_and_attr(adj_matrix_alpha)
                edge_index_beta, edge_attr_beta = adj_to_edge_index_and_attr(adj_matrix_beta)
                edge_index_theta, edge_attr_theta = adj_to_edge_index_and_attr(adj_matrix_theta)

                x_theta = torch.tensor(embedding_theta, dtype=torch.float)
                x_alpha = torch.tensor(embedding_alpha, dtype=torch.float)
                x_beta = torch.tensor(embedding_beta, dtype=torch.float)

                # Step 6: 构造 PyG 的 Data 对象
                data = Data(x=x_alpha, edge_index=edge_index_alpha, edge_attr=edge_attr_alpha, y=label)

                data.x_alpha = x_alpha
                data.x_theta = x_theta
                data.x_beta = x_beta

                data.edge_index_alpha = edge_index_alpha
                data.edge_attr_alpha = edge_attr_alpha
                data.edge_index_beta = edge_index_beta
                data.edge_attr_beta = edge_attr_beta
                data.edge_index_theta = edge_index_theta
                data.edge_attr_theta = edge_attr_theta

                graph_data_list.append(data)
                labels.append(label)
                subject_segments_id_list.append(subject_segments_id)

            assistance_information = {'origin subject id': subject_id , 'total subject segments': subject_segments , 'origin subject label': label}
            assistance_information_list.append(assistance_information)
            # print(labels)
    lmdb_env.close()
    return subject_segments_id_list,graph_data_list,labels,assistance_information_list,subject_id_list



def load_embedding_and_matrix_full_band(file_dir, lmdb_dir):
    EEG_REGULAR_CHANNELS = 68
    EEG_CHANNEL_INDICES = np.arange(0, EEG_REGULAR_CHANNELS)
    fs = 250.0  # 采样频率 (Hz)
    lowcut = 4.0
    highcut = 30.0

    subject_number = 0
    subject_id_list = []

    subject_segments_id_list = []
    graph_data_list = []
    labels = []
    assistance_information_list = []

    with open(file_dir, 'r') as f:
        subject_id_str = f.read()
        subject_id_list = subject_id_str.split(' ')
        subject_id_list = subject_id_list[:-1]

    print(subject_id_list)

    lmdb_env = lmdb.open(lmdb_dir, readonly=True)

    with (lmdb_env.begin() as txn):
        for subject_id in subject_id_list:
            print(subject_id)
            segment_number_key_str = subject_id + "total_segment_number"
            subject_segments = txn.get(segment_number_key_str.encode())
            subject_segments = np.frombuffer(subject_segments, dtype=np.int64)[0]
            print(subject_segments)
            if (subject_id.startswith('0201')):
                label = 1  # 以0201开头的是MD
            else:
                label = 0  # 否则是HC
            for i in range(subject_segments):
                subject_segments_id = subject_id + "segment_" + str(i + 1)
                adj_matrix_str = subject_segments_id + "_PCC_"
                embedding_str = subject_segments_id + "_embedding_"

                freqs_str = subject_segments_id + "_freqs_"

                matrix = txn.get(adj_matrix_str.encode())
                matrix = np.frombuffer(matrix, dtype=np.float64)
                matrix = np.reshape(matrix, (68, 68))
                adj_matrix = np.copy(matrix)
                adj_matrix = filter_matrix(adj_matrix)

                matrix = txn.get(embedding_str.encode())
                matrix = np.frombuffer(matrix, dtype=np.float64)
                embedding = np.reshape(matrix, (68, 257))

                matrix = txn.get(freqs_str.encode())
                matrix = np.frombuffer(matrix, dtype=np.float64)
                freqs = np.reshape(matrix, (257,))

                # 仅保留前20%的连接                                                 #保留20%的连接可能太少，尝试保留所有连接，但效果不佳，注意到大部分节点之间相似度高于0.9甚至0.95，仅有极少节点之间相似度为负，需保留这些节点数
                # percentile_80 = np.percentile(abs(hc_md_difference), 80)
                percentile_80 = np.percentile(abs(adj_matrix), 80)

                # 找出大于等于第80百分位数的所有元素的索引
                condition = abs(adj_matrix) >= percentile_80
                indices = np.argwhere(condition)

                mask = np.zeros(adj_matrix.shape, dtype=bool)
                for index in indices:
                    mask[tuple(index)] = True

                adj_matrix = np.where(mask, adj_matrix, 0)

                # 转换邻接矩阵为边索引和边特征
                edge_index, edge_attr = adj_to_edge_index_and_attr(adj_matrix)

                x = torch.tensor(embedding, dtype=torch.float)


                # Step 6: 构造 PyG 的 Data 对象
                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=label)

                graph_data_list.append(data)
                labels.append(label)
                subject_segments_id_list.append(subject_segments_id)

            assistance_information = {'origin subject id': subject_id, 'total subject segments': subject_segments,
                                      'origin subject label': label}
            assistance_information_list.append(assistance_information)
    lmdb_env.close()
    return subject_segments_id_list, graph_data_list, labels, assistance_information_list, subject_id_list





                # if (str(id) + '_alpha_PCC_' in key or
                #         str(id) + '_beta_PCC_' in key or
                #         str(id) + '_theta_PCC_' in key):
                #     value = txn.get(key.encode())
                #     matrix = np.frombuffer(value, dtype=np.float64)
                #     print(key.encode())
                #     print(matrix.shape)
                #     matrix = np.reshape(matrix, (68, 68))
                #     adj_matrix = np.copy(matrix)
                #     if 'alpha' in key:
                #         adj_matrix_alpha = filter_matrix(adj_matrix)
                #     elif 'beta' in key:
                #         adj_matrix_beta = filter_matrix(adj_matrix)
                #     elif 'theta' in key:
                #         adj_matrix_theta = filter_matrix(adj_matrix)
                # elif 'average_wave_difference' in key:
                #     value = txn.get(key.encode())
                #     matrix = np.frombuffer(value, dtype=np.float64)
                #     matrix = np.reshape(matrix, (68, 68))
                #     hc_md_difference = np.copy(matrix)




# 定义一个函数来计算固定波段每个通道的功率谱密度（PSD）
def compute_psd(eeg_signal,fs=250, nperseg=256, nfft=512):
    print(eeg_signal.shape)

    freqs, psd_template = welch(eeg_signal[0, :], fs=fs, nperseg=nperseg, nfft=nfft)
    psd_list_plus_freqs = np.zeros((eeg_signal.shape[0]+1, len(freqs)))                        #因为freqs和psd_template的形状都是(257,)所以把freqs放在psd_list的最后一个，一同输出

    for channel in range(eeg_signal.shape[0]):
        freqs, psd = welch(eeg_signal[channel,:], fs=fs, nperseg=nperseg, nfft=nfft)
        psd_list_plus_freqs[channel,:] = psd
    psd_list_plus_freqs[eeg_signal.shape[0],:] = freqs
    print(psd_list_plus_freqs[0])
    print(psd_list_plus_freqs.shape)
    return psd_list_plus_freqs
        # np.array(freq_list)

# 此函数为计算每个样本中每个通道的每个波段（如有波段）的功率谱密度(PSD)
def compute_psd_for_every_channel_and_band(filtered_eeg_data_and_labels_and_type):
    psds = []                                                                                       #样本数×波段数×通道数×采样点数（单波段时波段数忽略）
    freqs = []                                                                                      #样本数×波段数×采样点数
    for sample in filtered_eeg_data_and_labels_and_type:
        eeg_signal = sample['eeg_signals']
        print(eeg_signal.shape)

        if eeg_signal.ndim == 3:            #存在多波段
            sample_psd = []                                                                         #波段数×通道数×采样点数
            sample_freqs = []                                                                       #波段数×采样点数
            for i in range(eeg_signal.shape[2]):
                psd_list_plus_freqs = compute_psd(eeg_signal[:,:,i])
                sample_psd.append(psd_list_plus_freqs[:psd_list_plus_freqs.shape[0]-1])             #通道数×采样点数
                sample_freqs.append(psd_list_plus_freqs[psd_list_plus_freqs.shape[0]-1])            #采样点数
            psds.append(sample_psd)
            freqs.append(sample_freqs)
        elif eeg_signal.ndim == 2:          #仅有单个波段
            psd_list_plus_freqs = compute_psd(eeg_signal)
            sample_psd = psd_list_plus_freqs[:psd_list_plus_freqs.shape[0]-1]
            sample_freqs = psd_list_plus_freqs[psd_list_plus_freqs.shape[0]-1]
            psds.append(sample_psd)
            freqs.append(sample_freqs)

    return psds,freqs


def compute_de(eeg_signal):
    print(eeg_signal.shape)
    de_list = []
    for channel in range(eeg_signal.shape[0]):
        # print(eeg_signal[channel,:])
        de = differential_entropy(eeg_signal[channel,:])
        de_list.append(de)
        # print(de.shape)
        # print(de)
    return np.array(de_list)

def scale_data_extractor(ids):
    scale_path = '../EEG_128channels_resting_lanzhou_2015/subjects_information_EEG_128channels_resting_lanzhou_2015.xlsx'
    df = pd.read_excel(scale_path, sheet_name='Sheet1')
    # print(df)
    scale_data_array = []
    print(ids)
    for id in ids:
        # print(id)
        pd_array = df[(df['subject id'] == int(id))]
        # print(pd_array)
        # py_array = pd_array[['education（years）', 'PHQ-9', 'CTQ-SF','LES', 'SSRS', 'GAD-7', 'PSQI']].values
        # py_array = pd_array[['education（years）', 'CTQ-SF','LES', 'SSRS', 'GAD-7', 'PSQI']].values
        py_array = pd_array[['GAD-7']].values
        pd_array = np.array(py_array).reshape(-1)
        scale_data_array.append(pd_array)
    # print(scale_data_array)
    return scale_data_array

def weights_init(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)









def filter_matrix(matrix):          #filter several irrelevant relations
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if abs(matrix[i,j]) < 0.05:
                matrix[i,j] = 0.0
    return matrix


def adj_to_edge_index_and_attr(adj_matrix):
    # 将邻接矩阵转换为边索引和边属性
    adj_matrix = torch.from_numpy(adj_matrix)
    edge_indices = torch.nonzero(adj_matrix, as_tuple=True)
    edge_index = torch.stack(edge_indices, dim=0)
    edge_attr = adj_matrix[edge_indices].view(-1, 1)
    return edge_index, edge_attr




# 使用互相关矩阵构建图
def build_graph_from_psd(psd,label,id):                 #邻接矩阵各个联系仅需保留health_corr_avg-md_corr_avg绝对值较大的那些联系就行了,尝试保留前20%
    # Step 1: 计算相关性矩阵
    # corr_matrix = np.corrcoef(psd)  # shape: [num_nodes, num_nodes]

    lmdb_dir = '../EEG_128channels_resting_lanzhou_2015/PCC'
    lmdb_env = lmdb.open(lmdb_dir, readonly=True)
    # adj_matrix_alpha = np.random.randn(68, 68)     #different types of wave(alpha,beta,theta)
    # adj_matrix_beta = np.random.randn(68, 68)
    # adj_matrix_theta = np.random.randn(68, 68)
    adj_matrix_full = np.random.randn(68,68)
    hc_md_difference = np.zeros((68, 68), dtype=float)

    with (lmdb_env.begin() as lmdb_mod_txn):
        mod_cursor = lmdb_mod_txn.cursor()
        for idx, (key, value) in enumerate(mod_cursor):
            key = str(key, encoding='utf-8')
            with lmdb_env.begin() as txn:
                # if (str(id) + '_alpha_PCC_' in key or
                #     str(id) + '_beta_PCC_' in key or
                #     str(id) + '_theta_PCC_' in key):
                if str(id) + '_PCC_' in key:
                    value = txn.get(key.encode())
                    matrix = np.frombuffer(value, dtype=np.float64)
                    print(key.encode())
                    print(matrix.shape)
                    matrix = np.reshape(matrix, (68, 68))
                    adj_matrix = np.copy(matrix)
                    adj_matrix_full = filter_matrix(adj_matrix)
                    # if 'alpha' in key:
                    #     adj_matrix_alpha = filter_matrix(adj_matrix)
                    # elif 'beta' in key:
                    #     adj_matrix_beta = filter_matrix(adj_matrix)
                    # elif 'theta' in key:
                    #     adj_matrix_theta = filter_matrix(adj_matrix)
                # elif 'average_wave_difference' in key:
                #     value = txn.get(key.encode())
                #     matrix = np.frombuffer(value, dtype=np.float64)
                #     matrix = np.reshape(matrix, (68, 68))
                #     hc_md_difference = np.copy(matrix)
    lmdb_env.close()

    print(adj_matrix_full[0])

    # adj_matrix_full = np.nan_to_num(adj_matrix_full, nan=0.0)       #将可能出现的nan数据置为0，避免后续训练出错

    #仅保留前20%的连接                                                 #保留20%的连接可能太少，尝试保留所有连接，但效果不佳，注意到大部分节点之间相似度高于0.9甚至0.95，仅有极少节点之间相似度为负，需保留这些节点数
    # percentile_80 = np.percentile(abs(hc_md_difference), 80)
    percentile_80 = np.percentile(abs(adj_matrix_full), 80)

    # 找出大于等于第80百分位数的所有元素的索引
    condition = abs(adj_matrix_full) >= percentile_80
    indices = np.argwhere(condition)

    mask = np.zeros(adj_matrix_full.shape, dtype=bool)
    for index in indices:
        mask[tuple(index)] = True

    adj_matrix_full = np.where(mask, adj_matrix_full, 0)
    print(adj_matrix_full[0])

    # 转换邻接矩阵为边索引和边特征
    # edge_index_alpha,edge_attr_alpha = adj_to_edge_index_and_attr(adj_matrix_alpha)
    # edge_index_beta,edge_attr_beta = adj_to_edge_index_and_attr(adj_matrix_beta)
    # edge_index_theta,edge_attr_theta = adj_to_edge_index_and_attr(adj_matrix_theta)
    edge_index, edge_attr = adj_to_edge_index_and_attr(adj_matrix_full)

    # Step 2: 创建带边权的无向图
    # G = nx.from_numpy_array(corr_matrix)

    # Step 3: 可选：过滤掉弱相关边（例如相关系数小于阈值）
    # edges_to_remove = [(u, v) for u, v, d in G.edges(data=True) if abs(d['weight']) < threshold]
    # G.remove_edges_from(edges_to_remove)

    # Step 4: 获取 edge_index 和 edge_attr（边的权重）
    # edge_index = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous()
    # edge_attr = torch.tensor([G[u][v]['weight'] for u, v in G.edges], dtype=torch.float)

    # Step 5: 使用原始 psd 作为节点特征 x
    x = torch.tensor(psd, dtype=torch.float)

    # Step 6: 构造 PyG 的 Data 对象
    # data = Data(x=x, edge_index=edge_index_alpha, edge_attr=edge_attr_alpha,y=label)
    # data.edge_index_beta = edge_index_beta
    # data.edge_attr_beta = edge_attr_beta
    # data.edge_index_theta = edge_index_theta
    # data.edge_attr_theta = edge_attr_theta
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr,y=label)

    # print("节点特征 x:", data.x)
    # print("边索引 edge_index:", data.edge_index)
    # print("边权重 edge_attr:", data.edge_attr)

    return data

# 使用不同波段的互相关矩阵构建图
def build_graph_from_psd_multi_band(psd,label,id):                 #邻接矩阵各个联系仅需保留health_corr_avg-md_corr_avg绝对值较大的那些联系就行了,尝试保留前20%
    # Step 1: 计算相关性矩阵
    # corr_matrix = np.corrcoef(psd)  # shape: [num_nodes, num_nodes]
    psd = np.array(psd)
    print(psd.shape)

    lmdb_dir = '../EEG_128channels_resting_lanzhou_2015/PCC'
    lmdb_env = lmdb.open(lmdb_dir, readonly=True)
    adj_matrix_alpha = np.random.randn(68, 68)     #different types of wave(alpha,beta,theta)
    adj_matrix_beta = np.random.randn(68, 68)
    adj_matrix_theta = np.random.randn(68, 68)
    # hc_md_difference = np.zeros((68, 68), dtype=float)

    with (lmdb_env.begin() as lmdb_mod_txn):
        mod_cursor = lmdb_mod_txn.cursor()
        for idx, (key, value) in enumerate(mod_cursor):
            key = str(key, encoding='utf-8')
            with lmdb_env.begin() as txn:
                if (str(id) + '_alpha_PCC_' in key or
                    str(id) + '_beta_PCC_' in key or
                    str(id) + '_theta_PCC_' in key):
                    value = txn.get(key.encode())
                    matrix = np.frombuffer(value, dtype=np.float64)
                    print(key.encode())
                    print(matrix.shape)
                    matrix = np.reshape(matrix, (68, 68))
                    adj_matrix = np.copy(matrix)
                    if 'alpha' in key:
                        adj_matrix_alpha = filter_matrix(adj_matrix)
                    elif 'beta' in key:
                        adj_matrix_beta = filter_matrix(adj_matrix)
                    elif 'theta' in key:
                        adj_matrix_theta = filter_matrix(adj_matrix)
                # elif 'average_wave_difference' in key:
                #     value = txn.get(key.encode())
                #     matrix = np.frombuffer(value, dtype=np.float64)
                #     matrix = np.reshape(matrix, (68, 68))
                #     hc_md_difference = np.copy(matrix)
    lmdb_env.close()

    # print(adj_matrix_alpha[0])
    # print(adj_matrix_beta[0])
    # print(adj_matrix_theta[0])
    #
    # # adj_matrix_alpha = np.nan_to_num(adj_matrix_alpha, nan=0.0)       #将可能出现的nan数据置为0，避免后续训练出错
    # # adj_matrix_beta = np.nan_to_num(adj_matrix_beta, nan=0.0)
    # # adj_matrix_theta = np.nan_to_num(adj_matrix_theta, nan=0.0)
    #
    #仅保留前20%的连接                                                 #保留20%的连接可能太少，尝试保留所有连接，但效果不佳，注意到大部分节点之间相似度高于0.9甚至0.95，仅有极少节点之间相似度为负，需保留这些节点数
    # percentile_80 = np.percentile(abs(hc_md_difference), 80)
    percentile_80 = np.percentile(abs(adj_matrix_alpha), 80)

    # 找出大于等于第80百分位数的所有元素的索引
    condition = abs(adj_matrix_alpha) >= percentile_80
    indices = np.argwhere(condition)

    mask = np.zeros(adj_matrix_alpha.shape, dtype=bool)
    for index in indices:
        mask[tuple(index)] = True

    adj_matrix_alpha = np.where(mask, adj_matrix_alpha, 0)

    percentile_80 = np.percentile(abs(adj_matrix_beta), 80)

    # 找出大于等于第80百分位数的所有元素的索引
    condition = abs(adj_matrix_beta) >= percentile_80
    indices = np.argwhere(condition)

    mask = np.zeros(adj_matrix_beta.shape, dtype=bool)
    for index in indices:
        mask[tuple(index)] = True

    adj_matrix_beta = np.where(mask, adj_matrix_beta, 0)

    percentile_80 = np.percentile(abs(adj_matrix_theta), 80)

    # 找出大于等于第80百分位数的所有元素的索引
    condition = abs(adj_matrix_theta) >= percentile_80
    indices = np.argwhere(condition)

    mask = np.zeros(adj_matrix_theta.shape, dtype=bool)
    for index in indices:
        mask[tuple(index)] = True

    adj_matrix_theta = np.where(mask, adj_matrix_theta, 0)
    #
    # print(adj_matrix_alpha[0])
    # print(adj_matrix_beta[0])
    # print(adj_matrix_theta[0])

    # 转换邻接矩阵为边索引和边特征
    edge_index_alpha,edge_attr_alpha = adj_to_edge_index_and_attr(adj_matrix_alpha)
    edge_index_beta,edge_attr_beta = adj_to_edge_index_and_attr(adj_matrix_beta)
    edge_index_theta,edge_attr_theta = adj_to_edge_index_and_attr(adj_matrix_theta)

    # Step 2: 创建带边权的无向图
    # G = nx.from_numpy_array(corr_matrix)

    # Step 3: 可选：过滤掉弱相关边（例如相关系数小于阈值）
    # edges_to_remove = [(u, v) for u, v, d in G.edges(data=True) if abs(d['weight']) < threshold]
    # G.remove_edges_from(edges_to_remove)

    # Step 4: 获取 edge_index 和 edge_attr（边的权重）
    # edge_index = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous()
    # edge_attr = torch.tensor([G[u][v]['weight'] for u, v in G.edges], dtype=torch.float)

    # Step 5: 使用原始 psd 作为节点特征 x
    x_theta = torch.tensor(psd[0], dtype=torch.float)
    x_alpha = torch.tensor(psd[1], dtype=torch.float)
    x_beta = torch.tensor(psd[2], dtype=torch.float)
    # x = torch.tensor(psd[], dtype=torch.float)

    # Step 6: 构造 PyG 的 Data 对象
    data = Data(x=x_alpha, edge_index=edge_index_alpha, edge_attr=edge_attr_alpha,y=label)

    data.x_alpha = x_alpha
    data.x_theta = x_theta
    data.x_beta = x_beta

    data.edge_index_alpha = edge_index_alpha
    data.edge_attr_alpha = edge_attr_alpha
    data.edge_index_beta = edge_index_beta
    data.edge_attr_beta = edge_attr_beta
    data.edge_index_theta = edge_index_theta
    data.edge_attr_theta = edge_attr_theta
    # data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr,y=label)

    # print("节点特征 x:", data.x)
    # print("边索引 edge_index:", data.edge_index)
    # print("边权重 edge_attr:", data.edge_attr)

    return data



def create_train_test_loaders(graphs,assistance_information_list,subject_ids, test_ratio=0.2, batch_size=16, seed=42): #为了防止模型学习到个人的fingerprint（针对train和test中均存在由相同母样本分割而成的子样本，例如train中有'02030007segment_2', '02030007segment_6',test中有'02030007segment_12'）,引入subject_segments_id_list,subject_ids辅助切分train，test

    # 划分
    test_id_size = int(test_ratio * len(subject_ids))
    train_id_size = len(subject_ids) - test_id_size


    train_subject_ids, test_subject_ids = random_split(
        subject_ids,
        [train_id_size,test_id_size],
        generator=torch.Generator().manual_seed(seed)
    )

    train_indices = []
    test_indices = []

    segment_cumulative_index = 0

    for i in range(len(subject_ids)):
        # print(subject_ids[i])
        # print(assistance_information_list[i])

        num_segments = assistance_information_list[i]['total subject segments']

        segment_indices = list(range(segment_cumulative_index, segment_cumulative_index + num_segments))
        # print(subject_segments_id_list[segment_cumulative_index: segment_cumulative_index + num_segments])
        if subject_ids[i] in train_subject_ids:
            train_indices.extend(segment_indices)
        else:
            test_indices.extend(segment_indices)

        segment_cumulative_index += num_segments

    # print(train_indices)
    # print(list(train_subject_ids))
    # print(test_indices)
    # print(list(test_subject_ids))


    train_dataset = [graphs[i] for i in train_indices]
    test_dataset = [graphs[i] for i in test_indices]

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def create_train_test_loaders(graphs, test_ratio=0.2, batch_size=16, seed=42): #原来的分割方法直接随机打乱
    test_size = int(test_ratio * len(graphs))
    train_size = len(graphs) - test_size

    train_dataset,test_dataset = random_split(
        graphs,
        [train_size,test_size],
        generator=torch.Generator().manual_seed(seed)
    )

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader


def create_train_test_loaders_optimized(graphs, test_ratio=0.2, batch_size=16, seed=42):
    # 划分数据集 (使用 torch.utils.data.random_split)
    test_size = int(test_ratio * len(graphs))
    train_size = len(graphs) - test_size

    # 使用固定的生成器划分，以确保可复现性
    data_generator = torch.Generator().manual_seed(seed)
    train_dataset, test_dataset = random_split(
        graphs,
        [train_size, test_size],
        generator=data_generator
    )

    # 创建 PyG DataLoader
    # 注意：我们使用 PyG 的 DataLoader，它会自动处理图的批处理。
    # 并且在 DataLoader 中使用 data_generator 来固定 shuffle 行为
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=data_generator
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, test_loader


# 提示：请确保您在文件中使用了 from torch_geometric.loader import DataLoader

def batch_edgeindex_to_dense(edge_index, batch_size, num_nodes):
    """
    将 PyG 批次的 edge_index 转换为稠密邻接矩阵。
    Args:
        edge_index: [2, E_total]
        batch: [N_total] 节点所属图编号
        batch_size: 图数量
        num_nodes: 每图节点数
    Returns:
        dense_adj: [batch_size, num_nodes, num_nodes]
    """
    device = edge_index.device
    dense_adj = torch.zeros((batch_size, num_nodes, num_nodes), device=device)

    # 为每个图构建邻接矩阵
    for i in range(batch_size):
        node_offset = i * num_nodes
        mask = (edge_index[0] >= node_offset) & (edge_index[0] < node_offset + num_nodes)
        edges_i = edge_index[:, mask] - node_offset  # 还原到局部节点编号 [0, num_nodes)
        dense_adj[i, edges_i[0], edges_i[1]] = 1.0

    return dense_adj




if __name__ == "__main__":
    load_dir = '../EEG_128channels_resting_lanzhou_2015'
    # filtereegdata(load_dir)
    scale_data_extractor(['02010002', '02010004', '02010005', '02010006', '02010008', '02010010', '02010011', '02010012', '02010013', '02010015', '02010016', '02010018', '02010019', '02010021', '02010022', '02010023', '02010024', '02010025', '02010026', '02010028', '02010030', '02010033', '02010034', '02010036', '02020008', '02020010', '02020013', '02020014', '02020015', '02020016', '02020018', '02020019', '02020020', '02020021', '02020022', '02020023', '02020025', '02020026', '02020027', '02020029', '02030002', '02030003', '02030004', '02030005', '02030006', '02030007', '02030009', '02030014', '02030017', '02030018', '02030019', '02030020', '02030021'])