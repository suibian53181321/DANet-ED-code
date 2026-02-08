import numpy as np
from scipy.signal import welch
import networkx as nx
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from funcs import *
from model import *
import matplotlib

from subband_importance_visualization import plot_subgraph_importance, calculate_final_importance, plot_topomap_from_set
from subject_clustering_using_labels_visualization import transform_graph_data_into_embeddings, save_epoch_clusters

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
from sklearn.model_selection import StratifiedKFold
from contextlib import redirect_stdout
from torch_geometric.utils import to_dense_adj

matplotlib.rcParams['font.family'] = 'Times New Roman'  # 黑体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 关键：确保 SVG 输出时文字不转为路径
matplotlib.rcParams['svg.fonttype'] = 'none'




# 假设这是预处理后的EEG信号数据，形状为 (样本数, 时间点, 通道数)

if __name__ == '__main__':
    lmdb_dir = '../EEG_128channels_resting_lanzhou_2015/PCC'
    subject_id_txt = 'subject_id.txt'

    subject_segments_id_list,graph_data_list,labels,assistance_information_list,eeg_subject_ids = load_embedding_and_matrix(subject_id_txt,lmdb_dir)       # 经过修改后，train.py仅负责提取lmdb中的数据,将eeg信号中的数据进行提取得到embedding和adjacent matrix的过程主要由filter_and_cluster中的adjacent_matrix_eeg_new.py实现
    # subject_segments_id_list, graph_data_list, labels, assistance_information_list, eeg_subject_ids = load_embedding_and_matrix_full_band(subject_id_txt, lmdb_dir)


    print(subject_segments_id_list)

    HC_count = 0
    MD_count = 0
    for i in range(len(assistance_information_list)):
        if assistance_information_list[i]['origin subject label'] == 0:
            HC_count += assistance_information_list[i]['total subject segments']
        else :
            MD_count += assistance_information_list[i]['total subject segments']
    print("HC_count:", HC_count)
    print("MD_count:", MD_count)

    print(graph_data_list[0].edge_attr.shape)

    # print(graph_data_list[0].edge_attr_beta.shape)     #torch.Size([926, 1])torch.Size([257, num_nodes])
    # print(graph_data_list[0].x_beta.shape)
    # print(graph_data_list[0].x_beta)


    # #加入量表数据
    # origin_scale_data = scale_data_extractor(eeg_subject_ids)
    #
    # all_scale_features = np.stack(origin_scale_data)           #需要先对量表数据进行缩放
    # scale_scaler = StandardScaler()
    # all_scale_features = scale_scaler.fit_transform(all_scale_features)     #要赋值才能生效
    # print("First subject scaled features:", all_scale_features[0])
    #
    # scale_data_in_segments = []
    # for i in range(len(eeg_subject_ids)):
    #     num_segments = assistance_information_list[i]['total subject segments']
    #     scale_data_in_segments.append([all_scale_features[i]] * num_segments)               #scale_data_in_segments,array like(subject_num,segment_num,attr_num)
    # print(f"Graph List Length: {len(graph_data_list)}")
    # print(f"Scale Data Number: {len(scale_data_in_segments)}")
    # print(f"Subject Scale Data Length: {len(scale_data_in_segments[0])}")
    # print(f"Attribute Length: {len(scale_data_in_segments[0][0])}")
    # print(f"Subject Scale Data Value: {scale_data_in_segments[0][0]}")
    #
    # # 对eeg特征进行缩放
    # scaler = StandardScaler()
    # eeg_subject_index = 0
    # segment_count = 0
    # # for i, data in enumerate(graph_data_list):
    # #     graph_data_list[i].x = torch.tensor(scaler.fit_transform(data.x.numpy()), dtype=torch.float)
    # #     graph_data_list[i].x_alpha = torch.tensor(scaler.fit_transform(data.x_alpha.numpy()), dtype=torch.float)
    # #     graph_data_list[i].x_theta = torch.tensor(scaler.fit_transform(data.x_theta.numpy()), dtype=torch.float)
    # #     graph_data_list[i].x_beta = torch.tensor(scaler.fit_transform(data.x_beta.numpy()), dtype=torch.float)
    # #
    # #     graph_data_list[i].global_feature = torch.tensor(scale_data_in_segments[eeg_subject_index][segment_count].reshape(1,-1), dtype=torch.float)
    # #     segment_count += 1
    # #     if segment_count >= assistance_information_list[eeg_subject_index]['total subject segments']:
    # #         eeg_subject_index += 1
    # #         segment_count = 0
    # # print("Check final shape:", graph_data_list[0].global_feature.shape)
    # # print("Check value:", graph_data_list[0].global_feature)
    #
    # for i, data in enumerate(graph_data_list):
    #     source_data_tensor = torch.tensor(scale_data_in_segments[eeg_subject_index][segment_count], dtype=torch.float)
    #     # Reshape to [1, 6] and repeat N times along dim 0, 1 time along dim 1
    #     global_feature = source_data_tensor.view(1, -1).repeat(graph_data_list[i].x.shape[0], 1)
    #     graph_data_list[i].x = torch.cat((torch.tensor(scaler.fit_transform(data.x.numpy()), dtype=torch.float),global_feature), dim=1)
    #     graph_data_list[i].x_alpha = torch.cat((torch.tensor(scaler.fit_transform(data.x_alpha.numpy()), dtype=torch.float),global_feature), dim=1)
    #     graph_data_list[i].x_theta = torch.cat((torch.tensor(scaler.fit_transform(data.x_theta.numpy()), dtype=torch.float),global_feature), dim=1)
    #     graph_data_list[i].x_beta = torch.cat((torch.tensor(scaler.fit_transform(data.x_beta.numpy()), dtype=torch.float),global_feature), dim=1)
    #
    #     segment_count += 1
    #     if segment_count >= assistance_information_list[eeg_subject_index]['total subject segments']:
    #         eeg_subject_index += 1
    #         segment_count = 0
    # # print("Check final shape:", graph_data_list[0].global_feature.shape)
    # # print("Check value:", graph_data_list[0].global_feature)


    # 提取所有图的标签（用于分层交叉验证）
    all_labels = np.array([data.y for data in graph_data_list])
    print(all_labels)
    num_samples = all_labels.shape[0]

    for practice in range(1):
        seeds = [42,69,54,106,28]
        #多折交叉验证数据集划分
        k_fold = 5
        skf = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=seeds[practice])
        train_rounds = 50

        batch_size = 16
        num_nodes = 68
        num_features = graph_data_list[0].num_features
        print("num_features:", num_features)
        hidden_channels = 16
        out_channel = 1  # 使用单神经元输出
        # global_feature_channels = len(scale_data_in_segments[0][0]) #使用6个维度的图属性
        global_feature_channels = 0
        recon_weight = 0.3
        cls_weight = 0.7
        lr = 0.0004

        k_fold_train_epoch_losses = []
        k_fold_test_epoch_losses = []
        k_fold_train_accuracies = []
        k_fold_test_accuracies = []
        k_fold_best_train_performances = []
        k_fold_best_test_performances = []

        # with open("loss_log_three_fold_multiattention.txt", "a") as file:
        #     file.write("threefoldmultiattention:\n")
        for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(num_samples), all_labels)):
            print(f"\nFold {fold + 1}")

            # 构建子集
            train_subset = [graph_data_list[i] for i in train_idx]
            val_subset = [graph_data_list[i] for i in val_idx]

            print(train_idx[:200])
            print(val_idx[:200])
            specific_sample_idx = np.where(val_idx == 470)[0][0]  # 这是要用到的specific_sample的在val_idx的索引，可以在y_prob_test和scores_matrix通过索引访问
            val_embeddings, val_labels = transform_graph_data_into_embeddings(val_subset, num_nodes)
            save_epoch_clusters(val_embeddings, val_labels, 0)


            # 创建 DataLoader
            train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

            # model = GNNBaseline('GIN', num_features, hidden_channels, out_channel)
            # model = SimplestFullBandGAT(num_features,out_channel)
            # model = SimpleGCN(num_features, hidden_channels)
            # model = MultiBandGAT(num_features, hidden_channels,out_channel,global_feature_channels)
            # model = SingleFullBandGAT(num_features, hidden_channels, out_channel,global_feature_channels)
            # model = SingleFullBandGAT(num_features, hidden_channels, out_channel)
            # model = MultiBandGATWithNodeImportance(num_features, hidden_channels, out_channel,global_feature_channels, heads=1)
            # model = NewMultibandGATClassifier(num_features, hidden_channels, out_channel,global_feature_channels, heads=1)
            # model = NewFullBandGATWithNodeImportance(num_features, hidden_channels, out_channel, global_feature_channels,heads=1)
            model = NewMultiBandGATWithNodeImportance(num_features, hidden_channels, out_channel, global_feature_channels,
                                                      heads=1)

            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
            # optimizer = torch.optim.Adam(model.parameters(), lr=0.0008, weight_decay=1e-3)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            # criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
            # criterion = MultiTaskLossModuleForFullBand(recon_weight=recon_weight, cls_weight=cls_weight, pos_weight_edge=None)
            criterion = MultiTaskLossModule(recon_weight=recon_weight, cls_weight=cls_weight, pos_weight_edge=None)
            model.apply(weights_init)

            best_train_performance = {'epoch': 0, 'train_loss': float('inf'), 'accuracy': 0.0, 'precision': 0.0,
                                      'recall': 0.0, 'f1': 0.0, 'roc_auc': 0.0, 'loss': 0.0}
            best_test_performance = {'epoch': 0, 'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0,
                                     'roc_auc': 0.0}

            print(model)

            model = model.to(device)

            train_epoch_losses = []
            test_epoch_losses = []
            train_accuracies = []
            test_accuracies = []

            epoch_subband_weights = []

            # 训练模型
            for epoch in range(train_rounds):
                print('========== Epoch:', epoch + 1)

                model.train()
                total_loss = 0
                total_num = 0
                # 因为原来代码中重复将数据从CPU和GPU之间搬运导致CPU占用率畸高，GPU占用率很低，所以改进了此部分代码减少搬运同时在GPU上维护指标
                all_preds_train = []
                all_labels_eval_train = []
                all_probs_train = []

                for batch in train_loader:
                    batch = batch.to(device)
                    labels = batch.y.to(device)

                    # out = model(batch)
                    # out = out.float().view(-1)
                    # # print(out.shape)
                    # labels = labels.float()
                    # loss = criterion(out,labels)

                    # reconstructed_adj_matrix, out = model(
                    #     batch, return_attn=False)  # out包含原始分数，未进行sigmod（不能使用softmax）
                    # edge_index = batch_edgeindex_to_dense(batch.edge_index, labels.shape[0], num_nodes)
                    # out = out.float().view(-1)
                    # labels = labels.float()
                    # loss, loss_info = criterion(logits=reconstructed_adj_matrix,
                    #                             adj_target=edge_index,
                    #                             cls_logits=out,
                    #                             cls_target=labels)

                    reconstructed_alpha_adj_matrix, reconstructed_beta_adj_matrix, reconstructed_theta_adj_matrix, out,_,_ = model(
                        batch, return_attn=False)  # out包含原始分数，未进行sigmod（不能使用softmax）
                    edge_index_theta = to_dense_adj(batch.edge_index_theta, batch.batch, max_num_nodes=num_nodes)
                    edge_index_alpha = to_dense_adj(batch.edge_index_alpha, batch.batch, max_num_nodes=num_nodes)
                    edge_index_beta = to_dense_adj(batch.edge_index_beta, batch.batch, max_num_nodes=num_nodes)
                    out = out.float().view(-1)
                    labels = labels.float()
                    loss, loss_info = criterion(logits_theta=reconstructed_theta_adj_matrix,
                                                logits_alpha=reconstructed_alpha_adj_matrix,
                                                logits_beta=reconstructed_beta_adj_matrix,
                                                adj_target_theta=edge_index_theta,
                                                adj_target_alpha=edge_index_alpha,
                                                adj_target_beta=edge_index_beta,
                                                cls_logits=out,
                                                cls_target=labels)

                    optimizer.zero_grad()
                    loss.backward()
                    # for name, param in model.named_parameters():
                    #     if param.grad is not None:
                    #         print(
                    #             f"Gradient for {name}: mean={param.grad.mean().item()}, std={param.grad.std().item()}")
                    optimizer.step()

                    probabilities = torch.sigmoid(out)
                    pred = (probabilities > 0.5).float()

                    # print(probabilities)
                    # print(pred)
                    #
                    # print(f"predict:{pred}\n")
                    # print(f"labels:{labels}\n")
                    # 取消每个batch将数据搬运回cpu，提高GPU利用效率
                    all_preds_train.append(pred)
                    all_probs_train.append(probabilities)
                    all_labels_eval_train.append(labels)

                    total_loss += loss.item()
                    total_num += labels.size(0)

                y_true_train = torch.cat(all_labels_eval_train).detach().cpu().numpy().astype(int)
                y_pred_train = torch.cat(all_preds_train).detach().cpu().numpy().astype(int)
                y_prob_train = torch.cat(all_probs_train).detach().cpu().numpy()

                epoch_acc = accuracy_score(y_true_train, y_pred_train)
                epoch_pre = precision_score(y_true_train, y_pred_train, zero_division=0)  # add handler for zero_division events
                epoch_rec = recall_score(y_true_train, y_pred_train, zero_division=0)
                epoch_f1 = f1_score(y_true_train, y_pred_train, zero_division=0)
                epoch_auc = roc_auc_score(y_true_train, y_prob_train)
                train_epoch_loss = total_loss / total_num

                train_epoch_losses.append(train_epoch_loss)
                train_accuracies.append(epoch_acc)

                print(
                    f'Epoch: {epoch + 1}, Train loss: {train_epoch_loss:.4f}, Train acc: {epoch_acc:.4f}, Train pre: {epoch_pre:.4f}, '
                    f'Train f1: {epoch_f1:.4f}, Train rec: {epoch_rec:.4f}, Train auc: {epoch_auc:.4f}')

                if epoch_acc > best_train_performance['accuracy'] and train_epoch_loss < best_train_performance[
                    'train_loss']:
                    print('update best performance\n')
                    best_train_performance['epoch'] = epoch + 1
                    best_train_performance['train_loss'] = train_epoch_loss
                    best_train_performance['accuracy'] = epoch_acc
                    best_train_performance['precision'] = epoch_pre
                    best_train_performance['recall'] = epoch_rec
                    best_train_performance['f1'] = epoch_f1
                    best_train_performance['roc_auc'] = epoch_auc
                # loss = F.nll_loss(out, torch.tensor(np.array(eeg_labels)[train_indices]).long())\

                # 测试模型
                model.eval()

                num_total = 0
                test_loss = 0

                all_preds_test = []
                all_labels_eval_test = []
                all_probs_test = []

                test_embeddings = []
                attention_scores = []
                return_attn = False
                if epoch + 1 == train_rounds:
                    return_attn = True

                with (torch.no_grad()):
                    for batch in val_loader:
                        batch = batch.to(device)
                        labels = batch.y.to(device)

                        # out = model(batch)
                        # out = out.float().view(-1)
                        # labels = labels.float()
                        # loss = criterion(out, labels)

                        # reconstructed_adj_matrix, out = model(
                        #     batch, return_attn=False)  # out包含原始分数，未进行sigmod（不能使用softmax）
                        # # print(out.shape)
                        # # print(reconstructed_adj_matrix.shape)
                        # edge_index = batch_edgeindex_to_dense(batch.edge_index, labels.shape[0], num_nodes)
                        # out = out.float().view(-1)
                        # labels = labels.float()
                        # # print(out)
                        # # print(labels)
                        # loss, loss_info = criterion(logits=reconstructed_adj_matrix,
                        #                             adj_target=edge_index,
                        #                             cls_logits=out,
                        #                             cls_target=labels)
                        # # print(loss)
                        # # print(loss_info)

                        if return_attn:
                            reconstructed_alpha_adj_matrix, reconstructed_beta_adj_matrix, reconstructed_theta_adj_matrix, out, val_graph_embeddings, weights, attn_info = model(
                                batch, return_attn=True)
                            attention_scores.append(attn_info)
                        else :
                            reconstructed_alpha_adj_matrix, reconstructed_beta_adj_matrix, reconstructed_theta_adj_matrix, out, val_graph_embeddings,weights = model(
                                batch, return_attn=False)  # out包含原始分数，未进行sigmod（不能使用softmax）
                        # print(out.shape)
                        # print(reconstructed_alpha_adj_matrix.shape)
                        edge_index_theta = to_dense_adj(batch.edge_index_theta, batch.batch, max_num_nodes=num_nodes)
                        edge_index_alpha = to_dense_adj(batch.edge_index_alpha, batch.batch, max_num_nodes=num_nodes)
                        edge_index_beta = to_dense_adj(batch.edge_index_beta, batch.batch, max_num_nodes=num_nodes)
                        out = out.float().view(-1)
                        labels = labels.float()
                        # print(out)
                        # print(labels)
                        loss, loss_info = criterion(logits_theta=reconstructed_theta_adj_matrix,
                                                    logits_alpha=reconstructed_alpha_adj_matrix,
                                                    logits_beta=reconstructed_beta_adj_matrix,
                                                    adj_target_theta=edge_index_theta,
                                                    adj_target_alpha=edge_index_alpha,
                                                    adj_target_beta=edge_index_beta,
                                                    cls_logits=out,
                                                    cls_target=labels)
                        # print(loss)
                        # print(loss_info)

                        probabilities = torch.sigmoid(out)
                        pred = (probabilities > 0.5).float()

                        # print(probabilities)
                        # print(pred)
                        #
                        # print(f"predict:{pred}\n")
                        # print(f"labels:{labels}\n")

                        # 取消每个batch将数据搬运回cpu，提高GPU利用效率
                        all_preds_test.append(pred)
                        all_probs_test.append(probabilities)
                        all_labels_eval_test.append(labels)

                        test_embeddings.append(val_graph_embeddings)
                        batch_subband_weights = weights

                        total_loss += loss.item()
                        total_num += labels.size(0)

                    y_true_test = torch.cat(all_labels_eval_test).detach().cpu().numpy().astype(int)
                    y_pred_test = torch.cat(all_preds_test).detach().cpu().numpy().astype(int)
                    y_prob_test = torch.cat(all_probs_test).detach().cpu().numpy()


                    if (epoch+1) == 5 or (epoch+1) == train_rounds:
                        epoch_test_embeddings = torch.cat(test_embeddings).detach().cpu().numpy()
                        save_epoch_clusters(embeddings=epoch_test_embeddings, labels=y_true_test, epoch=epoch+1)

                    if return_attn and (epoch + 1) == train_rounds:
                        print(y_prob_test[:200])
                        # 1. Initialize a dictionary to store concatenated scores for each band
                        bands = ['alpha', 'beta', 'theta']
                        band_importance_data = {band: [] for band in bands}

                        # 2. Iterate through the list of dictionaries and collect scores
                        for score_dict in attention_scores:
                            for band in bands:
                                if score_dict[band] is not None:
                                    band_importance_data[band].append(score_dict[band])

                        print(band_importance_data['alpha'][0], len(band_importance_data['beta']),
                              len(band_importance_data['theta']))
                        # 3. Process and Plot each band individually
                        for band in bands:
                            if band_importance_data[band]:
                                # Concatenate all batches for this band
                                # [num_samples, num_nodes]
                                scores_tensor = torch.cat(band_importance_data[band]).detach().cpu().numpy()
                                print(len(scores_tensor))
                                num_segments = scores_tensor.shape[0] // num_nodes

                                scores_matrix = scores_tensor.reshape(num_segments, num_nodes)
                                print(f"Reshaped {band} scores to: {scores_matrix.shape}")

                                # access specific sample using previous acquired specific_sample_idx
                                sample_prob = y_prob_test[specific_sample_idx]
                                print(sample_prob)
                                sample_score = scores_matrix[specific_sample_idx]

                                # Calculate final importance (Average + Normalization)
                                final_score, normalized_sample_score = calculate_final_importance(scores_matrix,
                                                                                                  sample_score)
                                # Plot the topomap with a unique filename for each band
                                plot_topomap_from_set(final_score, save_path=f'MODMA_topomap_{band}.svg')
                                plot_topomap_from_set(normalized_sample_score,
                                                      save_path=f'MODMA_sample470_MDbutSCIDnotperformed_topomap_{band}.svg')


                    epoch_subband_weights.append(batch_subband_weights.detach().cpu().numpy())
                    print(epoch_subband_weights)

                    epoch_avg_acc = accuracy_score(y_true_test, y_pred_test)
                    epoch_pre = precision_score(y_true_test, y_pred_test, zero_division=0)  # add handler for zero_division events
                    epoch_rec = recall_score(y_true_test, y_pred_test, zero_division=0)
                    epoch_f1 = f1_score(y_true_test, y_pred_test, zero_division=0)
                    epoch_auc = roc_auc_score(y_true_test, y_prob_test)
                    test_epoch_loss = total_loss / total_num

                    test_epoch_losses.append(test_epoch_loss)
                    test_accuracies.append(epoch_avg_acc)

                    print(
                        f'Test Accuracy: {epoch_avg_acc:.4f}, Test loss: {test_epoch_loss:.4f}, Test pre: {epoch_pre:.4f}, Test rec: {epoch_rec:.4f}, Test f1:{epoch_f1:.4f}, Test auc: {epoch_auc:.4f}')

                    if epoch_avg_acc > best_test_performance['accuracy']:
                        print('update best performance\n')
                        best_test_performance['epoch'] = epoch + 1
                        best_test_performance['test_loss'] = test_epoch_loss
                        best_test_performance['accuracy'] = epoch_avg_acc
                        best_test_performance['precision'] = epoch_pre
                        best_test_performance['recall'] = epoch_rec
                        best_test_performance['f1'] = epoch_f1
                        best_test_performance['roc_auc'] = epoch_auc

                scheduler.step(test_epoch_loss)

            k_fold_train_epoch_losses.append(train_epoch_losses)
            k_fold_test_epoch_losses.append(test_epoch_losses)
            k_fold_train_accuracies.append(train_accuracies)
            k_fold_test_accuracies.append(test_accuracies)
            k_fold_best_train_performances.append(best_train_performance)
            k_fold_best_test_performances.append(best_test_performance)

            numpyarray = np.array(epoch_subband_weights)
            print(numpyarray.shape)
            plot_subgraph_importance(numpyarray)

        #     # 1. 定义需要统计的指标名称
        # metric_keys = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        # final_results = {key: [] for key in metric_keys}
        #
        # # 2. 从每一折的最佳表现中提取数值
        # for best_perf in k_fold_best_test_performances:
        #     for key in metric_keys:
        #         final_results[key].append(best_perf[key])
        #
        # # add the redirct option to automate training and log saving
        # with open("ablationResults/result_log.txt", "a", encoding="utf-8") as f:
        #     with redirect_stdout(f):
        #
        #         # 3. 计算并打印平均值和标准差
        #         # print(f"cls:recon={cls_weight}:{recon_weight},lr={lr},withGATdropout,batchsize={batch_size}")
        #         print(f"removeCrossAttn,lr={lr},batchsize={batch_size},practice{practice+1},seed={seeds[practice]}")
        #         # print(f"GIN,lr={lr},batchsize={batch_size},practice{practice + 1}")
        #         print("\n" + "=" * 30)
        #         print(f"{k_fold}-Fold Cross Validation Final Results:")
        #         print("=" * 30)
        #
        #         for key in metric_keys:
        #             values = np.array(final_results[key])
        #             mean_val = np.mean(values)
        #             std_val = np.std(values)
        #             print(f"{key.upper():<10}: {mean_val:.4f} ± {std_val:.4f}")
        #
        #         print("-" * 30)
        #         for i, best_perf in enumerate(k_fold_best_test_performances):
        #             print(f"Fold {i + 1} Accuracy: {best_perf['accuracy']:.4f}")
        #
        #         for i in range(k_fold):
        #             print(f"Fold {i + 1} best train and test performance:")
        #             print(
        #                 f"Best train Accuracy: {k_fold_best_train_performances[i]['accuracy']:.4f} , train precision: {k_fold_best_train_performances[i]['precision']:.4f}, train recall:{k_fold_best_train_performances[i]['recall']:.4f}, train f1:{k_fold_best_train_performances[i]['f1']:.4f}, train roc-auc: {k_fold_best_train_performances[i]['roc_auc']:.4f}, train loss: {k_fold_best_train_performances[i]['train_loss']:.4f}, epoch: {k_fold_best_train_performances[i]['epoch']}")
        #             print(
        #                 f"Best test Accuracy: {k_fold_best_test_performances[i]['accuracy']:.4f} , test precision: {k_fold_best_test_performances[i]['precision']:.4f}, test recall: {k_fold_best_test_performances[i]['recall']:.4f}, test f1:{k_fold_best_test_performances[i]['f1']:.4f}, test roc-auc: {k_fold_best_test_performances[i]['roc_auc']:.4f}, test loss: {k_fold_best_test_performances[i]['test_loss']:.4f}, epoch: {k_fold_best_test_performances[i]['epoch']}")
        #
        #             epoches = [i for i in range(train_rounds)]
        #
        #             plt.figure(figsize=(12, 10))
        #
        #             plt.subplot(2, 1, 1)
        #             plt.title(f'Comparison of train and test losses')
        #             plt.plot(epoches, k_fold_train_epoch_losses[i], 'cyan', label=f'train epoch losses')
        #             plt.plot(epoches, k_fold_test_epoch_losses[i], 'gold', label=f'test epoch losses')
        #             plt.xlabel('Epoch')
        #             plt.ylabel('Loss')
        #             plt.legend()
        #
        #             plt.subplot(2, 1, 2)
        #             plt.title(f'Comparison of train and test accuracies')
        #             plt.plot(epoches, k_fold_train_accuracies[i], 'cyan', label=f'train accuracies')
        #             plt.plot(epoches, k_fold_test_accuracies[i], 'gold', label=f'test accuracies')
        #             plt.xlabel('Epoch')
        #             plt.ylabel('Accuracy')
        #             plt.legend()
        #
        #             plt.tight_layout()
        #             # plt.savefig(
        #             #     f"sheet4_practice{practice + 1}_5Fold_Fold{i + 1}_dualtask_noscale_GATandBNlayeradded_cls-recon_{cls_weight}-{recon_weight}_withGATdropout_variablebandweight,lr{lr},batchsize_{batch_size}.png")
        #             plt.savefig(
        #                 f"ablationResults/sheet3_practice{practice + 1}_5Fold_Fold{i + 1}_removeCrossAttn,lr{lr},batchsize_{batch_size},seed_{seeds[practice]}.png")
        #             # plt.savefig(
        #             #     f"GCN,GAT,GIN,GraphSageResults/sheet3_practice{practice + 1}_5Fold_Fold{i + 1}_GIN,lr{lr},batchsize_{batch_size}.png")
        #             plt.close()
        #         print('\n')













    # #普通数据集划分
    # train_loader, val_loader = create_train_test_loaders(graph_data_list,test_ratio=0.2, batch_size=16)
    # # train_loader, val_loader = create_train_test_loaders_optimized(graph_data_list, test_ratio=0.2, batch_size=16)
    #
    # num_features = graph_data_list[0].num_features
    # print("num_features:", num_features)
    # hidden_channels = 16
    # out_channel = 1         #使用单神经元输出
    # # global_feature_channels = len(scale_data_in_segments[0][0]) #使用6个维度的图属性
    # global_feature_channels = 0
    # recon_weight = 0.2
    # cls_weight = 0.8
    # train_rounds = 50
    #
    # # model = SimpleGCN(num_features, hidden_channels, num_classes)
    # # model = MultiBandGAT(num_features, hidden_channels,out_channel,global_feature_channels)
    # # model = SingleFullBandGAT(num_features, hidden_channels, out_channel,global_feature_channels)
    # # model = SingleFullBandGAT(num_features, hidden_channels, out_channel)
    # # model = MultiBandGATWithNodeImportance(num_features, hidden_channels, out_channel,global_feature_channels, heads=1)
    # model = NewMultiBandGATWithNodeImportance(num_features, hidden_channels, out_channel, global_feature_channels, heads=1)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=5e-4)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # # criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
    # criterion = MultiTaskLossModule(recon_weight=recon_weight, cls_weight=cls_weight, pos_weight_edge=None)
    # model.apply(weights_init)
    #
    # best_train_performance = {'epoch':0,'train_loss':float('inf'),'accuracy':0.0,'precision':0.0,'recall':0.0,'f1':0.0,'roc_auc':0.0,'loss':0.0}
    # best_test_performance = {'epoch':0,'accuracy':0.0,'precision':0.0,'recall':0.0,'f1':0.0,'roc_auc':0.0}
    #
    # print(model)
    #
    # model = model.to(device)
    #
    # train_epoch_losses = []
    # test_epoch_losses = []
    # train_accuracies = []
    # test_accuracies = []
    #
    # # 训练模型
    # for epoch in range(train_rounds):
    #     print('========== Epoch:', epoch + 1)
    #
    #     model.train()
    #     total_loss = 0
    #     total_num = 0
    #     total_correct = 0
    #     total_predicts = []
    #     total_probabilities = []
    #     total_labels = []
    #
    #     for batch in train_loader:
    #         batch = batch.to(device)
    #         labels = batch.y.to(device)
    #
    #         # out = model(batch)
    #         # out = out.float().view(-1)
    #         # labels = labels.float()
    #         # loss = criterion(out,labels)
    #
    #         reconstructed_alpha_adj_matrix, reconstructed_beta_adj_matrix, reconstructed_theta_adj_matrix, out = model(batch,return_attn = False) #out包含原始分数，未进行sigmod（不能使用softmax）
    #         print(out.shape)
    #         edge_index_theta = batch_edgeindex_to_dense(batch.edge_index_theta, labels.shape[0], num_nodes)
    #         edge_index_alpha = batch_edgeindex_to_dense(batch.edge_index_alpha, labels.shape[0], num_nodes)
    #         edge_index_beta = batch_edgeindex_to_dense(batch.edge_index_beta,  labels.shape[0], num_nodes)
    #         out = out.float().view(-1)
    #         labels = labels.float()
    #         loss,loss_info = criterion( logits_theta = reconstructed_theta_adj_matrix,
    #                           logits_alpha = reconstructed_alpha_adj_matrix,
    #                           logits_beta = reconstructed_beta_adj_matrix,
    #                           adj_target_theta = edge_index_theta,
    #                           adj_target_alpha = edge_index_alpha,
    #                           adj_target_beta = edge_index_beta,
    #                           cls_logits = out,
    #                           cls_target = labels)
    #
    #         optimizer.zero_grad()
    #         loss.backward()
    #         # for name, param in model.named_parameters():
    #         #     if param.grad is not None:
    #         #         print(
    #         #             f"Gradient for {name}: mean={param.grad.mean().item()}, std={param.grad.std().item()}")
    #         optimizer.step()
    #
    #
    #         probabilities = torch.sigmoid(out)
    #         pred = (probabilities > 0.5).float()
    #
    #         print(probabilities)
    #         print(pred)
    #
    #         print(f"predict:{pred}\n")
    #         print(f"labels:{labels}\n")
    #         total_correct += (pred == labels).sum().item()
    #         total_loss += loss.item()
    #         total_num += labels.size(0)
    #         mean_loss = total_loss / total_num
    #         print(f"num_correct:{total_correct},num_total:{total_num},loss:{mean_loss}\n")
    #
    #         total_predicts.append(pred.detach().cpu().numpy().astype(int))
    #         total_probabilities.append(probabilities.detach().cpu().numpy())
    #         total_labels.append(labels.detach().cpu().numpy().astype(int))
    #
    #     y_true = np.concatenate(total_labels).astype(int)
    #     y_pred = np.concatenate(total_predicts).astype(int)
    #     y_prob = np.concatenate(total_probabilities)
    #
    #     epoch_acc = accuracy_score(y_true, y_pred)
    #     epoch_pre = precision_score(y_true, y_pred)
    #     epoch_rec = recall_score(y_true, y_pred)
    #     epoch_f1 = f1_score(y_true, y_pred)
    #     epoch_auc = roc_auc_score(y_true, y_prob)
    #     train_epoch_loss = total_loss / total_num
    #
    #     train_epoch_losses.append(train_epoch_loss)
    #     train_accuracies.append(epoch_acc)
    #
    #
    #     print(f'Epoch: {epoch + 1}, Train loss: {train_epoch_loss:.4f}, Train acc: {epoch_acc:.4f}, Train pre: {epoch_pre:.4f}, '
    #           f'Train f1: {epoch_f1:.4f}, Train rec: {epoch_rec:.4f}, Train auc: {epoch_auc:.4f}')
    #
    #     if epoch_acc > best_train_performance['accuracy'] and train_epoch_loss < best_train_performance['train_loss']:
    #         print('update best performance\n')
    #         best_train_performance['epoch'] = epoch + 1
    #         best_train_performance['train_loss'] = train_epoch_loss
    #         best_train_performance['accuracy'] = epoch_acc
    #         best_train_performance['precision'] = epoch_pre
    #         best_train_performance['recall'] = epoch_rec
    #         best_train_performance['f1'] = epoch_f1
    #         best_train_performance['roc_auc'] = epoch_auc
    #     # loss = F.nll_loss(out, torch.tensor(np.array(eeg_labels)[train_indices]).long())\
    #
    #
    #     # 测试模型
    #     model.eval()
    #     num_correct = 0
    #     num_total = 0
    #     test_loss = 0
    #     total_predicts = []
    #     total_probabilities = []
    #     total_labels = []
    #
    #     with (torch.no_grad()):
    #         for batch in val_loader:
    #             batch = batch.to(device)
    #             labels = batch.y.to(device)
    #
    #             # out = model(batch)
    #             # out = out.float().view(-1)
    #             # labels = labels.float()
    #             # loss = criterion(out, labels)
    #
    #             reconstructed_alpha_adj_matrix, reconstructed_beta_adj_matrix, reconstructed_theta_adj_matrix, out = model(batch,return_attn = False)  # out包含原始分数，未进行sigmod（不能使用softmax）
    #             print(out.shape)
    #             print(reconstructed_alpha_adj_matrix.shape)
    #             edge_index_theta = batch_edgeindex_to_dense(batch.edge_index_theta, labels.shape[0], num_nodes)
    #             edge_index_alpha = batch_edgeindex_to_dense(batch.edge_index_alpha, labels.shape[0], num_nodes)
    #             edge_index_beta = batch_edgeindex_to_dense(batch.edge_index_beta, labels.shape[0], num_nodes)
    #             out = out.float().view(-1)
    #             labels = labels.float()
    #             # print(out)
    #             # print(labels)
    #             loss,loss_info = criterion(logits_theta=reconstructed_theta_adj_matrix,
    #                              logits_alpha=reconstructed_alpha_adj_matrix,
    #                              logits_beta=reconstructed_beta_adj_matrix,
    #                              adj_target_theta=edge_index_theta,
    #                              adj_target_alpha=edge_index_alpha,
    #                              adj_target_beta=edge_index_beta,
    #                              cls_logits=out,
    #                              cls_target=labels)
    #             print(loss)
    #             print(loss_info)
    #
    #             probabilities = torch.sigmoid(out)
    #             pred = (probabilities > 0.5).float()
    #
    #             print(probabilities)
    #             print(pred)
    #
    #
    #             print(f"predict:{pred}\n")
    #             print(f"labels:{labels}\n")
    #             num_correct += (pred == labels).sum().item()
    #             num_total += labels.size(0)
    #             test_loss += loss.item()
    #             mean_loss = test_loss / num_total
    #             print(f"num_correct:{num_correct},num_total:{num_total},loss:{mean_loss}\n")
    #
    #             total_predicts.append(pred.detach().cpu().numpy().astype(int))
    #             total_probabilities.append(probabilities.detach().cpu().numpy())
    #             total_labels.append(labels.detach().cpu().numpy().astype(int))
    #
    #     y_true = np.concatenate(total_labels).astype(int)
    #     y_pred = np.concatenate(total_predicts).astype(int)
    #     y_prob = np.concatenate(total_probabilities)
    #
    #     epoch_pre = precision_score(y_true, y_pred)
    #     epoch_rec = recall_score(y_true, y_pred)
    #     epoch_f1 = f1_score(y_true, y_pred)
    #     epoch_auc = roc_auc_score(y_true, y_prob)
    #     epoch_avg_acc = num_correct / num_total
    #     test_epoch_loss = test_loss / num_total
    #
    #     test_epoch_losses.append(test_epoch_loss)
    #     test_accuracies.append(epoch_avg_acc)
    #
    #     print(f'Test Accuracy: {epoch_avg_acc:.4f}, Test loss: {test_epoch_loss:.4f}, Test pre: {epoch_pre:.4f}, Test rec: {epoch_rec:.4f}, Test f1:{epoch_f1:.4f}, Test auc: {epoch_auc:.4f}')
    #
    #
    #     if epoch_avg_acc > best_test_performance['accuracy'] :
    #         print('update best performance\n')
    #         best_test_performance['epoch'] = epoch + 1
    #         best_test_performance['test_loss'] = test_epoch_loss
    #         best_test_performance['accuracy'] = epoch_avg_acc
    #         best_test_performance['precision'] = epoch_pre
    #         best_test_performance['recall'] = epoch_rec
    #         best_test_performance['f1'] = epoch_f1
    #         best_test_performance['roc_auc'] = epoch_auc
    #
    #     scheduler.step(test_epoch_loss)
    #
    # print(f'Best train Accuracy: {best_train_performance['accuracy']:.4f} , train precision: {best_train_performance['precision']:.4f}, train recall: {best_train_performance['recall']:.4f}, train f1:{best_train_performance['f1']:.4f}, train roc-auc: {best_train_performance['roc_auc']:.4f}, train loss: {best_train_performance['train_loss']:.4f}, epoch: {best_train_performance['epoch']}')
    # print(f'Best test Accuracy: {best_test_performance['accuracy']:.4f} , test precision: {best_test_performance['precision']:.4f}, test recall: {best_test_performance['recall']:.4f}, test f1:{best_test_performance['f1']:.4f}, test roc-auc: {best_test_performance['roc_auc']:.4f}, test loss: {best_test_performance['test_loss']:.4f}, epoch: {best_test_performance['epoch']}')
    #
    #
    # epoches = [i for i in range(train_rounds)]
    # #loss函数可视化
    # plt.figure(figsize=(12, 10))
    #
    # # 绘制原始信号
    # plt.subplot(2, 1, 1)
    # plt.title(f'Comparison of train and test losses')
    # plt.plot(epoches, train_epoch_losses,'cyan', label=f'train epoch losses')
    # plt.plot(epoches, test_epoch_losses,'gold', label=f'test epoch losses')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.legend()
    #
    # # 绘制过滤后的信号
    # plt.subplot(2, 1, 2)
    # plt.title(f'Comparison of train and test accuracies')
    # plt.plot(epoches, train_accuracies,'cyan', label=f'train accuracies')
    # plt.plot(epoches, test_accuracies,'gold', label=f'test accuracies')
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy')
    # plt.legend()
    #
    # plt.tight_layout()
    # plt.show()

                # file.write(f'fold:{fold + 1},epoch:{epoch + 1},acc:{epoch_avg_acc:.4f},loss:{epoch_loss:.4f}\n')