import torch
import numpy as np
from torch_geometric.utils import to_dense_adj
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.utils import to_dense_adj
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, homogeneity_score
from sklearn.manifold import TSNE
from funcs import load_embedding_and_matrix_full_band, load_embedding_and_matrix
import matplotlib
matplotlib.use('Agg')

if __name__ == '__main__':
    lmdb_dir = '../EEG_128channels_resting_lanzhou_2015/PCC'
    subject_id_txt = 'subject_id.txt'

    # subject_segments_id_list, graph_data_list, labels, assistance_information_list, eeg_subject_ids = load_embedding_and_matrix_full_band(subject_id_txt, lmdb_dir)
    # # 1. Feature Extraction
    # # Since the structure is identical, we can flatten 'x' and 'edge_attr'
    # # into a single vector for each graph.
    # features = []
    # labels_true = []
    #
    # # Assume num_nodes is 128 for your EEG data
    # num_nodes = 68
    #
    # for data in graph_data_list:
    #     # 1. Handle Node Features (x)
    #     # Shape: [num_nodes * num_node_features]
    #     node_feats = data.x.view(-1).numpy()
    #
    #     # 2. Handle Edges (The Threshold Problem)
    #     # We convert sparse edges back into a dense 128x128 matrix
    #     # If edge_attr exists, we use it as the weights; otherwise, use 1s
    #     if hasattr(data, 'edge_attr') and data.edge_attr is not None:
    #         # to_dense_adj returns [batch, nodes, nodes]. We take index 0.
    #         adj = to_dense_adj(data.edge_index, edge_attr=data.edge_attr, max_num_nodes=num_nodes)[0]
    #     else:
    #         adj = to_dense_adj(data.edge_index, max_num_nodes=num_nodes)[0]
    #
    #     # Flatten the matrix: [128, 128] -> [16384]
    #     edge_feats = adj.numpy().flatten()
    #
    #     # 3. Combine
    #     combined_feats = np.concatenate([node_feats, edge_feats])
    #
    #     features.append(combined_feats)
    #     # labels_true.append(data.y)
    #
    # for (idx,item) in enumerate(assistance_information_list):
    #     labels_true.extend([idx+1]*item['total subject segments'])
    # print(labels_true)
    # # Now all vectors in 'features' are guaranteed to be the same length
    # X = np.array(features)
    # y_true = np.array(labels_true)
    #
    # # 4. Clustering & Validation
    # kmeans = KMeans(n_clusters=len(assistance_information_list), random_state=42, n_init='auto')
    # y_pred = kmeans.fit_predict(X)
    #
    # print(f"--- Subject-Level Clustering Results ---")
    # print(f"Total Unique Subjects: {len(X)}")
    # print(f"Adjusted Rand Index (ARI): {adjusted_rand_score(y_true, y_pred):.4f}")
    # print(f"Normalized Mutual Info (NMI): {normalized_mutual_info_score(y_true, y_pred):.4f}")
    #
    #
    # # 5. Dimensionality Reduction for Visualization
    # tsne = TSNE(n_components=2, perplexity=min(30, len(X) - 1), random_state=42)
    # X_embedded = tsne.fit_transform(X)
    #
    # # 6. Plotting with Discrete Legend
    # plt.figure(figsize=(10, 6))
    #
    # # We use a qualitative colormap like 'Set1' or 'tab10' for distinct marks
    # scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y_true, cmap='Set1', alpha=0.7, edgecolors='w')
    #
    # # Create the discrete legend
    # # scatter.legend_elements() automatically creates handles for the unique values in 'c'
    # plt.legend(*scatter.legend_elements(), title="Subject Labels")
    #
    # plt.title("t-SNE Visualization of EEG Graph Clusters")
    # plt.xlabel("Dimension 1")
    # plt.ylabel("Dimension 2")
    # plt.grid(True, linestyle='--', alpha=0.5)
    # plt.show()


    # subject_segments_id_list, graph_data_list, labels, assistance_information_list, eeg_subject_ids = load_embedding_and_matrix_full_band(subject_id_txt, lmdb_dir)
    subject_segments_id_list, graph_data_list, labels, assistance_information_list, eeg_subject_ids = load_embedding_and_matrix(subject_id_txt, lmdb_dir)

    import numpy as np
    import pandas as pd
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score

    # 1. Feature Extraction
    num_nodes = 68
    segment_features = []
    father_ground_truth = []  # The actual subject ID for each segment

    # Build a ground truth list where each segment is labeled by its father's ID
    for i, info in enumerate(assistance_information_list):
        n_segs = info['total subject segments']
        father_id = i  # Using index as a unique ID for clustering comparison

        # Identify the slice of data in graph_data_list
        start_idx = sum(d['total subject segments'] for d in assistance_information_list[:i])

        # for j in range(start_idx, start_idx + n_segs):
        #     data = graph_data_list[j]
        #     node_feats = data.x.view(-1).numpy()
        #     adj = to_dense_adj(data.edge_index, edge_attr=data.edge_attr, max_num_nodes=num_nodes)[0]
        #     edge_feats = adj.numpy().flatten()
        #
        #     segment_features.append(np.concatenate([node_feats, edge_feats]))
        #     father_ground_truth.append(father_id)

        for j in range(start_idx, start_idx + n_segs):
            data = graph_data_list[j]
            node_feats_alpha = data.x_alpha.view(-1).numpy()
            node_feats_theta = data.x_alpha.view(-1).numpy()
            node_feats_beta = data.x_beta.view(-1).numpy()

            adj_alpha = to_dense_adj(data.edge_index_alpha, edge_attr=data.edge_attr_alpha, max_num_nodes=num_nodes)[0]
            edge_feats_alpha = adj_alpha.numpy().flatten()
            adj_theta = to_dense_adj(data.edge_index_theta, edge_attr=data.edge_attr_theta, max_num_nodes=num_nodes)[0]
            edge_feats_theta = adj_theta.numpy().flatten()
            adj_beta = to_dense_adj(data.edge_index_beta, edge_attr=data.edge_attr_beta, max_num_nodes=num_nodes)[0]
            edge_feats_beta = adj_beta.numpy().flatten()

            segment_features.append(np.concatenate([node_feats_alpha, edge_feats_alpha,node_feats_beta, edge_feats_beta, node_feats_theta, edge_feats_theta]))
            father_ground_truth.append(father_id)

    X = np.array(segment_features)
    y_father = np.array(father_ground_truth)

    # 2. Clustering into N groups (N = Number of Fathers)
    num_fathers = len(assistance_information_list)
    print(f"Clustering {len(X)} segments into {num_fathers} subject-based groups...")

    kmeans = KMeans(n_clusters=num_fathers, random_state=42, n_init='auto')
    cluster_labels = kmeans.fit_predict(X)

    # 3. Evaluation: Homogeneity
    # Homogeneity measures if each cluster contains only members of a single father
    h_score = homogeneity_score(y_father, cluster_labels)
    ari_score = adjusted_rand_score(y_father, cluster_labels)

    print(f"\n--- Results ---")
    print(f"Homogeneity Score: {h_score:.4f} (1.0 = All sons stayed with their father)")
    print(f"Adjusted Rand Index: {ari_score:.4f}")

    # 4. Visualizing the "Subject Clusters"
    # We'll use t-SNE and color by Father ID.
    # If they are easily separated, you'll see N distinct, tight clusters.
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_embedded = tsne.fit_transform(X)

    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y_father, cmap='turbo', alpha=0.6, s=10)

    # 1. Get unique father IDs
    unique_fathers = np.unique(y_father)

    print(unique_fathers)

    # 1. Create a colormap that can handle 47 distinct values
    # We use 'gist_rainbow' or 'turbo' because they have a wide range of distinct hues
    num_fathers = len(unique_fathers)
    colors = plt.cm.get_cmap('turbo', num_fathers)

    # 2. Update the scatter plot to use the specific number of colors
    # Instead of passing a string 'tab20', we pass the specific color mapping
    plt.figure(figsize=(12, 8))  # Slightly wider figure to accommodate the legend
    scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1],
                          c=y_father,
                          cmap=colors,
                          alpha=0.6,
                          s=10)
    for f_id in np.unique(y_father):
        mask = (y_father == f_id)
        centroid = X_embedded[mask].mean(axis=0)
        plt.text(centroid[0], centroid[1], str(int(f_id)),
                 fontsize=9, fontweight='bold', ha='center')

    # 3. Force legend_elements to return ALL 47 fathers
    # The 'num' parameter tells it exactly how many markers to create
    handles, _ = scatter.legend_elements(num=num_fathers)

    # 4. Create the legend with custom labels
    plt.legend(handles, [f"Father {int(f)}" for f in unique_fathers],
               title="Original Subjects",
               loc='upper left',
               bbox_to_anchor=(1.02, 1),
               ncol=4,
               fontsize='x-small',  # Use 'x-small' if 'small' still gets cut off
               columnspacing=0.6)

    # 5. Expand the layout rectangle to prevent cutting
    plt.tight_layout(rect=[0, 0, 0.82, 1])  # Adjust layout to make room for the legend
    plt.savefig(
        f"visualizationResults/original_subject_distribution_cluster.png")