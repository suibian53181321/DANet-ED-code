import torch
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.utils import to_dense_adj
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from matplotlib.colors import ListedColormap
import matplotlib
matplotlib.use('Agg')
import os

def cluster_test():
    from funcs import load_embedding_and_matrix

    lmdb_dir = '../EEG_128channels_resting_lanzhou_2015/PCC'
    subject_id_txt = 'subject_id.txt'

    subject_segments_id_list, graph_data_list, labels, assistance_information_list, eeg_subject_ids = load_embedding_and_matrix(subject_id_txt, lmdb_dir)

    import numpy as np
    import matplotlib.pyplot as plt
    from torch_geometric.utils import to_dense_adj
    from sklearn.cluster import KMeans
    from sklearn.manifold import TSNE
    from sklearn.metrics import adjusted_rand_score

    # --- 1. Feature Extraction & Label Setup ---
    num_nodes = 68
    segment_features = []
    labels_ground_truth = []  # The binary labels (e.g., 0 or 1)
    father_ids = []  # To keep track of which son belongs to which father

    for i, info in enumerate(assistance_information_list):
        n_segs = info['total subject segments']
        label = info['origin subject label']
        f_id = info['origin subject id']

        # Calculate starting index for this father's segments in graph_data_list
        start_idx = sum(d['total subject segments'] for d in assistance_information_list[:i])

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
            labels_ground_truth.append(label)
            father_ids.append(f_id)

    X = np.array(segment_features)
    y_true = np.array(labels_ground_truth)
    y_father = np.array(father_ids)

    # --- 2. Cluster into 2 Groups ---
    kmeans = KMeans(n_clusters=2, random_state=42, n_init='auto')
    y_pred = kmeans.fit_predict(X)

    # --- 3. Visualization with 2 Colors for Ground Truth ---
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_embedded = tsne.fit_transform(X)

    plt.figure(figsize=(12, 8))

    custom_cmap = ListedColormap(['tab:blue', 'tab:orange'])

    # Use a colormap with only 2 distinct colors (e.g., 'coolwarm' or 'bwr')
    # We color by 'y_true' (Ground Truth Labels) instead of Father ID
    scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1],
                          c=y_true,
                          cmap=custom_cmap,
                          alpha=0.5,
                          s=15,
                          edgecolors='none')

    # Create a discrete legend for the 2 classes
    handles, _ = scatter.legend_elements()
    plt.legend(handles, ["HC", "MD"], title="Ground Truth Labels", loc='upper right')

    # # Plot Father IDs at the centroid of their sons to check for identity separation
    # for f_id in np.unique(y_father):
    #     mask = (y_father == f_id)
    #     centroid = X_embedded[mask].mean(axis=0)
    #     plt.text(centroid[0], centroid[1], str(int(f_id)),
    #              fontsize=8, fontweight='bold', ha='center', alpha=0.8)

    # plt.title("Clustering of Son Subjects (Colored by Ground Truth Labels)")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.savefig(
        f"visualizationResults/original_subject_distribution_cluster_label_separate.png")

    print(f"Adjusted Rand Index (ARI): {adjusted_rand_score(y_true, y_pred):.4f}")


def save_epoch_clusters(embeddings, labels, epoch, save_dir="visualizationResults"):
    """
    Visualizes and saves the cluster plot for a specific epoch.

    Args:
        embeddings (numpy.ndarray): Feature/Embedding matrix (num_samples, num_features).
        labels (numpy.ndarray): Ground truth labels (0 or 1).
        epoch (int): The current epoch number (0, 10, or 50).
        save_dir (str): Directory where the .svg will be saved.
    """
    # 1. Ensure the save directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 2. Dimensionality Reduction using t-SNE
    print(f"Generating t-SNE for Epoch {epoch}...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
    X_2d = tsne.fit_transform(embeddings)

    # 3. Setup the Visualization
    plt.figure(figsize=(10, 7))
    custom_cmap = ListedColormap(['tab:blue', 'tab:orange'])

    # Plot segments colored by Ground Truth
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1],
                          c=labels,
                          cmap=custom_cmap,
                          alpha=0.6,
                          s=15,
                          edgecolors='none')

    # 4. Add Legend and Labels
    handles, _ = scatter.legend_elements()
    plt.legend(handles, ["HC (Blue)", "MD (Orange)"],
               title="Ground Truth", loc='upper right')

    # title_prefix = "Raw Features" if epoch == 0 else "Learned Embeddings"
    # plt.title(f"{title_prefix} - Epoch {epoch}")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.grid(True, linestyle='--', alpha=0.3)

    # 5. Save the Figure
    filename = f"cluster_epoch_{epoch}.svg"
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight',format="svg")
    plt.close()  # Close to free up memory
    print(f"Successfully saved plot to {save_path}")

def transform_graph_data_into_embeddings(graph_data_list,num_nodes):
    segment_features = []
    segment_labels = []

    for i in range(len(graph_data_list)):
        data = graph_data_list[i]
        node_feats_alpha = data.x_alpha.view(-1).numpy()
        node_feats_theta = data.x_alpha.view(-1).numpy()
        node_feats_beta = data.x_beta.view(-1).numpy()
        label = data.y

        adj_alpha = to_dense_adj(data.edge_index_alpha, edge_attr=data.edge_attr_alpha, max_num_nodes=num_nodes)[0]
        edge_feats_alpha = adj_alpha.numpy().flatten()
        adj_theta = to_dense_adj(data.edge_index_theta, edge_attr=data.edge_attr_theta, max_num_nodes=num_nodes)[0]
        edge_feats_theta = adj_theta.numpy().flatten()
        adj_beta = to_dense_adj(data.edge_index_beta, edge_attr=data.edge_attr_beta, max_num_nodes=num_nodes)[0]
        edge_feats_beta = adj_beta.numpy().flatten()

        segment_features.append(np.concatenate([node_feats_alpha, edge_feats_alpha, node_feats_beta, edge_feats_beta, node_feats_theta, edge_feats_theta]))
        segment_labels.append(label)

    X = np.array(segment_features)
    y_true = np.array(segment_labels)
    return X, y_true

if __name__ == "__main__":
    cluster_test()