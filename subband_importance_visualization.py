import numpy as np
import matplotlib.pyplot as plt


def plot_subgraph_importance(importance_array, save_path='subgraph_importance.svg'):
    """
    Visualizes the importance of three EEG frequency bands over 50 epochs.

    Args:
        importance_array (numpy.ndarray): Array of shape (50, 3) containing
                                         importance weights for each band.
        save_path (str): The filename to save the resulting line chart.
    """
    # 1. Setup the x-axis (Epochs 1 to 50)
    epochs = np.arange(1, importance_array.shape[0]+1)
    band_names = [r'$\theta$ (Theta)', r'$\alpha$ (Alpha)', r'$\beta$ (Beta)']
    colors = ['tab:blue', 'tab:orange', 'tab:green']

    # 2. Plotting each band's importance
    # We iterate through the columns of the (50, 3) array
    for i in range(3):
        plt.plot(epochs, importance_array[:, i],
                 label=band_names[i],
                 color=colors[i],
                 linewidth=2)

    # 3. Customizing the Chart for Academic Publication
    # plt.title('Frequency Band Importance Weights over 50 Epochs', fontsize=14)
    plt.xlabel('Training Epoch', fontsize=12)
    plt.ylabel('Importance Weight', fontsize=12)
    plt.legend(loc='best', frameon=True)
    plt.grid(True, linestyle='--', alpha=0.6)

    # 4. Save the figure (Avoids .show() as per guidelines)
    plt.savefig(save_path, dpi=300, bbox_inches='tight',format="svg")
    plt.close()  # Free memory
    print(f"Line chart successfully saved to {save_path}")

# Example Usage:
# # Assuming 'importance_data' is your (50, 3) numpy array
# plot_subgraph_importance(importance_data)


import numpy as np


def calculate_final_importance(importance_array, sample_importance):
    """
    Processes raw importance data to produce normalized global and local importance scores.

    Args:
        importance_array (numpy.ndarray): Raw scores aggregated from GAT modules.
                                          Expected shape: [num_samples, 68]
        sample_importance (numpy.ndarray): Raw scores for a specific case study sample.
                                           Expected shape: (68,)

    Returns:
        tuple: (final_importance, normalized_sample_importance)
               Both are normalized (0-1) based on the global min-max bounds.
    """
    # 1. Compute Global Average Importance
    if importance_array.ndim > 1:
        final_importance = np.mean(importance_array, axis=0)
    else:
        final_importance = importance_array

    # 2. Joint Min-Max Normalization
    # We find the min/max across the global average to establish the baseline scale
    imp_min = final_importance.min()
    imp_max = final_importance.max()

    if imp_max > imp_min:
        # Normalize the global average
        final_importance = (final_importance - imp_min) / (imp_max - imp_min)

        # Normalize the specific sample using the same global parameters
        # This keeps the comparison "honest" relative to the dataset average
        normalized_sample_importance = (sample_importance - imp_min) / (imp_max - imp_min)

        # Optional: Clip values to [0, 1] if the sample importance exceeds global bounds
        normalized_sample_importance = np.clip(normalized_sample_importance, 0, 1)
    else:
        normalized_sample_importance = sample_importance

    return final_importance, normalized_sample_importance


def get_avg_channel_importance(model, loader, device):
    model.eval()
    total_importance = torch.zeros(68).to(device)  # For your 68 channels
    count = 0

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            # Ensure your model's forward or a specific method returns node_importance
            _, node_importance = model.extract_importance(data.x, data.edge_index)

            # Since the data is batched, we need to handle the batch indices
            # If you are visualizing a single graph representative,
            # you can just sum the importance within the batch
            total_importance += node_importance.view(-1, 68).mean(dim=0)
            count += 1

    return (total_importance / count).cpu().numpy()



import mne
import matplotlib.pyplot as plt


def plot_topomap_from_set(importance_values, set_file_path="02020022rest 20150707 1452..set", save_path='topomap_from_set.svg'):
    """
    Reads channel locations directly from an EEGLAB .set file and plots a topomap.
    """
    # 1. Read the EEGLAB .set file (header only to save memory)
    # preload=False is sufficient since we only need the info/montage
    raw = mne.io.read_raw_eeglab(set_file_path, preload=False)

    # 2. Extract the info and montage
    # MNE automatically parses the locations from the .set file
    info = raw.info

    # 3. Optional: Filter to your 68 channels if the .set file has more
    # If importance_values is 68, ensure the info object matches
    if len(info['ch_names']) > len(importance_values):
        # Pick only the first 68 channels or use specific names if known
        info = mne.pick_info(info, sel=range(len(importance_values)))

    # 4. Plotting
    fig, ax = plt.subplots(figsize=(8, 8))
    im, _ = mne.viz.plot_topomap(
        importance_values,
        info,
        axes=ax,
        show=False,
        cmap='YlOrRd',
        extrapolate='head'
    )

    plt.colorbar(im, ax=ax, label='Importance Weight')
    # plt.title('Spatial Importance (Locations from .set)')

    plt.savefig(save_path, dpi=300, bbox_inches='tight',format="svg")
    plt.close()
    print(f"Topomap saved using locations from {set_file_path}")

# Example Usage:
# plot_topomap_from_set(final_importance_score, 'your_data.set')