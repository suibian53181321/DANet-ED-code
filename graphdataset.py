import torch

class GraphDataset(Dataset):
    def __init__(self, graphs, labels):
        """
        初始化图数据集
        :param graphs: 包含所有图的数据列表
        :param labels: 包含所有图的标签列表
        """
        self.graphs = graphs
        self.labels = labels

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        """
        获取单个图及其标签
        """
        graph = self.graphs[idx]
        label = self.labels[idx]

        # 假设graph已经是Data对象，并且label是一个tensor或可以转换为tensor的格式
        # 如果不是，请根据实际情况调整
        return graph, torch.tensor(label, dtype=torch.long)