import math

from sklearn.neural_network import MLPClassifier
from torch_geometric.nn import GCNConv,global_mean_pool,GATConv
from torch.nn import Linear, Sequential, ReLU, Dropout, BatchNorm1d, Parameter
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
import torch.nn.functional as F
import torch




def print_tensor_info(tensor, name):
    print(f"{name}: mean={tensor.mean().item()}, std={tensor.std().item()}, unique_values={torch.unique(tensor).tolist()}")



def apply_bn_activation(x, bn: BatchNorm1d = None, activation=F.elu):              #将bn层和activation层统一处理，为了避免由于先使用非线性后使用bn导致均值偏移以及改变非线性，所以采用先bn后activation

    if bn is not None:
        x = bn(x)
    if activation is not None:
        x = activation(x)
    return x





class SimplestFullBandGAT(torch.nn.Module):
    def __init__(self, in_channels,  out_channels, heads=8):
        super(SimplestFullBandGAT, self).__init__()
        self.mlp = Sequential(
            Linear(in_channels, 8),  # 输入维度假设为 10
            ReLU(),
            Dropout(0.2),
            Linear(8, out_channels)  # 输出单个值用于二分类任务
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = global_mean_pool(x, batch)
        return self.mlp(x)

# 定义简单的GCN模型
class SimpleGCN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels):
        super(SimpleGCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.mlp = Sequential(Linear(hidden_channels, 1))

    def forward(self, data):
        x, edge_index,batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        x = global_mean_pool(x, batch)
        x = self.mlp(x)
        return x


class SingleFullBandGAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=8):
        super(SingleFullBandGAT, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels)
        self.conv2 = GATConv(hidden_channels, hidden_channels)
        self.conv3 = GATConv(hidden_channels, 2*hidden_channels)
        self.conv4 = GATConv(2*hidden_channels, hidden_channels)
        self.mlp1 = Sequential(
            Linear(hidden_channels, 64),
            # Linear(hidden_channels + global_feature_channels, 64),
            ReLU(True),
            # Dropout(0.1),
            Linear(64, 16),
            ReLU(True),
            # Dropout(0.3),
            Linear(16, 8),
            ReLU(True),
            # Dropout(0.5),
            Linear(8, out_channels)
        )
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # x, edge_index, batch, global_feature = data.x, data.edge_index, data.batch, data.global_feature
        x = self.conv1(x, edge_index).relu()
        # print_tensor_info(x, "After Conv1")
        # x = F.dropout(x, p=0.2, training=self.training)
        x = self.bn1(x)
        x = self.conv2(x, edge_index).relu()
        # print_tensor_info(x, "After Conv2")
        # x = F.dropout(x, p=0.2, training=self.training)
        x = self.bn1(x)
        x = global_mean_pool(x, batch)
        # print_tensor_info(x, "After Pooling")
        # print(x.shape)
        # print(x.shape,global_feature.shape)
        # x = torch.cat([x, global_feature], dim=1)
        x = self.mlp1(x)

        return x








#初始的没有important node prediction模块的多波段gat
class MultiBandGAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, global_feature_channels, heads=8):
        super(MultiBandGAT, self).__init__()
        self.conv_alpha = GATConv(in_channels, hidden_channels)
        self.conv_beta = GATConv(in_channels, hidden_channels)
        self.conv_theta = GATConv(in_channels, hidden_channels)
        self.conv_combined = GATConv(hidden_channels * 3, hidden_channels)
        self.conv_final = GATConv(hidden_channels , hidden_channels)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.mlp1 = Sequential(                                                 #mlp1为二分类任务所使用的mlp分类器
            # Linear(hidden_channels, 64),
            Linear(hidden_channels + global_feature_channels, 64),
            ReLU(True),
            # Dropout(0.1),
            Linear(64, 16),
            ReLU(True),
            # Dropout(0.3),
            Linear(16, 8),
            ReLU(True),
            # Dropout(0.1),
            Linear(8, out_channels)
        )

    def forward(self, data):
        # (x_alpha , x_beta , x_theta , edge_index_alpha , edge_index_beta , edge_index_theta ,edge_attr_alpha , edge_attr_beta , edge_attr_theta ,batch ) = \
        # (data.x_alpha , data.x_beta , data.x_theta , data.edge_index_alpha , data.edge_index_beta , data.edge_index_theta , data.edge_attr_alpha , data.edge_attr_beta , data.edge_attr_theta, data.batch)

        (x_alpha, x_beta, x_theta, edge_index_alpha, edge_index_beta, edge_index_theta, edge_attr_alpha, edge_attr_beta, edge_attr_theta, batch , global_feature) = \
        (data.x_alpha, data.x_beta, data.x_theta, data.edge_index_alpha, data.edge_index_beta, data.edge_index_theta,data.edge_attr_alpha, data.edge_attr_beta, data.edge_attr_theta, data.batch, data.global_feature)

        # print('x.shape:',x.shape,'edge_index_alpha.shape', edge_index_alpha.shape,'edge_index_beta.shape',edge_index_beta.shape,'edge_index_theta.shape',edge_index_theta.shape,'edge_attr_alpha.shape',edge_attr_alpha.shape,'edge_attr_beta.shape', edge_attr_beta.shape,'edge_attr_theta.shape', edge_attr_theta.shape)
        x_alpha = self.conv_alpha(x_alpha, edge_index_alpha, edge_attr_alpha).relu()
        x_alpha = self.bn1(x_alpha)
        # x_alpha = F.dropout(x_alpha, p=0.2, training=self.training)

        x_beta = self.conv_beta(x_beta, edge_index_beta, edge_attr_beta).relu()
        x_beta = self.bn1(x_beta)
        # x_beta = F.dropout(x_beta, p=0.2, training=self.training)

        x_theta = self.conv_theta(x_theta, edge_index_theta, edge_attr_theta).relu()
        x_theta = self.bn1(x_theta)
        # x_theta = F.dropout(x_theta, p=0.2, training=self.training)

        # 合并三个波段的特征
        x_combined = torch.cat([x_alpha, x_beta, x_theta], dim=-1)

        # 再次应用GAT层以融合跨波段信息
        x_fused = self.conv_combined(x_combined, edge_index_alpha, edge_attr_alpha).relu()  # 使用任意一个波段的边索引即可
        x_fused = self.bn1(x_fused)
        # x_fused = F.dropout(x_fused, p=0.2, training=self.training)

        # 应用最终的GAT层
        # print(x_fused.shape,edge_index_alpha.shape,edge_attr_alpha.shape)
        x_out = self.conv_final(x_fused, edge_index_alpha, edge_attr_alpha)  # 使用任意一个波段的边索引即可
        x_out = self.bn1(x_out)
        # x_out = F.dropout(x_out, p=0.2, training=self.training)

        print('x_out:',x_out.std(dim=0).mean())

        x = global_mean_pool(x_out, batch)

        x = torch.cat([x, global_feature], dim=1)

        x = self.mlp1(x)

        return x





#有important node prediction模块的multi-band gat
class GATEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, heads=1):                 #在这里先不加入多头机制，heads等于1
        super().__init__()
        self.in_channels = in_channels
        self.heads = 1
        self.hidden_channels = hidden_channels
        # self.dropout = dropout

        self.gat_band1_alpha = GATConv(in_channels, hidden_channels, heads=heads, concat=True)
        self.bn_band1_alpha = BatchNorm1d(hidden_channels * heads)

        self.gat_band1_beta = GATConv(in_channels, hidden_channels, heads=heads, concat=True)
        self.bn_band1_beta = BatchNorm1d(hidden_channels * heads)

        self.gat_band1_theta = GATConv(in_channels, hidden_channels, heads=heads, concat=True)
        self.bn_band1_theta = BatchNorm1d(hidden_channels * heads)

        fused_in = hidden_channels * heads * 3
        self.gat_fused = GATConv(fused_in, hidden_channels, heads=1, concat=True)
        self.bn_fused = BatchNorm1d(hidden_channels)

        self.gat_final = GATConv(hidden_channels, hidden_channels, heads=1, concat=True)
        self.bn_final = BatchNorm1d(hidden_channels)



    def forward(self, data, return_attn=False):

        x_theta = data.x_theta
        x_alpha = data.x_alpha
        x_beta = data.x_beta

        # read edge indices
        edge_index_theta = data.edge_index_theta
        edge_index_alpha = data.edge_index_alpha
        edge_index_beta = data.edge_index_beta

        # optional edge_attr
        edge_attr_theta = getattr(data, "edge_attr_theta", None)
        edge_attr_alpha = getattr(data, "edge_attr_alpha", None)
        edge_attr_beta = getattr(data, "edge_attr_beta", None)

        # ---- per-band first GAT (with optional attention returns)
        if return_attn:
            out_alpha, (edge_index_a, attn_a) = self.gat_band1_alpha(x_alpha, edge_index_alpha, return_attention_weights=True)
            out_alpha = apply_bn_activation(out_alpha, self.bn_band1_alpha, activation=F.relu)

            out_beta, (edge_index_b, attn_b) = self.gat_band1_beta(x_beta, edge_index_beta, return_attention_weights=True)
            out_beta = apply_bn_activation(out_beta, self.bn_band1_beta, activation=F.relu)

            out_theta, (edge_index_t, attn_t) = self.gat_band1_theta(x_theta, edge_index_theta, return_attention_weights=True)
            out_theta = apply_bn_activation(out_theta, self.bn_band1_theta, activation=F.relu)
        else:
            out_alpha = self.gat_band1_alpha(x_alpha, edge_index_alpha)
            out_alpha = apply_bn_activation(out_alpha, self.bn_band1_alpha, activation=F.relu)

            out_beta = self.gat_band1_beta(x_beta, edge_index_beta)
            out_beta = apply_bn_activation(out_beta, self.bn_band1_beta, activation=F.relu)

            out_theta = self.gat_band1_theta(x_theta, edge_index_theta)
            out_theta = apply_bn_activation(out_theta, self.bn_band1_theta, activation=F.relu)

        out_concat = torch.cat([out_alpha, out_beta, out_theta], dim=-1)  # [N, hidden*heads*3]

        # fused conv (we use alpha edges for fused graph by default) # 这边因为是以alpha波为主，所以采用alpha波的边
        if return_attn:
            out_fused, (edge_index_fused, attn_fused) = self.gat_fused(out_concat, edge_index_alpha, return_attention_weights=True)
            out_fused = apply_bn_activation(out_fused, self.bn_fused, activation=F.relu)
            out_final, (edge_index_out, attn_out) = self.gat_final(out_fused, edge_index_alpha, return_attention_weights=True)
            out_final = apply_bn_activation(out_final, self.bn_final, activation=F.relu)

            # return final embedding plus structured attention info
            # we'll provide attention lists grouped per-band/layer so caller can aggregate
            attn_info = {
                "alpha_layer1": (edge_index_a, attn_a),
                "beta_layer1": (edge_index_b, attn_b),
                "theta_layer1": (edge_index_t, attn_t),
                "fused_layer": (edge_index_fused, attn_fused),
                "out_layer": (edge_index_out, attn_out)
            }
            return out_final, attn_info
        else:
            out_fused = self.gat_fused(out_concat, edge_index_alpha)
            out_fused = apply_bn_activation(out_fused, self.bn_fused, activation=F.relu)
            out_final = self.gat_final(out_fused, edge_index_alpha)
            out_final = apply_bn_activation(out_final, self.bn_final, activation=F.relu)
            return out_final  # z_fusion, shape [N, hidden]


class GraphClassifier(torch.nn.Module):
    def __init__(self, fused_dim, out_channels = 1,global_feature_channels=7):
        super().__init__()
        self.fused_dim = fused_dim
        self.out_channels = out_channels
        self.global_feature_channels = global_feature_channels

        self.mlp_classifier = Sequential(
            Linear(fused_dim + global_feature_channels, 64),
            ReLU(True),
            Linear(64, 16),
            ReLU(True),
            Linear(16, 8),
            ReLU(True),
            Linear(8, out_channels)
        )

    def forward(self, z_fusion,batch,global_feature):

        print('z_fusion:',z_fusion.std(dim=0).mean())

        x = global_mean_pool(z_fusion, batch)            # [num_graphs, fused_dim]

        x = torch.cat([x, global_feature], dim=1)        # [num_graphs, fused_dim + global_feature_channels]

        x = self.mlp_classifier(x)                       # [num_graphs, 1]

        return x


class FusionSeparateDecoder(torch.nn.Module):                           #reconstruct 3 sub-graphs using mlps
    def __init__(self, in_channels,hidden_channels):
        super().__init__()
        self.mlp_alpha = Sequential(
            Linear(in_channels, hidden_channels),
            ReLU(),
            Linear(hidden_channels, hidden_channels),
            ReLU(),
        )
        self.mlp_beta = Sequential(
            Linear(in_channels, hidden_channels),
            ReLU(),
            Linear(hidden_channels, hidden_channels),
            ReLU(),
        )
        self.mlp_theta = Sequential(
            Linear(in_channels, hidden_channels),
            ReLU(),
            Linear(hidden_channels, hidden_channels),
            ReLU(),
        )

    def forward(self, z_fusion,batch):
        device = z_fusion.device
        num_graphs = batch.max().item() + 1
        reconstructed_adj_alpha = []
        reconstructed_adj_beta = []
        reconstructed_adj_theta = []

        reconstructed_feature_alpha = []
        reconstructed_feature_beta = []
        reconstructed_feature_theta = []

        for i in range(num_graphs):
            node_mask = (batch == i)
            z_i = z_fusion[node_mask]  # [68, hidden]
            reconstructed_alpha_i = self.mlp_alpha(z_i)
            reconstructed_beta_i = self.mlp_beta(z_i)
            reconstructed_theta_i = self.mlp_theta(z_i)

            # 得到 [68, 68] 的邻接重建矩阵
            logits_alpha_i = F.sigmoid(reconstructed_alpha_i @ reconstructed_alpha_i.t())
            logits_beta_i = F.sigmoid(reconstructed_beta_i @ reconstructed_beta_i.t())
            logits_theta_i = F.sigmoid(reconstructed_theta_i @ reconstructed_theta_i.t())

            reconstructed_adj_alpha.append(logits_alpha_i)
            reconstructed_adj_beta.append(logits_beta_i)
            reconstructed_adj_theta.append(logits_theta_i)

            reconstructed_feature_alpha.append(reconstructed_alpha_i)
            reconstructed_feature_beta.append(reconstructed_beta_i)
            reconstructed_feature_theta.append(reconstructed_theta_i)

        reconstructed_alpha = torch.stack(reconstructed_adj_alpha).to(device)
        reconstructed_beta = torch.stack(reconstructed_adj_beta).to(device)
        reconstructed_theta = torch.stack(reconstructed_adj_theta).to(device)

        return reconstructed_alpha, reconstructed_beta, reconstructed_theta, (reconstructed_feature_alpha, reconstructed_feature_beta, reconstructed_feature_theta)



class MultiBandGATWithNodeImportance(torch.nn.Module):
    def __init__(self,in_channels,hidden_channels, out_channels,global_feature_channels=7, heads=1, recon_weight=1.0):
        super().__init__()
        self.encoder = GATEncoder(in_channels, hidden_channels, heads=1)                  #encoder生成中间表征Z，不经过pooling
        self.decoder = FusionSeparateDecoder(hidden_channels,hidden_channels)           #decoder从Z中还原三个子图
        self.classifier = GraphClassifier(hidden_channels, out_channels, global_feature_channels)              #classifier对Z进行pooling，后生成logits
        self.recon_weight = recon_weight

    def forward(self, data,return_attn = False):
        batch = data.batch
        global_feature = data.global_feature
        if return_attn:
            z_fusion, attn_info = self.encoder(data, return_attn=True)
        else:
            z_fusion = self.encoder(data, return_attn=False)

        print(z_fusion.shape)

        if self.recon_weight > 0:
            reconstructed_alpha_adj_matrix, reconstructed_beta_adj_matrix, reconstructed_theta_adj_matrix, per_band_z = self.decoder(z_fusion, batch)    #return 3 reconstructed adj matrix
        else:
            reconstructed_alpha_adj_matrix = reconstructed_beta_adj_matrix = reconstructed_theta_adj_matrix = per_band_z = None

        classify_logits = self.classifier(z_fusion,batch,global_feature)

        if return_attn:
            return ( reconstructed_alpha_adj_matrix, reconstructed_beta_adj_matrix, reconstructed_theta_adj_matrix, classify_logits,
                per_band_z, attn_info, z_fusion)
        else:
            return reconstructed_alpha_adj_matrix, reconstructed_beta_adj_matrix, reconstructed_theta_adj_matrix, classify_logits



class MultiTaskLossModule(torch.nn.Module):
    def __init__(self, recon_weight=1.0, cls_weight=1.0, pos_weight_edge=None):
        super().__init__()
        # BCEWithLogitsLoss is numerically stable and appropriate for 0/1 adjacency targets
        if pos_weight_edge is not None:
            self.recon_criterion = BCEWithLogitsLoss(pos_weight=pos_weight_edge)
        else:
            self.recon_criterion = BCEWithLogitsLoss()
        self.cls_criterion = BCEWithLogitsLoss()
        self.recon_weight = recon_weight
        self.cls_weight = cls_weight

    def forward(self, logits_theta, logits_alpha, logits_beta,
                adj_target_theta, adj_target_alpha, adj_target_beta,
                cls_logits, cls_target):

        if self.recon_weight == 0:
            class_loss = self.cls_criterion(cls_logits, cls_target)
            return class_loss, {"recon_loss": 0.0, "class_loss": class_loss.item()}

        loss_t = self.recon_criterion(logits_theta.flatten(), adj_target_theta.flatten())
        loss_a = self.recon_criterion(logits_alpha.flatten(), adj_target_alpha.flatten())
        loss_b = self.recon_criterion(logits_beta.flatten(), adj_target_beta.flatten())
        recon_loss = (loss_t + loss_a + loss_b) / 3.0

        # print(cls_logits)
        # print(cls_target)

        class_loss = self.cls_criterion(cls_logits, cls_target)

        total_loss = self.recon_weight * recon_loss + self.cls_weight * class_loss
        return total_loss, {"recon_loss": recon_loss.item(), "class_loss": class_loss.item()}



class MultiTaskLossModuleForFullBand(torch.nn.Module):
    def __init__(self, recon_weight=1.0, cls_weight=1.0, pos_weight_edge=None):
        super().__init__()
        # BCEWithLogitsLoss is numerically stable and appropriate for 0/1 adjacency targets
        if pos_weight_edge is not None:
            self.recon_criterion = BCEWithLogitsLoss(pos_weight=pos_weight_edge)
        else:
            self.recon_criterion = BCEWithLogitsLoss()
        self.cls_criterion = BCEWithLogitsLoss()
        self.recon_weight = recon_weight
        self.cls_weight = cls_weight

    def forward(self, logits, adj_target,cls_logits, cls_target):

        if self.recon_weight == 0:
            class_loss = self.cls_criterion(cls_logits, cls_target)
            return class_loss, {"recon_loss": 0.0, "class_loss": class_loss.item()}

        recon_loss = self.recon_criterion(logits.flatten(), adj_target.flatten())

        # print(cls_logits)
        # print(cls_target)

        class_loss = self.cls_criterion(cls_logits, cls_target)

        total_loss = self.recon_weight * recon_loss + self.cls_weight * class_loss
        return total_loss, {"recon_loss": recon_loss.item(), "class_loss": class_loss.item()}


class CrossAttentionBlock(torch.nn.Module):
    """
    执行 Q (Query) 对 K (Key) 和 V (Value) 的交叉注意力。
    这里假设 Q, K, V 都是形状为 (batch_size, feature_dim) 的向量。
    """

    def __init__(self, feature_dim):
        super().__init__()
        self.feature_dim = feature_dim

        # 线性层用于将输入向量投影到 Q, K, V 空间
        # 由于输入是向量 (B, D)，我们不需要多头注意力，使用单头实现简化
        self.W_q = Linear(feature_dim, feature_dim, bias=False)
        self.W_k = Linear(feature_dim, feature_dim, bias=False)
        self.W_v = Linear(feature_dim, feature_dim, bias=False)

        # 尺度因子，用于稳定梯度
        self.scale = 1.0 / math.sqrt(feature_dim)

    def forward(self, Q, K, V):
        """
        Q: Query 向量 (例如 V1)
        K: Key 向量 (例如 V2)
        V: Value 向量 (例如 V2)
        """
        # 1. 线性投影
        q = self.W_q(Q).unsqueeze(1)  # (B, 1, D)
        k = self.W_k(K).unsqueeze(1)  # (B, 1, D)
        v = self.W_v(V).unsqueeze(1)  # (B, 1, D)

        # 2. 计算注意力得分
        # (B, 1, D) @ (B, D, 1) -> (B, 1, 1)
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # 3. Softmax 得到注意力权重
        attention_weights = F.softmax(attention_scores, dim=-1)  # (B, 1, 1)

        # 4. 加权求和
        # (B, 1, 1) @ (B, 1, D) -> (B, 1, D)
        # 然后去除维度为 1 的维度 -> (B, D)
        attended_output = torch.matmul(attention_weights, v).squeeze(1)

        return attended_output  # (B, D)



class SharedFullBandEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, heads=1):
        super().__init__()
        self.gat = torch.nn.ModuleList([
            GATConv(in_channels, hidden_channels * 4, heads=heads,dropout=0.2),
            GATConv(hidden_channels * 4, hidden_channels * 2, heads=heads,dropout=0.1),
            GATConv(hidden_channels * 2, hidden_channels, heads=heads)
        ])
        self.bn = torch.nn.ModuleList([
            BatchNorm1d(hidden_channels * 4), BatchNorm1d(hidden_channels * 2), BatchNorm1d(hidden_channels)
        ])

    def forward_band(self, x, edge_index, gats, bns, band_name, return_attn):
        attn_list = {}
        for i in range(len(gats)):
            if return_attn:
                x, (edge, attn) = gats[i](x, edge_index, return_attention_weights=True)
                attn_list[f"{band_name}_layer{i + 1}"] = (edge, attn)
            else:
                x = gats[i](x, edge_index)
            x = apply_bn_activation(x, bns[i], activation=F.relu)
        return x, attn_list

    def forward(self, data, return_attn=False):
        z, attn = self.forward_band(data.x, data.edge_index, self.gat, self.bn, "full_band",
                                         return_attn)

        all_attn = {**attn} if return_attn else None
        return z, all_attn





class NewFullBandGATClassifier(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, global_feature_channels=7, heads=1):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.global_feature_channels = global_feature_channels
        self.heads = heads

        self.attn1 = CrossAttentionBlock(hidden_channels)                               #因为QKV都是同一个序列，没有必要使用attn2进行重复的注意力系数计算
        self.mlp = Sequential(Linear(hidden_channels*2, hidden_channels),
                                    ReLU())

        self.classifier = Sequential(
            Linear(hidden_channels + global_feature_channels, 64),
            ReLU(),
            # Dropout(0.2),
            Linear(64, 16),
            ReLU(),
            # Dropout(0.2),
            Linear(16, 8),
            ReLU(),
            # Dropout(0.1),
            Linear(8, out_channels)
        )

    def forward(self, embeddings, data, return_attn=False):
        cls_attn = {}
        z = embeddings          #直接使用经过SharedFullBandEncoder编码后的中间向量z

        out = global_mean_pool(z, data.batch)

        out_enhanced = self.attn1(Q=out,K=out,V=out)

        out_fused = self.mlp(torch.cat((out,out_enhanced),dim=-1))

        if hasattr(data, 'global_feature') and data.global_feature is not None:
            out_fused = torch.cat([out_fused, data.global_feature], dim=1)

        out  = self.classifier(out_fused)

        return (out, cls_attn) if return_attn else out



class NewFullBandGAE(torch.nn.Module):
    def __init__(self, hidden_channels, heads=1):  # 在这里先不加入多头机制，heads等于1
        super().__init__()
        self.heads = 1

        self.mlp = Sequential(
            Dropout(0.3),
            Linear(hidden_channels*heads, hidden_channels),
            ReLU(),
            Dropout(0.2),
            Linear(hidden_channels, hidden_channels),
            ReLU(),
            Dropout(0.2),
        )

    def forward(self, embeddings, data, return_attn=False):
        recon_attn = {}

        # 获取 Batch 信息，避免循环
        z = embeddings  # 直接使用经过SharedFullBandEncoder编码后的中间向量z
        num_graphs = data.num_graphs
        num_nodes = z.size(0) // num_graphs

        # 批量变换形状为 (Batch, Nodes, Hidden)
        z_batch = z.view(num_graphs, num_nodes, -1)

        recon_z = self.mlp(z_batch)

        # 使用 torch.bmm 一次性计算所有图的邻接矩阵重构
        # (B, N, D) @ (B, D, N) -> (B, N, N)
        logits_z = torch.bmm(recon_z, recon_z.transpose(1, 2))

        reconstructed_torch_adj = torch.sigmoid(logits_z)

        if return_attn:
            return reconstructed_torch_adj, recon_z, recon_attn
        else:
            return reconstructed_torch_adj, recon_z



class NewFullBandGATWithNodeImportance(torch.nn.Module):
    def __init__(self,in_channels,hidden_channels, out_channels,global_feature_channels=7, heads=1):
        super().__init__()
        self.shared_encoder = SharedFullBandEncoder(in_channels, hidden_channels, heads)                                   #shared_encoder为两个任务共用的encoder，功能为使用gat生成全波段节点表示
        self.fullband_classifier = NewFullBandGATClassifier(hidden_channels, out_channels, global_feature_channels)                  #classifier使用shared_encoder生成的节点表示，经过pooling生成图表示然后通过selfattention得到综合表示后进行融合并分类
        self.fullband_GAE = NewFullBandGAE(hidden_channels)            #GAE使用shared_encoder生成的节点表示，这个GAE中仅包含GAE中的decoder

    def forward(self, data,return_attn = False):
        # batch = data.batch
        # global_feature = data.global_feature

        embeddings, attn_info = self.shared_encoder(data, return_attn=return_attn)

        if return_attn:
            cls_logits, cls_attn_info = self.fullband_classifier(embeddings, data, True)
            reconstructed_adj_matrix, full_band_z, recon_attn_info = self.fullband_GAE(embeddings, data, True)
            # Unpack recon_results and return in your original format
            return (reconstructed_adj_matrix, cls_logits,full_band_z, attn_info, cls_attn_info, recon_attn_info)
        else:
            cls_logits = self.fullband_classifier(embeddings, data, False)
            reconstructed_adj_matrix, full_band_z = self.fullband_GAE(embeddings, data, False)
            return reconstructed_adj_matrix, cls_logits








class SharedMultiBandEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, heads=1):
        super().__init__()
        self.gat_alpha = torch.nn.ModuleList([
            GATConv(in_channels, hidden_channels * 4, heads=heads,dropout=0.2),
            GATConv(hidden_channels * 4, hidden_channels * 2, heads=heads,dropout=0.1),
            GATConv(hidden_channels * 2, hidden_channels, heads=heads)
        ])
        self.bn_alpha = torch.nn.ModuleList([
            BatchNorm1d(hidden_channels * 4), BatchNorm1d(hidden_channels * 2), BatchNorm1d(hidden_channels)
        ])

        # self.gat_alpha = torch.nn.ModuleList([
        #     GATConv(in_channels, hidden_channels * 4, heads=heads),
        #     GATConv(hidden_channels * 4, hidden_channels * 2, heads=heads),
        #     GATConv(hidden_channels * 2, hidden_channels, heads=heads)
        # ])
        # self.gat_alpha = torch.nn.ModuleList([
        #     GATConv(in_channels, hidden_channels, heads=heads)
        # ])
        # self.bn_alpha = torch.nn.ModuleList([
        #      BatchNorm1d(hidden_channels)
        # ])

        self.gat_beta = torch.nn.ModuleList([
            GATConv(in_channels, hidden_channels * 4, heads=heads,dropout=0.2),
            GATConv(hidden_channels * 4, hidden_channels * 2, heads=heads,dropout=0.1),
            GATConv(hidden_channels * 2, hidden_channels, heads=heads)
        ])
        self.bn_beta = torch.nn.ModuleList([
            BatchNorm1d(hidden_channels * 4), BatchNorm1d(hidden_channels * 2), BatchNorm1d(hidden_channels)
        ])

        # self.gat_beta = torch.nn.ModuleList([
        #     GATConv(in_channels, hidden_channels * 4, heads=heads),
        #     GATConv(hidden_channels * 4, hidden_channels * 2, heads=heads),
        #     GATConv(hidden_channels * 2, hidden_channels, heads=heads)
        # ])
        # self.gat_beta = torch.nn.ModuleList([
        #     GATConv(in_channels, hidden_channels, heads=heads)
        # ])
        # self.bn_beta = torch.nn.ModuleList([
        #      BatchNorm1d(hidden_channels)
        # ])

        self.gat_theta = torch.nn.ModuleList([
            GATConv(in_channels, hidden_channels * 4, heads=heads,dropout=0.2),
            GATConv(hidden_channels * 4, hidden_channels * 2, heads=heads,dropout=0.1),
            GATConv(hidden_channels * 2, hidden_channels, heads=heads)
        ])
        self.bn_theta = torch.nn.ModuleList([
            BatchNorm1d(hidden_channels * 4), BatchNorm1d(hidden_channels * 2), BatchNorm1d(hidden_channels)
        ])

        # self.gat_theta = torch.nn.ModuleList([
        #     GATConv(in_channels, hidden_channels * 4, heads=heads),
        #     GATConv(hidden_channels * 4, hidden_channels * 2, heads=heads),
        #     GATConv(hidden_channels * 2, hidden_channels, heads=heads)
        # ])
        # self.gat_theta =  torch.nn.ModuleList([
        #     GATConv(in_channels, hidden_channels, heads=heads)
        # ])
        # self.bn_theta = torch.nn.ModuleList([
        #      BatchNorm1d(hidden_channels)
        # ])

    def forward_band(self, x, edge_index, gats, bns, band_name, return_attn):
        node_importance = None

        for i in range(len(gats)):
            # 1. Capture attention weights for the current layer
            if return_attn:
                # We must specify return_attention_weights=True for EACH specific GAT layer
                x, (edge, attn) = gats[i](x, edge_index, return_attention_weights=True)

                # 2. Extract importance from the FINAL GAT layer of this specific band
                if i == len(gats) - 1:
                    # Average across heads: [num_edges, heads] -> [num_edges]
                    alpha_mean = attn.mean(dim=1)

                    # 2. FILTER SELF-LOOPS
                    # Identify edges where source != target
                    mask = edge[0] != edge[1]

                    # Apply mask to edges and weights
                    neighbor_edges = edge[1][mask]
                    neighbor_weights = alpha_mean[mask]

                    # 3. AGGREGATE ONLY NEIGHBOR ATTENTION
                    num_nodes = x.size(0)
                    node_importance = torch.zeros(num_nodes, device=x.device)

                    # Summing attention received ONLY from other channels
                    node_importance.scatter_add_(0, neighbor_edges, neighbor_weights)
                    print(node_importance)
            else:
                x = gats[i](x, edge_index)

            # Apply normalization and activation
            x = F.relu(bns[i](x))

        return x, node_importance

    def forward(self, data, return_attn=False):
        z_a, attn_a = self.forward_band(data.x_alpha, data.edge_index_alpha, self.gat_alpha, self.bn_alpha, "alpha",
                                         return_attn)
        z_b, attn_b = self.forward_band(data.x_beta, data.edge_index_beta, self.gat_beta, self.bn_beta, "beta", return_attn)
        z_t, attn_t = self.forward_band(data.x_theta, data.edge_index_theta, self.gat_theta, self.bn_theta, "theta",
                                         return_attn)

        # Return the pooled representations and the node importance for each band
        all_attn = {"alpha": attn_a, "beta": attn_b, "theta": attn_t} if return_attn else None
        return (z_a, z_b, z_t), all_attn




class NewMultibandGATClassifier(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, global_feature_channels=7, heads=1):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.global_feature_channels = global_feature_channels
        self.heads = heads

        self.attn_alpha1 = CrossAttentionBlock(hidden_channels)
        self.attn_alpha2 = CrossAttentionBlock(hidden_channels)
        self.mlp_alpha = Sequential(Linear(hidden_channels*3, hidden_channels),
                                    ReLU())

        self.attn_beta1 = CrossAttentionBlock(hidden_channels)
        self.attn_beta2 = CrossAttentionBlock(hidden_channels)
        self.mlp_beta = Sequential(Linear(hidden_channels*3, hidden_channels),
                                    ReLU())

        self.attn_theta1 = CrossAttentionBlock(hidden_channels)
        self.attn_theta2 = CrossAttentionBlock(hidden_channels)
        self.mlp_theta = Sequential(Linear(hidden_channels*3, hidden_channels),
                                    ReLU())

        self.band_weights = Parameter(torch.ones(3))

        self.classifier = Sequential(
            Linear(hidden_channels + global_feature_channels, 64),
            ReLU(),
            # Dropout(0.2),
            Linear(64, 16),
            ReLU(),
            # Dropout(0.2),
            Linear(16, 8),
            ReLU(),
            # Dropout(0.1),
            Linear(8, out_channels)
        )

    def forward(self, embeddings, data, return_attn=False):
        cls_attn = {}
        z_a, z_b, z_t = embeddings          #直接使用经过SharedMultiBandEncoder编码后的中间向量z_a,z_b,z_t

        out_alpha = global_mean_pool(z_a, data.batch)
        out_beta = global_mean_pool(z_b, data.batch)
        out_theta = global_mean_pool(z_t, data.batch)

        # out = self.classifier((out_alpha + out_beta + out_theta) / 3)

        out_alpha_enhanced1 = self.attn_alpha1(Q=out_alpha,K=out_beta,V=out_beta)
        out_alpha_enhanced2 = self.attn_alpha2(Q=out_alpha,K=out_theta,V=out_theta)

        out_beta_enhanced1 = self.attn_beta1(Q=out_beta,K=out_alpha,V=out_alpha)
        out_beta_enhanced2 = self.attn_beta2(Q=out_beta,K=out_theta,V=out_theta)

        out_theta_enhanced1 = self.attn_theta1(Q=out_theta,K=out_alpha,V=out_alpha)
        out_theta_enhanced2 = self.attn_theta2(Q=out_theta,K=out_beta,V=out_beta)

        out_alpha_fused = self.mlp_alpha(torch.cat((out_alpha,out_alpha_enhanced1,out_alpha_enhanced2),dim=-1))
        out_beta_fused = self.mlp_beta(torch.cat((out_beta,out_beta_enhanced1,out_beta_enhanced2),dim=-1))
        out_theta_fused = self.mlp_theta(torch.cat((out_theta,out_theta_enhanced1,out_theta_enhanced2),dim=-1))

        weights = F.softmax(self.band_weights,dim=0)

        out_fused = weights[0]*out_alpha_fused + weights[1]*out_beta_fused + weights[2]*out_theta_fused

        if hasattr(data, 'global_feature') and data.global_feature is not None:
            out_fused = torch.cat([out_fused, data.global_feature], dim=1)

        out  = self.classifier(out_fused)

        # return (out, cls_attn) if return_attn else out
        return (out, cls_attn) if return_attn else (out, out_fused,weights)


class NewMultibandGAE(torch.nn.Module):
    def __init__(self, hidden_channels, heads=1):  # 在这里先不加入多头机制，heads等于1
        super().__init__()
        self.heads = 1

        self.mlp_alpha = Sequential(
            Dropout(0.3),
            Linear(hidden_channels*heads, hidden_channels),
            ReLU(),
            Dropout(0.2),
            Linear(hidden_channels, hidden_channels),
            ReLU(),
            Dropout(0.2),
        )
        self.mlp_beta = Sequential(
            Dropout(0.3),
            Linear(hidden_channels * heads, hidden_channels),
            ReLU(),
            Dropout(0.2),
            Linear(hidden_channels, hidden_channels),
            ReLU(),
            Dropout(0.2),
        )
        self.mlp_theta = Sequential(
            Dropout(0.3),
            Linear(hidden_channels * heads, hidden_channels),
            ReLU(),
            Dropout(0.2),
            Linear(hidden_channels, hidden_channels),
            ReLU(),
            Dropout(0.2),
        )

    def forward(self, embeddings, data, return_attn=False):
        recon_attn = {}

        #获取 Batch 信息，避免循环
        z_a, z_b, z_t = embeddings  # 直接使用经过SharedMultiBandEncoder编码后的中间向量z_a,z_b,z_t
        num_graphs = data.num_graphs
        num_nodes = z_a.size(0) // num_graphs

        #批量变换形状为 (Batch, Nodes, Hidden)
        z_a_batch = z_a.view(num_graphs, num_nodes, -1)
        z_b_batch = z_b.view(num_graphs, num_nodes, -1)
        z_t_batch = z_t.view(num_graphs, num_nodes, -1)

        #批量通过 MLP (Linear 层支持 Batch 维度)
        recon_a = self.mlp_alpha(z_a_batch)
        recon_b = self.mlp_beta(z_b_batch)
        recon_t = self.mlp_theta(z_t_batch)

        #使用 torch.bmm 一次性计算所有图的邻接矩阵重构
        # (B, N, D) @ (B, D, N) -> (B, N, N)
        logits_a = torch.bmm(recon_a, recon_a.transpose(1, 2))
        logits_b = torch.bmm(recon_b, recon_b.transpose(1, 2))
        logits_t = torch.bmm(recon_t, recon_t.transpose(1, 2))

        reconstructed_alpha = torch.sigmoid(logits_a)
        reconstructed_beta = torch.sigmoid(logits_b)
        reconstructed_theta = torch.sigmoid(logits_t)

        if return_attn:
            return reconstructed_alpha, reconstructed_beta, reconstructed_theta, (recon_a, recon_b, recon_t), recon_attn
        else:
            return reconstructed_alpha, reconstructed_beta, reconstructed_theta, (recon_a, recon_b, recon_t)

class NewMultiBandGATWithNodeImportance(torch.nn.Module):
    def __init__(self,in_channels,hidden_channels, out_channels,global_feature_channels=7, heads=1):
        super().__init__()
        self.shared_encoder = SharedMultiBandEncoder(in_channels, hidden_channels, heads)                                   #shared_encoder为两个任务共用的encoder，功能为使用分波段gat生成分波段节点表示
        self.multiband_classifier = NewMultibandGATClassifier(hidden_channels, out_channels, global_feature_channels)                  #classifier使用shared_encoder生成的节点表示，经过pooling生成图表示然后通过两两crossattention得到综合表示后进行融合并分类
        self.multiband_GAE = NewMultibandGAE(hidden_channels)            #GAE使用shared_encoder生成的节点表示，这个GAE中仅包含GAE中的decoder，重建损失为三个子图的重建损失相加

    def forward(self, data,return_attn = False):
        # batch = data.batch
        # global_feature = data.global_feature

        embeddings, attn_info = self.shared_encoder(data, return_attn=return_attn)

        # if return_attn:
        #     cls_logits, cls_attn_info = self.multiband_classifier(embeddings, data, True)
        #     reconstructed_alpha_adj_matrix, reconstructed_beta_adj_matrix, reconstructed_theta_adj_matrix, per_band_z, recon_attn_info = self.multiband_GAE(embeddings, data, True)
        #     # Unpack recon_results and return in your original format
        #     return (reconstructed_alpha_adj_matrix, reconstructed_beta_adj_matrix, reconstructed_theta_adj_matrix,
        #             cls_logits,per_band_z, attn_info, cls_attn_info, recon_attn_info)
        if return_attn:
            cls_logits, graph_embeddings, weights = self.multiband_classifier(embeddings, data, False)
            reconstructed_alpha_adj_matrix, reconstructed_beta_adj_matrix, reconstructed_theta_adj_matrix, per_band_z = self.multiband_GAE(
                embeddings, data, False)
            return reconstructed_alpha_adj_matrix, reconstructed_beta_adj_matrix, reconstructed_theta_adj_matrix, cls_logits, graph_embeddings, weights,attn_info
        else:
            cls_logits , graph_embeddings,weights = self.multiband_classifier(embeddings, data, False)
            reconstructed_alpha_adj_matrix, reconstructed_beta_adj_matrix, reconstructed_theta_adj_matrix, per_band_z = self.multiband_GAE(embeddings, data, False)
            return reconstructed_alpha_adj_matrix, reconstructed_beta_adj_matrix, reconstructed_theta_adj_matrix, cls_logits,graph_embeddings, weights
        # if return_attn:
        #     cls_logits, cls_attn_info = self.multiband_classifier(data, return_attn=True)
        #     # print(cls_logits.shape)
        #     reconstructed_alpha_adj_matrix, reconstructed_beta_adj_matrix, reconstructed_theta_adj_matrix, per_band_z, recon_attn_info = self.multiband_GAE(data, return_attn=True)  # return 3 reconstructed adj matrix
        #     return (reconstructed_alpha_adj_matrix, reconstructed_beta_adj_matrix, reconstructed_theta_adj_matrix, cls_logits,
        #             per_band_z, cls_attn_info, recon_attn_info)
        # else:
        #     cls_logits = self.multiband_classifier(data, return_attn=False)
        #     # print(cls_logits.shape)
        #     reconstructed_alpha_adj_matrix, reconstructed_beta_adj_matrix, reconstructed_theta_adj_matrix, per_band_z = self.multiband_GAE(data, return_attn=False)
        #     return reconstructed_alpha_adj_matrix, reconstructed_beta_adj_matrix, reconstructed_theta_adj_matrix, cls_logits
        #     # return cls_logits


from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv, global_mean_pool

# 通用基准模型包装器
class GNNBaseline(torch.nn.Module):
    def __init__(self, model_type, in_channels, hidden_channels, out_channels):
        super(GNNBaseline, self).__init__()
        self.model_type = model_type

        if model_type == 'GCN':
            self.conv = GCNConv(in_channels, hidden_channels)
        elif model_type == 'GAT':
            self.conv = GATConv(in_channels, hidden_channels, heads=1)
        elif model_type == 'GraphSAGE':
            self.conv = SAGEConv(in_channels, hidden_channels)
        elif model_type == 'GIN':
            # GIN 需要一个内部 MLP
            nn = Sequential(Linear(in_channels, hidden_channels), ReLU(), Linear(hidden_channels, hidden_channels))
            self.conv = GINConv(nn)

        self.classifier = Sequential(
            Linear(hidden_channels, 16),
            ReLU(),
            Linear(16, out_channels)
        )

    def forward(self, data):
        # 基准模型统一使用全波段数据 data.x
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv(x, edge_index).relu()
        x = global_mean_pool(x, batch)
        return self.classifier(x)