import logging
import numpy as np
import torch
from collections import defaultdict

from utils.utils import MergeLayer, PredictionLayer_DTDG,PredictionLayer
from modules.memory import Memory
from modules.message_aggregator import get_message_aggregator
from modules.message_function import get_message_function
from modules.memory_updater import get_memory_updater
from modules.embedding_module import get_embedding_module
from model.time_encoding import TimeEncode
from torch import nn
import math

class MyEmbeddingModule(nn.Module):
    def __init__(self, n_node_features, dropout):
        super(MyEmbeddingModule, self).__init__()
        self.n_node_features = n_node_features
        self.dropout = dropout

    def compute_embedding(self, memory, source_nodes, timestamps, time_diffs=None):
        pass

class TimeEmbedding(MyEmbeddingModule):
    def __init__(self, n_node_features, dropout=0.1):
        super(TimeEmbedding, self).__init__(n_node_features, dropout)

        class NormalLinear(nn.Linear):
            # From Jodie code
            def reset_parameters(self):
                stdv = 1. / math.sqrt(self.weight.size(1))
                self.weight.data.normal_(0, stdv)
                if self.bias is not None:
                    self.bias.data.normal_(0, stdv)

        self.embedding_layer = NormalLinear(1, self.n_node_features)

    def compute_embedding(self, memory, source_nodes, timestamps, time_diffs=None):
        source_embeddings = memory[source_nodes, :] * (1 + self.embedding_layer(time_diffs.unsqueeze(1)))

        return source_embeddings


class CDOD(torch.nn.Module):
    def __init__(self, neighbor_finder, node_features,hidden_layer,hidden_dim,concat_mode,model_mode,divide_base,DTDG_DIV,
                 device, n_layers=2,
                 n_heads=2, dropout=0.1, use_memory=False,
                 memory_update_at_start=True, message_dimension=100,
                 memory_dimension=500, embedding_module_type="graph_attention",
                 message_function="mlp",
                 mean_time_shift_src=0, std_time_shift_src=1, mean_time_shift_dst=0,
                 std_time_shift_dst=1, n_neighbors=None, aggregator_type="last",
                 memory_updater_type="gru",
                 use_destination_embedding_in_message=False,
                 use_source_embedding_in_message=False,
                 dyrep=False,
                 output_len=30):
        super(CDOD, self).__init__()
        self.output_len = output_len
        self.n_layers = n_layers
        self.neighbor_finder = neighbor_finder
        self.device = device
        self.logger = logging.getLogger(__name__)
        self.node_raw_features = torch.from_numpy(node_features.astype(np.float32)).to(device)
        self.n_node_features = self.node_raw_features.shape[1]
        self.n_nodes = self.node_raw_features.shape[0]
        self.n_edge_features = 1
        # self.embedding_dimension = memory_dimension
        self.embedding_dimension = self.n_node_features
        self.n_neighbors = n_neighbors
        self.use_memory = use_memory
        self.n_time_features = 8
        self.time_encoder = TimeEncode(dimension=self.n_time_features)
        self.memory = None
        self.DTDG_DIV = DTDG_DIV
        self.model_mode = model_mode
        self.mean_time_shift_src = mean_time_shift_src
        self.std_time_shift_src = std_time_shift_src
        self.mean_time_shift_dst = mean_time_shift_dst
        self.std_time_shift_dst = std_time_shift_dst

        if self.use_memory:
            self.memory_dimension = memory_dimension
            raw_message_dimension = self.memory_dimension + self.n_edge_features + self.n_node_features
            message_dimension = raw_message_dimension
            self.memory = Memory(n_nodes=self.n_nodes,
                                 memory_dimension=self.memory_dimension,
                                 input_dimension=message_dimension,
                                 message_dimension=message_dimension,
                                 device=device)
            self.message_aggregator = get_message_aggregator(aggregator_type=aggregator_type,
                                                             embedding_dimension=self.memory_dimension,
                                                             device=device)
            self.message_function = get_message_function(module_type="identity",
                                                         raw_message_dimension=raw_message_dimension,
                                                         message_dimension=message_dimension)
            self.memory_updater = get_memory_updater(module_type="CT+DT",
                                                     memory=self.memory,
                                                     message_dimension=message_dimension,
                                                     memory_dimension=self.memory_dimension,
                                                     device=device, n_nodes=self.n_nodes, hidden_dim=hidden_dim)

            self.DTDG_memory_updater = get_memory_updater(module_type="DTDG",
                                                     memory=self.memory,
                                                     message_dimension=message_dimension,
                                                     memory_dimension=self.memory_dimension,
                                                     device=device, n_nodes=self.n_nodes, hidden_dim=hidden_dim)

            self.CTDG_memory_updater = get_memory_updater(module_type="CTDG",
                                                         memory=self.memory,
                                                         message_dimension=message_dimension,
                                                         memory_dimension=self.memory_dimension,
                                                         device=device, n_nodes=self.n_nodes, hidden_dim=hidden_dim)

        self.embedding_module = get_embedding_module(module_type="graph_attention",
                                                     node_features=self.node_raw_features,
                                                     # edge_features=self.edge_raw_features,
                                                     memory=self.memory,
                                                     neighbor_finder=self.neighbor_finder,
                                                     time_encoder=self.time_encoder,
                                                     n_layers=self.n_layers,
                                                     n_node_features=self.n_node_features,
                                                     n_edge_features=self.n_edge_features,
                                                     n_time_features=self.n_time_features,
                                                     embedding_dimension=self.embedding_dimension,
                                                     device=self.device,
                                                     n_heads=n_heads, dropout=dropout,
                                                     use_memory=use_memory,
                                                     n_neighbors=self.n_neighbors)
        self.hidden_layer = hidden_layer
        self.hidden_dim = hidden_dim
        self.fc = torch.nn.Linear(self.n_nodes, self.hidden_dim)
        self.act = torch.nn.ReLU()
        torch.nn.init.xavier_normal_(self.fc.weight)
        self.divide_base = divide_base
        self.h0 = nn.Parameter(torch.FloatTensor(self.hidden_layer, self.n_nodes, self.hidden_dim)).to(self.device)
        self.c0 = nn.Parameter(torch.FloatTensor(self.hidden_layer, self.n_nodes, self.hidden_dim)).to(self.device)
        self.lstm = nn.GRU(self.n_nodes, self.hidden_dim, self.hidden_layer).to(device)
        if model_mode==3:
            self.predict_od = PredictionLayer_DTDG(self.embedding_dimension, self.embedding_dimension, self.hidden_dim,
                                                   self.embedding_dimension, concat_mode, self.n_nodes)
        else:
            self.predict_od = PredictionLayer(self.embedding_dimension, self.embedding_dimension, self.n_nodes)

    def set_time_static(self, mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst):
        self.mean_time_shift_src = mean_time_shift_src
        self.std_time_shift_src = std_time_shift_src
        self.mean_time_shift_dst = mean_time_shift_dst
        self.std_time_shift_dst = std_time_shift_dst

    def compute_od_matrix(self, source_nodes, destination_nodes, edge_times, edge_features,input,
                          predicted_nodes, predicted_time, n_neighbors=20):

        edge_features = edge_features.unsqueeze(1)
        divide = max(len(source_nodes)//self.divide_base, 1)
        unique_sources_list = []
        source_id_to_messages_list = []
        for k in range(divide):
            begin = k*self.divide_base
            end = min((k+1)*self.divide_base,len(source_nodes))
            unique_sources, source_id_to_messages = self.get_raw_messages(source_nodes[begin:end],
                                                                          destination_nodes[begin:end],
                                                                          edge_times[begin:end], edge_features[begin:end],
                                                                          predicted_time=predicted_time)
            unique_sources_list.append(unique_sources)
            source_id_to_messages_list.append(source_id_to_messages)

        input = torch.unsqueeze(input,0)
        output, hn = self.lstm(input, self.h0)
        self.h0 = nn.Parameter(hn)
        DTDG_output = output[-1]  # output[window, batch, hidden]
        if self.model_mode==0:
            updated_memory, updated_last_update = self.get_updated_memory_new(unique_sources_list, source_id_to_messages_list, DTDG_output)
        elif self.model_mode==1:
            updated_memory, updated_last_update = self.get_updated_memory_DTDG(unique_sources_list, source_id_to_messages_list, DTDG_output)
        elif self.model_mode==2 or self.model_mode==3:
            updated_memory, updated_last_update = self.get_updated_memory_CTDG(unique_sources_list, source_id_to_messages_list)

        updated_time_diffs = - updated_last_update + predicted_time #relative time
        updated_time_diffs = (updated_time_diffs - self.mean_time_shift_src) / self.std_time_shift_src
        updated_node_embedding,neighbors,attention_score_weights = self.embedding_module.compute_embedding(memory=updated_memory,
                                                                         source_nodes=predicted_nodes,
                                                                         timestamps=np.asarray(
                                                                             [predicted_time for _ in
                                                                              predicted_nodes]),
                                                                         n_layers=self.n_layers,
                                                                         n_neighbors=n_neighbors,
                                                                         time_diffs=updated_time_diffs,
                                                                         edge_features=edge_features[
                                                                                       :len(predicted_nodes)])
        if self.model_mode==3:
            od_matrix = self.predict_od(updated_node_embedding,DTDG_output)
        else:
            od_matrix = self.predict_od(updated_node_embedding)
        return od_matrix,updated_node_embedding,neighbors,attention_score_weights

    def update_memory(self, nodes, messages):
        # Aggregate messages for the same nodes
        unique_nodes, unique_messages, unique_timestamps = self.message_aggregator.aggregate(nodes,messages)
        if len(unique_nodes) > 0:
            unique_messages = self.message_function.compute_message(unique_messages)

        self.memory_updater.update_memory(unique_nodes, unique_messages,
                                          timestamps=unique_timestamps)

    def get_updated_memory_new(self, nodes_list, messages_list, DTDG_output):
        # Aggregate messages for the same nodes
        divide = len(nodes_list)
        unique_nodes_list = []
        unique_messages_list = []
        for k in range(divide):
            nodes = nodes_list[k]
            messages = messages_list[k]
            unique_nodes, unique_messages, unique_timestamps = self.message_aggregator.aggregate(nodes,messages)
            unique_nodes_list.append(unique_nodes)
            unique_messages_list.append(unique_messages)
        updated_memory, updated_last_update = self.memory_updater.get_updated_memory(unique_nodes_list,
                                                                                     unique_messages_list,None,DTDG_output,
                                                                                     timestamps=unique_timestamps)
        return updated_memory, updated_last_update

    def get_updated_memory_DTDG(self, nodes_list, messages_list, DTDG_output):
        updated_memory, updated_last_update = self.DTDG_memory_updater.get_updated_memory(DTDG_output = DTDG_output)
        return updated_memory, updated_last_update

    def get_updated_memory_CTDG(self, nodes_list, messages_list):
        divide = len(nodes_list)
        unique_nodes_list = []
        unique_messages_list = []
        for k in range(divide):
            nodes = nodes_list[k]
            messages = messages_list[k]
            unique_nodes, unique_messages, unique_timestamps = self.message_aggregator.aggregate(nodes,messages)
            unique_nodes_list.append(unique_nodes)
            unique_messages_list .append(unique_messages)
        updated_memory, updated_last_update = self.CTDG_memory_updater.get_updated_memory(unique_nodes_list,
                                                                                     unique_messages_list,None,
                                                                                     timestamps=unique_timestamps)
        return updated_memory, updated_last_update

    def get_updated_memory(self, nodes, messages):
        unique_nodes, unique_messages, unique_timestamps = self.message_aggregator.aggregate(nodes,messages)
        if len(unique_nodes) > 0:
            unique_messages = self.message_function.compute_message(unique_messages)
        updated_memory, updated_last_update = self.memory_updater.get_updated_memory(unique_nodes,
                                                                                     unique_messages,
                                                                                     timestamps=unique_timestamps)
        return updated_memory, updated_last_update

    def get_raw_messages(self, source_nodes, destination_nodes, edge_times, edge_features, predicted_time):
        edge_times = torch.from_numpy(edge_times).float().to(self.device)
        destination_memory = self.memory.get_memory(destination_nodes)
        edge_time_delta = edge_times - predicted_time
        edge_time_delta_encoding = torch.exp(edge_time_delta / self.output_len).unsqueeze(1) * self.node_raw_features[destination_nodes]
        source_message = torch.cat([destination_memory, edge_features, edge_time_delta_encoding],dim=1)
        messages = dict()
        unique_sources = np.unique(source_nodes)
        for node_i in unique_sources:
            ind = (source_nodes == node_i)
            messages[node_i] = [source_message[ind], edge_times[ind]]
        return unique_sources, messages

    def set_neighbor_finder(self, neighbor_finder):
        self.neighbor_finder = neighbor_finder
        self.embedding_module.neighbor_finder = neighbor_finder

    def del_neighbor_finder(self):
        del self.neighbor_finder
        del self.embedding_module.neighbor_finder
