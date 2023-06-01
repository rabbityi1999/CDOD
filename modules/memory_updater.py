from torch import nn
import torch
import torch.nn.functional as F

class MemoryUpdater(nn.Module):
    def update_memory(self, unique_node_ids, unique_messages, timestamps):
        pass

class SequenceMemoryUpdater(MemoryUpdater):
    def __init__(self, memory, message_dimension, memory_dimension, device):
        super(SequenceMemoryUpdater, self).__init__()
        self.memory = memory
        self.layer_norm = torch.nn.LayerNorm(memory_dimension)
        self.message_dimension = message_dimension
        self.device = device

    def get_updated_memory(self, unique_node_ids, unique_messages, timestamps):
        if len(unique_node_ids) <= 0:
            return self.memory.memory.data.clone(), self.memory.last_update.data.clone()

        assert (self.memory.get_last_update(unique_node_ids) <= timestamps).all().item(), "Trying to " \
                                                                                          "update memory to time in the past"

        updated_memory = self.memory.memory.data.clone()
        updated_memory[unique_node_ids] = self.memory_updater(unique_messages, updated_memory[unique_node_ids])
        updated_last_update = self.memory.last_update.data.clone()
        updated_last_update[unique_node_ids] = timestamps
        self.memory.set_memory(unique_node_ids, updated_memory.detach())

        return updated_memory, updated_last_update

class GRUMemoryUpdater(SequenceMemoryUpdater):
    def __init__(self, memory, message_dimension, memory_dimension, device):
        super(GRUMemoryUpdater, self).__init__(memory, message_dimension, memory_dimension, device)

        self.memory_updater = nn.GRUCell(input_size=message_dimension,
                                         hidden_size=memory_dimension)

class RNNMemoryUpdater(SequenceMemoryUpdater):
    def __init__(self, memory, message_dimension, memory_dimension, device):
        super(RNNMemoryUpdater, self).__init__(memory, message_dimension, memory_dimension, device)

        self.memory_updater = nn.RNNCell(input_size=message_dimension,
                                         hidden_size=memory_dimension)

class DTDGMemoryUpdater(MemoryUpdater):
    def __init__(self, memory, message_dimension, memory_dimension, device,n_nodes,hidden_dim):
        super(DTDGMemoryUpdater, self).__init__()
        self.device = device
        self.n_nodes = n_nodes
        self.memory = memory
        self.memory_dimension = memory_dimension
        self.layer_norm = torch.nn.LayerNorm(memory_dimension)
        self.message_dimension = message_dimension
        self.W_i = nn.Linear(hidden_dim , 3*memory_dimension)
        self.W_h = nn.Linear(memory_dimension , 3*memory_dimension)

    def get_updated_memory(self, DTDG_output, memory=None):
        if memory is None:
            updated_memory = self.memory.memory.data.clone()
        else:
            updated_memory = memory

        DTDG_output = DTDG_output.squeeze()
        i = self.W_i(DTDG_output)
        h = self.W_h(updated_memory)
        r = i[:, :self.memory_dimension] + h[:, :self.memory_dimension]
        z = i[:, self.memory_dimension:self.memory_dimension * 2] + h[:, self.memory_dimension:self.memory_dimension * 2]
        n = torch.tanh(i[:, self.memory_dimension * 2:] + torch.sigmoid(r) * h[:, self.memory_dimension * 2:])
        output = (1 - torch.sigmoid(z)) * n + torch.sigmoid(z) * updated_memory
        updated_memory = output
        updated_last_update = self.memory.last_update.data.clone()
        node_idx = [_ for _ in range(len(updated_memory))]
        self.memory.set_memory(node_idx, updated_memory.detach())

        return updated_memory, updated_last_update


class CTDTMemoryUpdater(MemoryUpdater):
    def __init__(self, memory, message_dimension, memory_dimension, device,n_nodes,hidden_dim):
        super(CTDTMemoryUpdater, self).__init__()
        self.device = device
        self.n_nodes = n_nodes
        self.memory = memory
        self.memory_dimension = memory_dimension
        self.layer_norm = torch.nn.LayerNorm(memory_dimension)
        self.message_dimension = message_dimension
        self.W_C = nn.Linear(message_dimension,2*memory_dimension)
        self.W_D = nn.Linear(hidden_dim,2*memory_dimension)
        self.W_h = nn.Linear(memory_dimension,4*memory_dimension)

    def update_memory(self, unique_node_ids, unique_messages, timestamps):
        if len(unique_node_ids) <= 0:
            return

        assert (self.memory.get_last_update(unique_node_ids) <= timestamps).all().item(), "Trying to " \
                                                                                          "update memory to time in the past"
        memory = self.memory.get_memory(unique_node_ids)
        self.memory.last_update[unique_node_ids] = timestamps
        updated_memory = self.memory_updater(unique_messages, memory)
        self.memory.set_memory(unique_node_ids, updated_memory.detach())

    def get_updated_memory(self, unique_node_ids_list, unique_messages_list, CTDT_messages, DTDG_output, timestamps=None, memory=None):
        if len(unique_node_ids_list[0]) <= 0:
            return self.memory.memory.data.clone(), self.memory.last_update.data.clone()
        if memory is None:
            updated_memory = self.memory.memory.data.clone()
        else:
            updated_memory = memory
        divide = len(unique_node_ids_list)
        for k in range(divide-1):
            unique_message = unique_messages_list[k]
            unique_node_ids = unique_node_ids_list[k]
            i_C = self.W_C(unique_message)
            h = self.W_h(updated_memory[unique_node_ids])

            Gate_C = i_C[:, : self.memory_dimension] + h[: , self.memory_dimension : self.memory_dimension * 2]
            h_C = torch.tanh(i_C[:, self.memory_dimension:] + h[:, self.memory_dimension * 3:])
            updated_memory[unique_node_ids] = (1-torch.sigmoid(Gate_C))*h_C + torch.sigmoid(Gate_C)*updated_memory[unique_node_ids]

        unique_message = unique_messages_list[-1]
        unique_node_ids = unique_node_ids_list[-1]
        i_C = self.W_C(unique_message)
        i_D = self.W_D(DTDG_output[unique_node_ids])
        h = self.W_h(updated_memory[unique_node_ids])

        Gate_D = i_D[:, :self.memory_dimension] + h[:, :self.memory_dimension]
        Gate_C = i_C[:, :self.memory_dimension] + h[:, self.memory_dimension:self.memory_dimension * 2]

        h_D = torch.tanh(i_D[:, self.memory_dimension : ] + h[:, self.memory_dimension * 2: self.memory_dimension * 3])
        h_C = torch.tanh(i_C[:, self.memory_dimension : self.memory_dimension*2 ] + h[:, self.memory_dimension * 3: ])

        output = (torch.sigmoid(Gate_D)/2) * h_D  + (torch.sigmoid(Gate_C)/2) * h_C + ((2 - torch.sigmoid(Gate_D) - torch.sigmoid(Gate_C))/2) * updated_memory[unique_node_ids]

        updated_memory[unique_node_ids] = output
        updated_last_update = self.memory.last_update.data.clone()
        if timestamps is not None:
            assert (self.memory.get_last_update(unique_node_ids) <= timestamps).all().item(), "Trying to " \
                                                                                              "update memory to time in the past"
            updated_last_update[unique_node_ids] = timestamps.float()

        self.memory.set_memory(unique_node_ids, updated_memory[unique_node_ids].detach())
        return updated_memory, updated_last_update


class CTDGMemoryUpdater(MemoryUpdater):
    def __init__(self, memory, message_dimension, memory_dimension, device,n_nodes,hidden_dim):
        super(CTDGMemoryUpdater, self).__init__()
        self.device = device
        self.n_nodes = n_nodes
        self.memory = memory
        self.memory_dimension = memory_dimension
        self.layer_norm = torch.nn.LayerNorm(memory_dimension)
        self.message_dimension = message_dimension
        self.W_C = nn.Linear(message_dimension,2*memory_dimension)
        self.W_h = nn.Linear(memory_dimension,2*memory_dimension)

    def get_updated_memory(self, unique_node_ids_list, unique_messages_list, CTDT_messages, timestamps=None, memory=None):
        if len(unique_node_ids_list[0]) <= 0:
            return self.memory.memory.data.clone(), self.memory.last_update.data.clone()
        if memory is None:
            updated_memory = self.memory.memory.data.clone()
        else:
            updated_memory = memory
        divide = len(unique_node_ids_list)
        for k in range(divide):
            unique_message = unique_messages_list[k]
            unique_node_ids = unique_node_ids_list[k]
            i_C = self.W_C(unique_message)
            h = self.W_h(updated_memory[unique_node_ids])
            Gate_C = i_C[:, : self.memory_dimension] + h[: , :self.memory_dimension ]
            h_C = torch.tanh(i_C[:, self.memory_dimension:] + h[:, self.memory_dimension : ])
            updated_memory[unique_node_ids] = (1-torch.sigmoid(Gate_C))*h_C + torch.sigmoid(Gate_C)*updated_memory[unique_node_ids]
        updated_last_update = self.memory.last_update.data.clone()
        if timestamps is not None:
            assert (self.memory.get_last_update(unique_node_ids) <= timestamps).all().item(), "Trying to " \
                                                                                              "update memory to time in the past"
            updated_last_update[unique_node_ids] = timestamps.float()
        self.memory.set_memory(unique_node_ids, updated_memory[unique_node_ids].detach())
        return updated_memory, updated_last_update


def get_memory_updater(module_type, memory, message_dimension, memory_dimension, device, n_nodes,hidden_dim=200):
    if module_type == "gru":
        return GRUMemoryUpdater(memory, message_dimension, memory_dimension, device)
    elif module_type == "rnn":
        return RNNMemoryUpdater(memory, message_dimension, memory_dimension, device)
    elif module_type== "CT+DT":
        return CTDTMemoryUpdater(memory, message_dimension, memory_dimension, device,n_nodes,hidden_dim)
    elif module_type == "DTDG":
        return DTDGMemoryUpdater(memory, message_dimension, memory_dimension, device, n_nodes, hidden_dim)
    elif module_type == "CTDG":
        return CTDGMemoryUpdater(memory, message_dimension, memory_dimension, device, n_nodes, hidden_dim)
