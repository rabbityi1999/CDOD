import math
from tqdm import tqdm, trange
import numpy as np
import torch
# from sklearn.metrics import average_precision_score, mean_squared_error, mean_absolute_error
from utils.utils import get_neighbor_finder, build_od_matrix
from utils.data_processing import compute_time_statistics
import re
import pickle

def mean_squared_error(real, truth):
    return np.sqrt(np.mean((real-truth)**2))

def mean_absolute_error(real, truth):
    return np.mean(np.abs(real-truth))

def SMAPE(pred, gt):
    count = gt.shape[1] * gt.shape[0]
    ave = gt.sum(axis=(0, 1)) / count
    upsilon = math.sqrt(((gt - ave) ** 2).sum(axis=(0, 1)) / count)
    all_loss = 2 * abs(pred - gt) / (pred + gt + upsilon)
    all_loss = all_loss.sum(axis=(0, 1))
    all_loss = all_loss / count
    return float(all_loss)


def PCC(pred, gt):
    # print(pred)
    pred_s = pred.reshape(pred.shape[1] * pred.shape[0])
    print(pred_s)
    gt_s = gt.reshape(gt.shape[1] * gt.shape[0])
    print(gt_s)
    pccs = np.corrcoef(pred_s, gt_s)
    # print(pccs)
    return pccs[0][1]


# def eval_edge_prediction(model, negative_edge_sampler, data, n_neighbors, batch_size=200):
#     # Ensures the random sampler uses a seed for evaluation (i.e. we sample always the same
#     # negatives for validation / test set)
#     assert negative_edge_sampler.seed is not None
#     negative_edge_sampler.reset_random_state()
#
#     val_ap, val_auc = [], []
#     with torch.no_grad():
#         model = model.eval()
#         # While usually the test batch size is as big as it fits in memory, here we keep it the same
#         # size as the training batch size, since it allows the memory to be updated more frequently,
#         # and later test batches to access information from interactions in previous test batches
#         # through the memory
#         TEST_BATCH_SIZE = batch_size
#         num_test_instance = len(data.sources)
#         num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)
#
#         for k in range(num_test_batch):
#             s_idx = k * TEST_BATCH_SIZE
#             e_idx = min(num_test_instance, s_idx + TEST_BATCH_SIZE)
#             sources_batch = data.sources[s_idx:e_idx]
#             destinations_batch = data.destinations[s_idx:e_idx]
#             timestamps_batch = data.timestamps[s_idx:e_idx]
#             edge_idxs_batch = data.edge_idxs[s_idx: e_idx]
#
#             size = len(sources_batch)
#             _, negative_samples = negative_edge_sampler.sample(size)
#
#             pos_prob, neg_prob = model.compute_edge_probabilities(sources_batch, destinations_batch,
#                                                                   negative_samples, timestamps_batch,
#                                                                   edge_idxs_batch, n_neighbors)
#
#             pred_score = np.concatenate([(pos_prob).cpu().numpy(), (neg_prob).cpu().numpy()])
#             true_label = np.concatenate([np.ones(size), np.zeros(size)])
#
#             val_ap.append(average_precision_score(true_label, pred_score))
#             val_auc.append(roc_auc_score(true_label, pred_score))
#
#     return np.mean(val_ap), np.mean(val_auc)


def eval_od_prediction(model, data, edge_features, n_neighbors, st, ed, USE_MEMORY, args, mini_batchsize, number_node,
                       device,
                       batch_size=1, config=None):
    # Ensures the random sampler uses a seed for evaluation (i.e. we sample always the same
    # negatives for validation / test set)
    # model.memory.__init_memory__()
    input_len = config["input_len"]
    output_len = config["output_len"]
    day_cycle = config["day_cycle"]
    day_start = config["day_start"]
    day_end = config["day_end"]
    sample = config["sample"]
    n_nodes = config["n_nodes"]
    # val_rmse, val_mae = [], []
    # val_pcc, val_mape = [], []
    prediction = []
    label = []
    with torch.no_grad():
        model = model.eval()
        # While usually the test batch size is as big as it fits in memory, here we keep it the same
        # size as the training batch size, since it allows the memory to be updated more frequently,
        # and later test batches to access information from interactions in previous test batches
        # through the memory
        num_test_batch = (ed - st - input_len) // output_len

        head = 0
        tail1 = 0  # [head,tail1) real [tail1,tail2)
        tail2 = 0
        data_size = len(data.timestamps)
        batch_range = trange(num_test_batch)
        for j in batch_range:
            # print(
            #     "================================Test: Batch: %d/%d================================" % (
            #         j, num_test_batch))

            # if USE_MEMORY:
            #    model.memory.__init_memory__()
            st1 = j * output_len + st
            ed1 = j * output_len + st + input_len
            ed2 = j * output_len + st + input_len + output_len

            while head < data_size and data.timestamps[head] < st1:
                head += 1
            while tail1 < data_size and data.timestamps[tail1] < ed1:
                tail1 += 1
            while tail2 < data_size and data.timestamps[tail2] < ed2:
                tail2 += 1

            if ed1 % day_cycle < day_start or ed1 % day_cycle >= day_end:
                continue

            if head == tail1 or tail1 == tail2:
                continue

            start_idx = head
            end_idx = tail1
            now_time = ed1
            begin_time = st1

            sources_batch, destinations_batch = data.sources[start_idx:end_idx:sample], \
                                                data.destinations[start_idx:end_idx:sample]

            edge_idxs_batch = data.edge_idxs[start_idx:end_idx:sample]
            edge_features_batch = torch.Tensor(edge_features[edge_idxs_batch]).to(device)
            timestamps_batch = data.timestamps[start_idx:end_idx:sample]

            ngh_finder = get_neighbor_finder(data, args.uniform, head, tail1)
            model.del_neighbor_finder()
            model.set_neighbor_finder(ngh_finder)

            # model.compute_temporal_embeddings(
            #     sources_batch, destinations_batch, [], timestamps_batch, edge_idxs_batch)

            od_matrix_predicted = model.compute_od_matrix(sources_batch, destinations_batch,
                                                          timestamps_batch, edge_features_batch,
                                                          np.arange(n_nodes),
                                                          ed1,
                                                          n_neighbors)

            od_matrix_real = build_od_matrix(data.sources[tail1:tail2],
                                             data.destinations[tail1:tail2], n_nodes)

            prediction.append(od_matrix_predicted.cpu().numpy())
            label.append(od_matrix_real)

            # val_rmse.append(math.sqrt(mean_squared_error(od_matrix_real, od_matrix_predicted.cpu())))
            # val_mae.append(mean_absolute_error(od_matrix_real, od_matrix_predicted.cpu()))
            # val_mape.append(SMAPE(od_matrix_predicted.cpu(), od_matrix_real))

    stacked_prediction = np.stack(prediction)
    # stacked_prediction[stacked_prediction < 0] = 0
    stacked_label = np.stack(label)

    reshaped_prediction = stacked_prediction.reshape(-1)
    reshaped_label = stacked_label.reshape(-1)
    mse = mean_squared_error(reshaped_prediction, reshaped_label)
    mae = mean_absolute_error(reshaped_prediction, reshaped_label)
    pcc = np.corrcoef(reshaped_prediction, reshaped_label)[0][1]
    smape = np.mean(2 * np.abs(reshaped_prediction - reshaped_label) / (
            np.abs(reshaped_prediction) + np.abs(reshaped_label) + 1))
    print(mse, mae, pcc, smape)
    return mse, mae, pcc, smape, stacked_prediction, stacked_label
'''
val_batch_num = val_batch_num,
                                                                        val_input_batch = val_input_batch,
                                                                        val_od_matrix_real_list = val_od_matrix_real_list,
                                                                        val_st1_list = val_st1_list,
                                                                        val_ed1_list = val_ed1_list,
                                                                        val_start_idx_list = val_start_idx_list,
                                                                        val_end_idx_list = val_end_idx_list,
'''

def eval_od_prediction_DTDG(model, data, edge_features, n_neighbors,
                            val_batch_num, val_input_batch, val_od_matrix_real_list,val_st1_list,val_ed1_list,val_start_idx_list,val_end_idx_list,
    args, device, config=None):
    sample = config["sample"]
    n_nodes = config["n_nodes"]

    prediction = []
    label = []
    with torch.no_grad():
        model = model.eval()
        num_test_batch = val_batch_num
        batch_range = trange(num_test_batch)
        for j in batch_range:
            st1 = val_st1_list[j]
            ed1 = val_ed1_list[j]
            start_idx = val_start_idx_list[j]
            end_idx = val_end_idx_list[j]

            sources_batch, destinations_batch = data.sources[start_idx:end_idx:sample], data.destinations[start_idx:end_idx:sample]
            edge_idxs_batch = data.edge_idxs[start_idx:end_idx:sample]
            edge_features_batch = torch.Tensor(edge_features[edge_idxs_batch]).to(device)
            timestamps_batch = data.timestamps[start_idx:end_idx:sample]

            ngh_finder = get_neighbor_finder(data, args.uniform, start_idx, end_idx)
            model.del_neighbor_finder()
            model.set_neighbor_finder(ngh_finder)

            od_matrix_predicted = model.compute_od_matrix(sources_batch, destinations_batch,
                                                          timestamps_batch, edge_features_batch,val_input_batch[j],
                                                          np.arange(n_nodes),
                                                          ed1,
                                                          n_neighbors)

            od_matrix_real = val_od_matrix_real_list[j]
            prediction.append(od_matrix_predicted.cpu().numpy())
            label.append(od_matrix_real.cpu().numpy())

    stacked_prediction = np.stack(prediction)
    stacked_label = np.stack(label)

    reshaped_prediction = stacked_prediction.reshape(-1)
    reshaped_label = stacked_label.reshape(-1)
    mse = mean_squared_error(reshaped_prediction, reshaped_label)
    mae = mean_absolute_error(reshaped_prediction, reshaped_label)
    pcc = np.corrcoef(reshaped_prediction, reshaped_label)[0][1]
    smape = np.mean(2 * np.abs(reshaped_prediction - reshaped_label) / (
            np.abs(reshaped_prediction) + np.abs(reshaped_label) + 1))
    print(mse, mae, pcc, smape)
    return mse, mae, pcc, smape, stacked_prediction, stacked_label



def eval_od_prediction_CTDTGRU(model, data, edge_features, n_neighbors,DTDG_DIV,
                            val_batch_num, val_input_batch3, val_od_matrix_real_list,val_ed1_list3,val_start_list3,val_end_list3,
    args, device, config=None):
    sample = config["sample"]
    n_nodes = config["n_nodes"]
    embeddings = []
    prediction = []
    label = []
    neighbors_list=[]
    attention_score_weights_list=[]
    with torch.no_grad():
        model = model.eval()
        num_test_batch = val_batch_num

        batch_range = trange(num_test_batch)

        for j in batch_range:
            ed1 = val_ed1_list3[j * DTDG_DIV]
            start_idx = val_start_list3[j * DTDG_DIV:(j + 1) * DTDG_DIV]  # 现在有3个
            end_idx = val_end_list3[j * DTDG_DIV:(j + 1) * DTDG_DIV]
            ngh_finder = get_neighbor_finder(data, args.uniform, start_idx[0], end_idx[-1])
            model.del_neighbor_finder()
            model.set_neighbor_finder(ngh_finder)
            # Train using only training graph
            od_matrix_real = val_od_matrix_real_list[j]

            for k in range(DTDG_DIV):
                # print("k:", k)
                # print("start_idx[k]:", start_idx[k])
                # print("end_idx[k]:", end_idx[k])
                sources_batch3 = data.sources[start_idx[k]:end_idx[k]:sample]
                destinations_batch3 = data.destinations[start_idx[k]:end_idx[k]:sample]
                timestamps_batch3 = data.timestamps[start_idx[k]:end_idx[k]:sample]
                edge_idxs_batch = data.edge_idxs[start_idx[k]:end_idx[k]:sample]
                edge_features_batch3 = torch.Tensor(edge_features[edge_idxs_batch]).to(device)
                od_matrix_predicted,updated_node_embedding,neighbors,attention_score_weights = model.compute_od_matrix(sources_batch3, destinations_batch3,
                                                              timestamps_batch3, edge_features_batch3,val_input_batch3[j][k],
                                                              np.arange(n_nodes),
                                                              ed1,
                                                              n_neighbors)
                if k==(DTDG_DIV-1):
                    embeddings.append(updated_node_embedding.detach().cpu().numpy())
                    prediction.append(od_matrix_predicted.cpu().numpy())
                    label.append(od_matrix_real.cpu().numpy())
                    neighbors_list.append(neighbors.cpu().numpy())
                    attention_score_weights_list.append(attention_score_weights.cpu().numpy())

    stacked_prediction = np.stack(prediction)
    stacked_label = np.stack(label)



    reshaped_prediction = stacked_prediction.reshape(-1)
    reshaped_label = stacked_label.reshape(-1)
    truth_mean = np.mean(label)

    print("truth_mean:", truth_mean)
    mse = mean_squared_error(reshaped_prediction, reshaped_label)
    mae = mean_absolute_error(reshaped_prediction, reshaped_label)
    pcc = np.corrcoef(reshaped_prediction, reshaped_label)[0][1]
    smape = np.mean(2 * np.abs(reshaped_prediction - reshaped_label) / (np.abs(reshaped_prediction) + np.abs(reshaped_label) + 1))

    mask = reshaped_prediction >= truth_mean
    mse2 = mean_squared_error(reshaped_prediction[mask], reshaped_label[mask])
    mae2 = mean_absolute_error(reshaped_prediction[mask], reshaped_label[mask])
    pcc2 = np.corrcoef(reshaped_prediction[mask], reshaped_label[mask])[0][1]
    smape2 = np.mean(
        2 * np.abs(reshaped_prediction[mask] - reshaped_label[mask]) / (np.abs(reshaped_prediction[mask]) + np.abs(reshaped_label[mask]) + 1))

    print(mse, mae, pcc, smape)
    return mse, mae, pcc, smape,mse2, mae2, pcc2, smape2, stacked_prediction, stacked_label,embeddings,neighbors_list,attention_score_weights_list

'''
val_input_batch3 完全没变，只有CTDG的变了
'''
def eval_od_prediction_CTDTGRU_unaligned(model, path,data, edge_features, n_neighbors,DTDG_DIV,
                             val_input_batch3,val_start_list3,val_end_list3,
    args, device, config=None):
    sample = config["sample"]
    n_nodes = config["n_nodes"]
    prediction = []
    label = []
    data_dict = pickle.load(open(path,"rb"))
    offset_od_matrix_list = data_dict["offset_od_matrix_list"]
    off_st_time_list = data_dict["off_st_time_list"]
    off_start_idx_list = data_dict["off_start_idx_list"]
    with torch.no_grad():
        model = model.eval()
        print("len off_st_time_list:",len(off_st_time_list))
        print("len val_start_list3:",len(val_start_list3))
        print("len offset_od_matrix_list:",len(offset_od_matrix_list))
        last_end = val_start_list3[0]
        for j in range(len(offset_od_matrix_list)-1): #比正常的要少预测一个
            # ed1 = val_ed1_list3[j * DTDG_DIV]
            # print("j:",j)
            ed1 = off_st_time_list[j]
            start_idx = val_start_list3[j * DTDG_DIV:(j + 1) * DTDG_DIV]  # 现在有3个
            # start_idx[0] = last_end #我改的部分

            end_idx = val_end_list3[j * DTDG_DIV:(j + 1) * DTDG_DIV]
            # end_idx[-1] = off_start_idx_list[j] #我改的部分2
            last_end = off_start_idx_list[j]
            ngh_finder = get_neighbor_finder(data, args.uniform, start_idx[0], end_idx[-1])
            model.del_neighbor_finder()
            model.set_neighbor_finder(ngh_finder)
            # Train using only training graph
            for k in range(DTDG_DIV): # 现在是按照 小于10分钟来写的
                # print("k:",k)
                # print("start_idx[k]:",start_idx[k])
                # print("end_idx[k]:",end_idx[k])
                if start_idx[k]!=end_idx[k]:
                    sources_batch3 = data.sources[start_idx[k]:end_idx[k]:sample]
                    destinations_batch3 = data.destinations[start_idx[k]:end_idx[k]:sample]
                    timestamps_batch3 = data.timestamps[start_idx[k]:end_idx[k]:sample]
                    edge_idxs_batch = data.edge_idxs[start_idx[k]:end_idx[k]:sample]
                    edge_features_batch3 = torch.Tensor(edge_features[edge_idxs_batch]).to(device)
                    od_matrix_predicted = model.compute_od_matrix(sources_batch3, destinations_batch3,
                                                                  timestamps_batch3, edge_features_batch3,val_input_batch3[j][k],
                                                                  np.arange(n_nodes),
                                                                  ed1,
                                                                  n_neighbors)
                    if k==(DTDG_DIV-1):
                        prediction.append(od_matrix_predicted.cpu().numpy())
                        label.append(offset_od_matrix_list[j])

    stacked_prediction = np.stack(prediction)
    stacked_label = np.stack(label)
    reshaped_prediction = stacked_prediction.reshape(-1)
    reshaped_label = stacked_label.reshape(-1)
    mse = mean_squared_error(reshaped_prediction, reshaped_label)
    mae = mean_absolute_error(reshaped_prediction, reshaped_label)
    pcc = np.corrcoef(reshaped_prediction, reshaped_label)[0][1]
    smape = np.mean(2 * np.abs(reshaped_prediction - reshaped_label) / (np.abs(reshaped_prediction) + np.abs(reshaped_label) + 1))
    print(mse, mae, pcc, smape)
    return mse, mae, pcc, smape, stacked_prediction, stacked_label

def eval_od_prediction2(data, edge_features, n_neighbors, st, ed, USE_MEMORY, args, mini_batchsize, number_node, device,
                        batch_size=1, config=None):
    # Ensures the random sampler uses a seed for evaluation (i.e. we sample always the same
    # negatives for validation / test set)
    # model.memory.__init_memory__()
    input_len = config["input_len"]
    output_len = config["output_len"]
    day_cycle = config["day_cycle"]
    day_start = config["day_start"]
    day_end = config["day_end"]
    sample = config["sample"]
    n_nodes = config["n_nodes"]
    # val_rmse, val_mae = [], []
    # val_pcc, val_mape = [], []
    prediction = []
    label = []
    with torch.no_grad():
        # While usually the test batch size is as big as it fits in memory, here we keep it the same
        # size as the training batch size, since it allows the memory to be updated more frequently,
        # and later test batches to access information from interactions in previous test batches
        # through the memory
        num_test_batch = (ed - st - input_len) // output_len

        head = 0
        tail1 = 0  # [head,tail1) real [tail1,tail2)
        tail2 = 0
        data_size = len(data.timestamps)
        batch_range = trange(num_test_batch)
        for j in batch_range:
            # print(
            #     "================================Test: Batch: %d/%d================================" % (
            #         j, num_test_batch))

            # if USE_MEMORY:
            #    model.memory.__init_memory__()
            st1 = j * output_len + st
            ed1 = j * output_len + st + input_len
            ed2 = j * output_len + st + input_len + output_len

            while head < data_size and data.timestamps[head] < st1:
                head += 1
            while tail1 < data_size and data.timestamps[tail1] < ed1:
                tail1 += 1
            while tail2 < data_size and data.timestamps[tail2] < ed2:
                tail2 += 1

            if ed1 % day_cycle < day_start or ed1 % day_cycle >= day_end:
                continue

            if head == tail1 or tail1 == tail2:
                continue

            od_matrix_predicted = build_od_matrix(data.sources[head:tail1],
                                                  data.destinations[head:tail1], n_nodes)

            od_matrix_real = build_od_matrix(data.sources[tail1:tail2],
                                             data.destinations[tail1:tail2], n_nodes)

            prediction.append(od_matrix_predicted)
            label.append(od_matrix_real)

            # val_rmse.append(math.sqrt(mean_squared_error(od_matrix_real, od_matrix_predicted.cpu())))
            # val_mae.append(mean_absolute_error(od_matrix_real, od_matrix_predicted.cpu()))
            # val_mape.append(SMAPE(od_matrix_predicted.cpu(), od_matrix_real))

    stacked_prediction = np.stack(prediction)
    stacked_prediction[stacked_prediction < 0] = 0
    stacked_label = np.stack(label)

    reshaped_prediction = stacked_prediction.reshape(-1)
    reshaped_label = stacked_label.reshape(-1)
    mse = mean_squared_error(reshaped_prediction, reshaped_label)
    mae = mean_absolute_error(reshaped_prediction, reshaped_label)
    pcc = np.corrcoef(reshaped_prediction, reshaped_label)[0][1]
    smape = np.mean(2 * np.abs(reshaped_prediction - reshaped_label) / (
            np.abs(reshaped_prediction) + np.abs(reshaped_label) + 1))
    print(mse, mae, pcc, smape)
    return mse, mae, pcc, smape, stacked_prediction, stacked_label


# def eval_node_classification(tgn, decoder, data, edge_idxs, batch_size, n_neighbors):
#     pred_prob = np.zeros(len(data.sources))
#     num_instance = len(data.sources)
#     num_batch = math.ceil(num_instance / batch_size)
#
#     with torch.no_grad():
#         decoder.eval()
#         tgn.eval()
#         for k in range(num_batch):
#             s_idx = k * batch_size
#             e_idx = min(num_instance, s_idx + batch_size)
#
#             sources_batch = data.sources[s_idx: e_idx]
#             destinations_batch = data.destinations[s_idx: e_idx]
#             timestamps_batch = data.timestamps[s_idx:e_idx]
#             edge_idxs_batch = edge_idxs[s_idx: e_idx]
#
#             source_embedding, destination_embedding, _ = tgn.compute_temporal_embeddings(sources_batch,
#                                                                                          destinations_batch,
#                                                                                          destinations_batch,
#                                                                                          timestamps_batch,
#                                                                                          edge_idxs_batch,
#                                                                                          n_neighbors)
#             pred_prob_batch = decoder(source_embedding).sigmoid()
#             pred_prob[s_idx: e_idx] = pred_prob_batch.cpu().numpy()
#
#     auc_roc = roc_auc_score(data.labels, pred_prob)
#     return auc_roc
