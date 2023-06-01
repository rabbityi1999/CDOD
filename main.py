import math
import logging
import time
import sys
import argparse
import torch
import shutil
import numpy as np
import pickle
from pathlib import Path
import random
import os
from torch.autograd import Variable
from tqdm import tqdm, trange
from evaluation.evaluation import eval_od_prediction_CTDTGRU
from model.CDOD import CDOD
from utils.utils import EarlyStopMonitor, RandEdgeSampler, get_neighbor_finder, build_od_matrix
from utils.data_processing import get_od1_data_new, compute_time_statistics

'''
model for final version
'''

config = {
    "Beijing": {
        "dataname": "BJ",
        "data_path": "/home/xuyi/OD_pred/ODdata/Beijing/final3.npy",
        "preprocess_data_path": "/home/xuyi/OD_pred/ODdata/Beijing/",
        "input_len": 30,
        "output_len": 30,
        "mini_batchsize": 300000,
        "day_cycle": 1440,
        "day_start": 360,
        "day_end": 1320,
        "mean": 2.169455659096243,
        "std": 7.707586221204564,
        "sample": 5,
        "val": 0.75,
        "test": 0.875,
        "start_weekday": 5,
        "n_nodes": 268,
        "train_day": 42,
        "val_day": 7,
        "test_day": 7
    },
    "NewYork": {
        "dataname": "NY",
        "data_path": "/home/xuyi/OD_pred/ODdata/NYtaxi/final3.npy",
        "preprocess_data_path": "/home/xuyi/OD_pred/ODdata/NYtaxi/",
        "input_len": 1800,
        "output_len": 1800,
        "mini_batchsize": 300000,
        "day_cycle": 86400,
        "day_start": -1,
        "day_end": 86401,
        "mean": 2.169455659096243,  # seems no use
        "std": 7.707586221204564,  # seems no use
        "sample": 1,
        "start_weekday": 1,
        "n_nodes": 63,
        "train_day": 139,
        "val_day": 21,
        "test_day": 21
    },
    "Sample_NewYork": {
        "dataname": "Sample_NY",
        "data_path": "/home/xuyi/OD_pred/data/NYTaxi/sampled.npy",
        "DTDG_path": '/home/xuyi/OD_pred/ODdata/DTDG_data/Sampled_NYTaxi/',
        "preprocess_data_path": "/home/xuyi/OD_pred/ODdata/sampled_NYtaxi/",
        "CTDTOD_path": '/home/xuyi/OD_pred/ODdata/CTDTOD_data/Sampled_NYTaxi/',
        "input_len": 1800,
        "output_len": 1800,
        "mini_batchsize": 300000,
        "day_cycle": 86400,
        "train_day": 5,
        "val_day": 1,
        "test_day": 1,
        "day_start": -1,
        "day_end": 86401,
        "sample": 1,
        "n_nodes": 63
    },
    "debug": {
        "input_len": 30,
        "output_len": 30,
        "mini_batchsize": 300000,
        "day_cycle": 1440,
        "day_start": 480,
        "day_end": 1320,
        "mean": 2.169455659096243,
        "std": 7.707586221204564,
        "sample": 3,
        "val": 0.75,
        "test": 0.875,
        "start_weekday": 5
    }
}

### Argument and global variables
parser = argparse.ArgumentParser('CDOD self-supervised training')
parser.add_argument('-d', '--data', type=str, help='Dataset name (eg. wikipedia or reddit)',
                    default='NewYork')
# data : complete_data bike taxi
parser.add_argument('--bs', type=int, default=1, help='Batch_size')
parser.add_argument('--window', type=int, default=1, help='window size')
parser.add_argument('--time_mode', type=str, default="simple", help='simple//complex_add//complex_add2')
parser.add_argument('--concat_mode', type=int, default=0, help='0:concat 1:+/2 ')
parser.add_argument('--hidden_dim', type=int, default=200, help='Dimensions of the DTDG')

'''
NY
max_times 3553
min_times 58
average times 1732

BJ
max_times 34028
min_times 974
average times 3346
'''
parser.add_argument('--model_mode', type=int, default=0, help='0:DTDG+CTDG 1:DTDG 2:CTDG 3:ablation co-updater use direct concat')
parser.add_argument('--lr', type=float, default=0.00001, help='Learning rate')
parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')#default=5
parser.add_argument('--divide_base', type=int, default=1500, help='every how many edges CTDG update')#Beijing 10000 要参考record_times统计量设置
parser.add_argument('--DTDG_divide', type=int, default=1, help='divide dtdg input ')#Beijing 10000
parser.add_argument('--divide_pred', type=bool, default=True, help='if predict and backward at each divide')#Beijing 10000
parser.add_argument('--prefix', type=str, default='NY_n_degree=16', help='Prefix to name the checkpoints') #如果在一个数据集上面同时跑，务必写好predix
parser.add_argument('--device', type=str, default="cuda:3", help='Idx for the gpu to use: cpu, cuda:0, etc.')
parser.add_argument('--loss', type=str, default="MSEloss", choices=["MSEloss", "odloss"], help='Type of memory updater')
parser.add_argument('--n_runs', type=int, default=1, help='Number of   runs')

parser.add_argument('--memory_updater', type=str, default="gru", choices=["gru", "rnn","CTDT"], help='Type of memory updater')
parser.add_argument('--hidden_layer', type=int, default=2, help='Dimensions of the time embedding default=2')
parser.add_argument('--n_degree', type=int, default=16, help='Number of neighbors to sample default=10')
parser.add_argument('--n_head', type=int, default=2, help='Number of heads used in attention layer default=2') #query_dim=276 NY72
parser.add_argument('--n_epoch', type=int, default=500, help='Number of epochs')
parser.add_argument('--n_layer', type=int, default=1, help='Number of network layers')
parser.add_argument('--drop_out', type=float, default=0.1, help='Dropout probability')
parser.add_argument('--node_dim', type=int, default=64, help='Dimensions of the node embedding default 64')
parser.add_argument('--time_dim', type=int, default=64, help='Dimensions of the time embedding')
parser.add_argument('--backprop_every', type=int, default=1, help='Every how many batches to '
                                                                  'backprop')
parser.add_argument('--use_memory', action='store_true',
                    help='Whether to augment the model with a node memory') #store_true就代表着一旦有这个参数，做出动作“将其值标为True”，也就是没有时，默认状态下其值为False。反之亦然，store_false也就是默认为True，一旦命令中有此参数，其值则变为False。
parser.add_argument('--embedding_module', type=str, default="graph_attention", choices=[
    "graph_attention", "graph_sum", "identity", "time"], help='Type of embedding module')
parser.add_argument('--message_function', type=str, default="identity", choices=[
    "mlp", "identity"], help='Type of message function')
parser.add_argument('--aggregator', type=str, default="hybrid", help='Type of message '
                                                                  'aggregator: last mean sum hybrid')
parser.add_argument('--memory_update_at_end', action='store_true',
                    help='Whether to updmeate memory at the end or at the start of the batch')
parser.add_argument('--message_dim', type=int, default=100, help='Dimensions of the messages')
parser.add_argument('--memory_dim', type=int, default=268, help='Dimensions of the memory for '
                                                                'each user NY:64 BJ:268')
parser.add_argument('--seed', type=int, default=0, help='Random seed')
parser.add_argument('--different_new_nodes', action='store_true',
                    help='Whether to use disjoint set of new nodes for train and val')
parser.add_argument('--uniform', action='store_true',
                    help='take uniform sampling from temporal neighbors')
parser.add_argument('--randomize_features', action='store_true',
                    help='Whether to randomize node features')
parser.add_argument('--use_destination_embedding_in_message', action='store_true',
                    help='Whether to use the embedding of the destination node as part of the message')
parser.add_argument('--use_source_embedding_in_message', action='store_true',
                    help='Whether to use the embedding of the source node as part of the message')
parser.add_argument('--dyrep', action='store_true',
                    help='Whether to run the dyrep model')

try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)

divide_pred = args.divide_pred
DTDG_DIV = args.DTDG_divide
divide_base = args.divide_base
BATCH_SIZE = args.bs
NUM_NEIGHBORS = args.n_degree
NUM_NEG = 1
NUM_EPOCH = args.n_epoch
NUM_HEADS = args.n_head
DROP_OUT = args.drop_out
DATA = args.data
NUM_LAYER = args.n_layer
LEARNING_RATE = args.lr
NODE_DIM = args.node_dim
TIME_DIM = args.time_dim
USE_MEMORY = args.use_memory
MESSAGE_DIM = args.message_dim
MEMORY_DIM = args.memory_dim
device = args.device
hidden_layer = args.hidden_layer
hidden_dim = args.hidden_dim
concat_mode = args.concat_mode
model_mode = args.model_mode
Time_mode = args.time_mode
mini_batchsize = 10000
Path("./saved_models/").mkdir(parents=True, exist_ok=True)
Path("./saved_checkpoints/").mkdir(parents=True, exist_ok=True)
MODEL_SAVE_PATH = f'./saved_models/{args.prefix}-{args.data}-{args.concat_mode}-{args.model_mode}_CDOD+DTDG.pth'
get_checkpoint_path = lambda \
        epoch: f'./saved_checkpoints/{args.prefix}-{args.data}-{epoch}.pth'
get_checkpoint_model_path = lambda \
        epoch: f'./saved_checkpoints/{args.prefix}-{args.data}-{epoch}-model.pth'

### set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
Path("log/").mkdir(parents=True, exist_ok=True)
fh = logging.FileHandler('log/{}_{}.log'.format(str(time.time()), args.prefix))
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.WARN)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)
logger.info(args)
begin_CDOD_time = time.time()
### Extract data for training, validation and testing
node_features, edge_features ,train_data, val_data, test_data \
    , val_time, test_time, all_time = get_od1_data_new(config=config[DATA], n_nodes=config[DATA]["n_nodes"],
                                                       node_feature_len=args.memory_dim)

input_len = config[DATA]['input_len']
print("input_len:",input_len)
output_len = config[DATA]['output_len']
day_cycle = config[DATA]['day_cycle']
day_start = config[DATA]['day_start']
day_end = config[DATA]['day_end']
sample = config[DATA]['sample']
n_nodes = config[DATA]['n_nodes']
DTDG_path_dir = config[DATA]['preprocess_data_path']
mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst = 0.22967577600236688, 4.494076813452364, 0.2296868367119358, 4.341163674494456
Window_length = args.window


class OD_loss(torch.nn.Module):
    def __init__(self):
        super(OD_loss, self).__init__()
        self.pro = torch.nn.ReLU()

    def forward(self, predict, truth):
        mask = (truth < 1)
        mask2 = (predict > 0)
        loss = torch.mean(((predict - truth) ** 2) * ~mask + ((mask2 * predict - truth) ** 2) * mask)
        return loss

if args.loss == "odloss":
    logger.info("self od loss!!!!!")
    criterion = OD_loss()
else:
    criterion = torch.nn.MSELoss()

def genrate_tensor_matrix(path):
    data_dict = pickle.load(open(path,"rb"))

    st1_list3 = data_dict['st1_list3']
    ed1_list3 = data_dict['ed1_list3']
    start_idx_list3 = data_dict['start_idx_list3']
    end_idx_list3 = data_dict['end_idx_list3']
    od_matrix_list3 = torch.tensor(data_dict['od_matrix_list3']).to(device)
    od_matrix_list = torch.tensor(data_dict['od_matrix_list']).to(device)

    batch_num = len(od_matrix_list) - Window_length # 这里制作batch采用的是截取法
    batch_range = trange(batch_num)

    input_batch = Variable(torch.zeros(batch_num, Window_length, n_nodes, n_nodes)).to(device)
    input_batch3 = Variable(torch.zeros(batch_num, DTDG_DIV, n_nodes, n_nodes)).to(device)
    std_list3_tensor = []
    ed1_list3_tensor = []
    start_list3_tensor = []
    end_list3_tensor = []
    od_matrix_real_list = Variable(torch.zeros(batch_num, n_nodes, n_nodes)).to(device)  # 里面的元素有batch 个

    for batch_idx in batch_range:
        input_batch[batch_idx] = od_matrix_list[batch_idx : batch_idx + Window_length]
        od_matrix_real_list[batch_idx] = od_matrix_list[batch_idx + Window_length]

        for k in range(DTDG_DIV):
            input_batch3[batch_idx][k] = od_matrix_list3[batch_idx*DTDG_DIV +k]
            std_list3_tensor.append(st1_list3[batch_idx * DTDG_DIV + k])
            ed1_list3_tensor.append(ed1_list3[batch_idx * DTDG_DIV + k])
            start_list3_tensor.append(start_idx_list3[batch_idx * DTDG_DIV + k])
            end_list3_tensor.append(end_idx_list3[batch_idx * DTDG_DIV + k])

    return batch_num, od_matrix_real_list,input_batch3, std_list3_tensor, ed1_list3_tensor, start_list3_tensor, end_list3_tensor

def train_main():
    now_seed = args.seed
    test_mae_list = []
    rmse_list = []
    pcc_list = []
    smape_list = []
    # 已经没有用的变量了 train_input_batch st1_list ed1_list start_idx_list end_idx_list
    train_batch_num, train_od_matrix_real_list, train_input_batch3, train_std_list3, train_ed1_list3, train_start_list3, train_end_list3 = \
        genrate_tensor_matrix(DTDG_path_dir + "trainDTCT_" + str(DTDG_DIV) + "GRU.pkl")

    val_batch_num, val_od_matrix_real_list, val_input_batch3, val_std_list3, val_ed1_list3, val_start_list3, val_end_list3 = \
        genrate_tensor_matrix(DTDG_path_dir + "valDTCT_" + str(DTDG_DIV) + "GRU.pkl")

    test_batch_num, test_od_matrix_real_list, test_input_batch3, test_std_list3, test_ed1_list3, test_start_list3, test_end_list3 = \
        genrate_tensor_matrix(DTDG_path_dir + "testDTCT_" + str(DTDG_DIV) + "GRU.pkl")
    for i in range(args.n_runs):
        random.seed(now_seed)
        torch.manual_seed(now_seed)
        torch.cuda.manual_seed_all(now_seed)
        np.random.seed(now_seed)
        logger.info( '%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5')
        logger.info('the seed is {}'.format(now_seed))
        print( "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5")
        print("now seed", now_seed)
        now_seed += 10
        results_path = "results/{}_{}.pkl".format(args.prefix, i) if i > 0 else "results/{}.pkl".format(args.prefix)
        Path("results/").mkdir(parents=True, exist_ok=True)

        # Initialize Model
        model = CDOD(neighbor_finder=None, node_features=node_features,  # change neighbor_finder
                  hidden_layer=hidden_layer, hidden_dim=hidden_dim, concat_mode=concat_mode,
                  model_mode = model_mode,
                  divide_base=divide_base,DTDG_DIV = DTDG_DIV,
                  device=device,
                  n_layers=NUM_LAYER,
                  n_heads=NUM_HEADS, dropout=DROP_OUT, use_memory=USE_MEMORY,
                  message_dimension=MESSAGE_DIM, memory_dimension=MEMORY_DIM,
                  memory_update_at_start=not args.memory_update_at_end,
                  embedding_module_type=args.embedding_module,
                  message_function=args.message_function,
                  aggregator_type=args.aggregator,
                  memory_updater_type=args.memory_updater,
                  n_neighbors=NUM_NEIGHBORS,
                  mean_time_shift_src=mean_time_shift_src, std_time_shift_src=std_time_shift_src,
                  mean_time_shift_dst=mean_time_shift_dst, std_time_shift_dst=std_time_shift_dst,
                  use_destination_embedding_in_message=args.use_destination_embedding_in_message,
                  use_source_embedding_in_message=args.use_source_embedding_in_message,
                  dyrep=args.dyrep, output_len=output_len)

        model.set_time_static(mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst)
        # criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        model = model.to(device)

        val_mses = []
        epoch_times = []
        total_epoch_times = []
        train_losses = []
        early_stopper = EarlyStopMonitor(max_round=args.patience, higher_better=False)
        num_batch = train_batch_num

        min_times = 100000
        max_times = 0
        record_times=[]
        bigger_than = 0
        for epoch in range(NUM_EPOCH):
            print("================================Epoch: %d================================" % epoch)
            start_epoch = time.time()
            logger.info('start {} epoch'.format(epoch))
            m_loss = []
            if USE_MEMORY:
                model.memory.__init_memory__()
            model = model.train()
            batch_range = trange(num_batch)
            for j in batch_range:
                ed1 = train_ed1_list3[j*DTDG_DIV]
                start_idx = train_start_list3[j*DTDG_DIV:(j+1)*DTDG_DIV]
                end_idx = train_end_list3[j*DTDG_DIV:(j+1)*DTDG_DIV]
                train_ngh_finder = get_neighbor_finder(train_data, args.uniform, start_idx[0], end_idx[-1])
                model.del_neighbor_finder()
                model.set_neighbor_finder(train_ngh_finder)
                od_matrix_real = train_od_matrix_real_list[j]
                for k in range(DTDG_DIV):
                    sources_batch3 = train_data.sources[start_idx[k]:end_idx[k]:sample]
                    destinations_batch3 = train_data.destinations[start_idx[k]:end_idx[k]:sample]
                    timestamps_batch3 = train_data.timestamps[start_idx[k]:end_idx[k]:sample]
                    edge_idxs_batch = train_data.edge_idxs[start_idx[k]:end_idx[k]:sample]
                    edge_features_batch3 = torch.Tensor(edge_features[edge_idxs_batch]).to(device)
                    od_matrix_predicted,updated_node_embedding,neighbors,attention_score_weights = model.compute_od_matrix(sources_batch3, destinations_batch3,
                                                                timestamps_batch3, edge_features_batch3, train_input_batch3[j][k],
                                                                np.arange(n_nodes),
                                                                ed1,
                                                                NUM_NEIGHBORS)

                    if divide_pred==True or k==(DTDG_DIV-1):
                        optimizer.zero_grad()
                        loss = criterion(od_matrix_predicted, od_matrix_real)
                        loss.backward()
                        optimizer.step()
                        m_loss.append(loss.item())
                        batch_range.set_description(f"train_loss: {m_loss[-1]};")
                        torch.cuda.empty_cache()

            epoch_time = time.time() - start_epoch
            epoch_times.append(epoch_time)

            val_mse, val_mae, val_pcc, val_smape,\
            val_mse2, val_mae2, val_pcc2, val_smape2, \
            _, _,embeddings,neighbors_list,attention_score_weights_list = eval_od_prediction_CTDTGRU(model=model,
                                                                                 data=val_data,
                                                                                 edge_features=edge_features,
                                                                                 n_neighbors=NUM_NEIGHBORS,
                                                                                 DTDG_DIV=DTDG_DIV,
                                                                                 val_batch_num=val_batch_num,
                                                                                 val_input_batch3=val_input_batch3,
                                                                                 val_od_matrix_real_list=val_od_matrix_real_list,
                                                                                 val_ed1_list3=val_ed1_list3,
                                                                                 val_start_list3 = val_start_list3,
                                                                                 val_end_list3 = val_end_list3,
                                                                                 args=args,
                                                                                 device=device, config=config[DATA])
            val_mses.append(val_mse)
            train_losses.append(np.mean(m_loss))
            total_epoch_time = time.time() - start_epoch
            total_epoch_times.append(total_epoch_time)
            # Save temporary results to disk
            logger.info('epoch: {} took {:.2f}s'.format(epoch, total_epoch_time))
            logger.info('Epoch mean train loss: {}'.format(np.mean(m_loss)))
            logger.info(
                f'Epoch val metric: mae, mse, rmse, pcc, smape, {val_mae}, {val_mse}, {np.sqrt(val_mse)}, {val_pcc}, {val_smape}')
            pickle.dump({
                "val_mses": val_mses,
                # "new_nodes_val_aps": new_nodes_val_aps,
                "train_losses": train_losses,
                "epoch_times": epoch_times,
                "total_epoch_times": total_epoch_times
            }, open(results_path, "wb"))

            # Early stopping
            if early_stopper.early_stop_check(val_mse):
                logger.info('No improvement over {} epochs, stop training'.format(early_stopper.max_round))
                best_model_path = get_checkpoint_path(early_stopper.best_epoch)
                shutil.copy(best_model_path, MODEL_SAVE_PATH)
                break
            else:
                print(get_checkpoint_path(epoch))
                torch.save({"state_dict": model.state_dict(), "memory": model.memory.backup_memory()},
                           get_checkpoint_path(epoch))
                torch.save(model, get_checkpoint_model_path(epoch))

        logger.info(f'Loading the best model at epoch {early_stopper.best_epoch}')
        best_model_path = get_checkpoint_path(early_stopper.best_epoch)
        best_model = torch.load(best_model_path)
        model.load_state_dict(best_model["state_dict"])
        model.memory.restore_memory(best_model["memory"])
        logger.info(f'Loaded the best model at epoch {early_stopper.best_epoch} for inference')
        model.eval()
        ### Test
        test_mse, test_mae, test_pcc, test_smape,test_mse2, test_mae2, test_pcc2, test_smape2, \
        prediction, label,embeddings ,neighbors_list,attention_score_weights_list = eval_od_prediction_CTDTGRU(model=model,
                                                                                         data=test_data,
                                                                                         edge_features=edge_features,
                                                                                         n_neighbors=NUM_NEIGHBORS,
                                                                                         DTDG_DIV=DTDG_DIV,
                                                                                         val_batch_num=test_batch_num,
                                                                                         val_input_batch3=test_input_batch3,
                                                                                         val_od_matrix_real_list=test_od_matrix_real_list,
                                                                                         val_ed1_list3=test_ed1_list3,
                                                                                         val_start_list3=test_start_list3,
                                                                                         val_end_list3=test_end_list3,
                                                                                         args=args,
                                                                                         device=device, config=config[DATA])

        logger.info(
            'Test statistics:-- mae: {}, mse: {}, rmse: {}, pcc: {}, smape:{}'.format(test_mae, test_mse, np.sqrt(test_mse),
                                                                                      test_pcc, test_smape))

        logger.info(
            'Test statistics:-- mae2: {}, mse2: {}, rmse2: {}, pcc2: {}, smape2:{}'.format(test_mae2, test_mse2,
                                                                                      np.sqrt(test_mse2),
                                                                                      test_pcc2, test_smape2))

        test_mae_list.append(test_mae)
        rmse_list.append(np.sqrt(test_mse))
        pcc_list.append(test_pcc)
        smape_list.append(test_smape)
        print("time spent:",time.time() - begin_CDOD_time)
        print("model_mode:",model_mode,"args.prefix:",args.prefix,"args.lr:",args.lr,"hidden dim:",hidden_dim,"Window_length:",Window_length,"Time_mode:",Time_mode,"now_seed:",now_seed)
        # print(args)
        # Save results for this run
        pickle.dump({
            "val_mses": val_mses,
            # "new_nodes_val_aps": new_nodes_val_aps,
            "test_mse": test_mse,
            "test_mae": test_mae,
            "test_pcc": test_pcc,
            "test_smape": test_smape,
            # "new_node_test_ap": nn_test_ap,
            "epoch_times": epoch_times,
            "train_losses": train_losses,
            "total_epoch_times": total_epoch_times,
            "prediction": prediction,
            "label": label,
            "embeddings":embeddings,
            "neighbors_list":neighbors_list,
            "attention_score_weights_list":attention_score_weights_list
        }, open(results_path, "wb"))
        logger.info('Saving CDOD model')
        logger.info('CDOD model saved')

    mae_np = np.array(test_mae_list)
    rmse_np = np.array(rmse_list)
    pcc_np = np.array(pcc_list)
    smape_np = np.array(smape_list)
    logger.info(
        'Test mean all:-- mae: {}, rmse: {}, pcc: {}, smape:{}'.format(np.mean(mae_np), np.mean(rmse_np), np.mean(pcc_np),
                                                                                   np.mean(smape_np)))
    logger.info(
        'Test var all:-- mae: {}, rmse: {}, pcc: {}, smape:{}'.format(
            np.sqrt(np.var(mae_np)), np.sqrt(np.var(rmse_np)), np.sqrt(np.var(pcc_np)),
            np.sqrt(np.var(smape_np))))
    print(
        'Test mean all:-- mae: {}, rmse: {}, pcc: {}, smape:{}'.format(np.mean(mae_np), np.mean(rmse_np),
                                                                                      np.mean(pcc_np),
                                                                                      np.mean(smape_np)))
    print('Test var all:-- mae: {}, rmse: {}, pcc: {}, smape:{}'.format(
            np.sqrt(np.var(mae_np)), np.sqrt(np.var(rmse_np)), np.sqrt(np.var(pcc_np)),
            np.sqrt(np.var(smape_np))))

def data_proprecess():

    if not os.path.exists(DTDG_path_dir):
        os.mkdir(DTDG_path_dir)

    phase_list = ['train','val','test']
    final_time = [val_time, test_time - val_time , all_time - test_time ]
    abs_time = [0, val_time, test_time]
    data_size = [train_data.n_interactions, val_data.n_interactions,test_data.n_interactions]
    data_list = [train_data,val_data,test_data]

    for idx,phase in enumerate(phase_list):
        data_dict = {}  # final save
        print("phase = ",phase)
        od_matrix_list = []
        od_matrix_list3 = []
        st1_list3 = []
        ed1_list3 = []
        start_idx_list3 = []
        end_idx_list3 = []
        num_batch = (final_time[idx] - input_len) // output_len # the sum of time matrix
        ed_set = []
        tail_set = []
        for item in range(DTDG_DIV+1):
            tail_set.append(0)
            ed_set.append(0)
        for j in trange(num_batch):
            for item in range(DTDG_DIV+1):
                ed_set[item] = j * output_len + abs_time[idx] + (input_len//DTDG_DIV) * item
                # print("j:",j,"item:",item,"ed_set:",ed_set[item])
            if ed_set[-1] % day_cycle < day_start or ed_set[-1] % day_cycle > day_end:
                continue
            for item in range(DTDG_DIV+1):
                while tail_set[item] < data_size[idx] and data_list[idx].timestamps[tail_set[item]] < ed_set[item]:
                    tail_set[item] += 1
            if tail_set[0] == tail_set[-1]:
                continue
            for item in range(DTDG_DIV):
                matrix_1 = build_od_matrix(data_list[idx].sources[tail_set[item]:tail_set[item+1]], data_list[idx].destinations[tail_set[item]:tail_set[item+1]], n_nodes)
                od_matrix_list3.append(matrix_1)
                st1_list3.append(ed_set[item])
                ed1_list3.append(ed_set[item+1])
                start_idx_list3.append(tail_set[item])
                end_idx_list3.append(tail_set[item+1])

            matrix = build_od_matrix(data_list[idx].sources[tail_set[0]:tail_set[-1]], data_list[idx].destinations[tail_set[0]:tail_set[-1]], n_nodes)
            od_matrix_list.append(matrix)

        data_dict['od_matrix_list'] = od_matrix_list
        data_dict['od_matrix_list3'] = od_matrix_list3
        data_dict['st1_list3'] = st1_list3
        data_dict['ed1_list3'] = ed1_list3
        data_dict['start_idx_list3'] = start_idx_list3
        data_dict['end_idx_list3'] = end_idx_list3

        pickle.dump(data_dict,open(DTDG_path_dir + phase_list[idx]+"DTCT_"+str(DTDG_DIV)+"GRU.pkl","wb"))


if __name__ == '__main__':

    train_main()
    sys.exit(0)
