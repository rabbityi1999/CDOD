import numpy as np

class Data:
    def __init__(self, sources, destinations, timestamps, edge_idxs, labels):
        self.sources = sources
        self.destinations = destinations
        self.timestamps = timestamps
        self.edge_idxs = edge_idxs
        self.labels = labels
        self.n_interactions = len(sources)
        self.unique_nodes = set(sources) | set(destinations)
        self.n_unique_nodes = len(self.unique_nodes)

class MyData:
    def __init__(self, sources, destinations, timestamps, edge_idxs, number_node):
        self.sources = sources
        self.destinations = destinations
        self.timestamps = timestamps
        self.edge_idxs = edge_idxs
        self.n_interactions = len(sources)
        self.unique_nodes = range(number_node)  # 老韩的代码里面是 self.unique_nodes = set(list(range(n_nodes)))
        self.n_unique_nodes = len(self.unique_nodes)

def compute_time_statistics(sources, destinations, timestamps):
    last_timestamp_sources = dict()
    last_timestamp_dst = dict()
    all_timediffs_src = []
    all_timediffs_dst = []
    for k in range(len(sources)):
        source_id = sources[k]
        dest_id = destinations[k]
        c_timestamp = timestamps[k]
        if source_id not in last_timestamp_sources.keys():
            last_timestamp_sources[source_id] = 0
        if dest_id not in last_timestamp_dst.keys():
            last_timestamp_dst[dest_id] = 0
        all_timediffs_src.append(c_timestamp - last_timestamp_sources[source_id])
        all_timediffs_dst.append(c_timestamp - last_timestamp_dst[dest_id])
        last_timestamp_sources[source_id] = c_timestamp
        last_timestamp_dst[dest_id] = c_timestamp
    assert len(all_timediffs_src) == len(sources)
    assert len(all_timediffs_dst) == len(sources)
    mean_time_shift_src = np.mean(all_timediffs_src)
    std_time_shift_src = np.std(all_timediffs_src)
    mean_time_shift_dst = np.mean(all_timediffs_dst)
    std_time_shift_dst = np.std(all_timediffs_dst)
    return mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst

def upper_bound(nums, target):
    l, r = 0, len(nums) - 1
    pos = -1
    while l <= r:
        mid = int((l + r) / 2)
        if nums[mid] > target:
            r = mid - 1
            pos = mid
        else:  # >
            l = mid + 1
    return pos

def get_od1_data_new(config, n_nodes, node_feature_len):
    whole_data = np.load(config["data_path"]).astype("int").reshape([-1, 3])
    print("data loaded")
    val_time = (config["train_day"]) * config["day_cycle"]
    test_time = (config["train_day"] + config["val_day"]) * config["day_cycle"]
    all_time = (config["train_day"] + config["val_day"] + config["test_day"]) * config["day_cycle"]
    sources = whole_data[:, 0]
    destinations = whole_data[:, 1]
    timestamps = whole_data[:, 2]
    edge_idxs = np.arange(whole_data.shape[0])
    n_nodes = config["n_nodes"]
    if config["dataname"] == "NY":
        node_features = np.concatenate([np.diag(np.ones(config["n_nodes"])), np.zeros([config["n_nodes"], 1])], axis=1)
    else:
        node_features = np.diag(np.ones(config["n_nodes"]))
    train_mask = upper_bound(timestamps, val_time)
    val_mask = upper_bound(timestamps, test_time)
    train_data = MyData(sources[:train_mask], destinations[:train_mask], timestamps[:train_mask],
                        edge_idxs[:train_mask],
                        n_nodes)
    val_data = MyData(sources[train_mask:val_mask], destinations[train_mask:val_mask], timestamps[train_mask:val_mask],
                      edge_idxs[train_mask:val_mask], n_nodes)
    test_data = MyData(sources[val_mask:], destinations[val_mask:], timestamps[val_mask:], edge_idxs[val_mask:],
                       n_nodes)
    edge_features = np.ones(
        int(len(destinations[:train_mask]) + len(destinations[train_mask:val_mask]) + len(timestamps[val_mask:])))
    return node_features, edge_features, train_data, val_data, test_data, val_time, test_time, all_time

def get_od1_data_old(config, n_nodes, node_feature_len):
    data_loc = {
        "BJ": ["D:/Workspace/ODPrediction/data/Beijing/train.npy", "D:/Workspace/ODPrediction/data/Beijing/test.npy",
               "D:/Workspace/ODPrediction/data/Beijing/val.npy"],
        "NY": ["D:/Workspace/ODPrediction/data/NYtaxi/train.npy", "D:/Workspace/ODPrediction/data/NYtaxi/val.npy",
               "D:/Workspace/ODPrediction/data/NYtaxi/test.npy"]
    }
    train = np.load(data_loc[config["dataname"]][0])
    val = np.load(data_loc[config["dataname"]][1])
    test = np.load(data_loc[config["dataname"]][2])
    print(train[-2], val[-2], test[-2])
    edge_features = np.ones(int((len(train) + len(val) + len(test)) // 4))
    if config["dataname"] == "NY":
        node_features = np.concatenate([np.diag(np.ones(config["n_nodes"])), np.zeros([config["n_nodes"], 1])], axis=1)
    else:
        node_features = np.diag(np.ones(config["n_nodes"]))

    val_time = config["train_day"] * config["day_cycle"]
    test_time = (config["train_day"] + config["val_day"]) * config["day_cycle"]
    all_time = (config["train_day"] + config["val_day"] + config["test_day"]) * config["day_cycle"]
    train_data = MyData(train[0::4], train[1::4], train[2::4],
                        range(int(len(train)) // 4), train[3::4], config["n_nodes"])
    st = int(len(train)) // 4
    val_data = MyData(val[0::4], val[1::4], val[2::4],
                      range(st, st + int(len(val)) // 4), val[3::4], config["n_nodes"])

    st = st + int(len(val)) // 4
    test_data = MyData(test[0::4], test[1::4], test[2::4],
                       range(st, st + int(len(test)) // 4), test[3::4], config["n_nodes"])

    print("The training dataset has {} interactions, involving {} different nodes".format(
        train_data.n_interactions, train_data.n_unique_nodes))
    print("The validation dataset has {} interactions, involving {} different nodes".format(
        val_data.n_interactions, val_data.n_unique_nodes))
    print("The test dataset has {} interactions, involving {} different nodes".format(
        test_data.n_interactions, test_data.n_unique_nodes))

    return node_features, edge_features, train_data, val_data, test_data, val_time, test_time, all_time

def get_od_data(config):
    whole_data = np.load(config["data_path"]).astype("int").reshape([-1, 3])
    print("data loaded")
    all_time = (config["train_day"] + config["val_day"] + config["test_day"]) * config["day_cycle"]
    val_time, test_time = (config["train_day"]) * config["day_cycle"], (config["train_day"] + config["val_day"]) * \
                          config["day_cycle"]
    sources = whole_data[:, 0]
    destinations = whole_data[:, 1]
    timestamps = whole_data[:, 2]
    edge_idxs = np.arange(whole_data.shape[0])
    n_nodes = config["n_nodes"]
    node_features = np.diag(np.ones(n_nodes))

    train_mask = upper_bound(timestamps, val_time)
    val_mask = upper_bound(timestamps, test_time)
    full_data = Data(sources, destinations, timestamps, edge_idxs, n_nodes)
    train_data = Data(sources[:train_mask], destinations[:train_mask], timestamps[:train_mask], edge_idxs[:train_mask],
                      n_nodes)
    val_data = Data(sources[train_mask:val_mask], destinations[train_mask:val_mask], timestamps[train_mask:val_mask],
                    edge_idxs[train_mask:val_mask], n_nodes)
    test_data = Data(sources[val_mask:], destinations[val_mask:], timestamps[val_mask:], edge_idxs[val_mask:], n_nodes)

    print("The dataset has {} interactions, involving {} different nodes".format(full_data.n_interactions,
                                                                                 full_data.n_unique_nodes))
    print("The training dataset has {} interactions, involving {} different nodes".format(
        train_data.n_interactions, train_data.n_unique_nodes))
    print("The validation dataset has {} interactions, involving {} different nodes".format(
        val_data.n_interactions, val_data.n_unique_nodes))
    print("The test dataset has {} interactions, involving {} different nodes".format(
        test_data.n_interactions, test_data.n_unique_nodes))

    return n_nodes, node_features, full_data, train_data, val_data, test_data, val_time, test_time, all_time