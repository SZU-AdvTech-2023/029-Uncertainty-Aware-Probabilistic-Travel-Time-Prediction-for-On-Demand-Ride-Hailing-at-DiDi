def generate_config():
    d_model = 128
    deep_config = {
        "dense": {"size": 1},  # Distance
        "sparse": [
            {"col": 0, "name": "Day", "size": 7, "dim": 32},
            {"col": 1, "name": "Time", "size": 96, "dim": 32},
        ],
        "out_dim": 128
    }
    wide_config = {
        "dense": {"size": 1},  # Distance
        "sparse": [
            {"col": 0, "name": "Day", "size": 7, "dim": 32},
            {"col": 1, "name": "Time", "size": 96, "dim": 32},
        ],
        "out_dim": 128
    }
    rnn_config = {
        "dense": {"size": 1},  # LinkDistances,
        "sparse": [
            {"col": 0, "name": "LinkIDs", "size": 43897, "dim": 64},
        ],
        "input_size": 128,
        "hidden_size": 128,
        "num_layers": 2,
        "num_directions": 1,
    }
    return wide_config, deep_config, rnn_config
