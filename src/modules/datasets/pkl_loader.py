import pickle


def load_data(path, is_channel_first=False):
    with open(path, 'rb') as f:
        content = pickle.load(f)
    # check if validation set exists
    if len(content) == 4:
        train_x, train_y, test_x, test_y = content
        val_x, val_y = None, None
    else:
        train_x, train_y, val_x, val_y, test_x, test_y = content
    # correct shape
    if len(train_x.shape) == 2:
        train_x = train_x.reshape(*train_x.shape, 1)
        train_y = train_x.reshape(*train_y.shape, 1)
    elif is_channel_first:
        train_x = train_x.transpose(0, 2, 1)
        test_x = test_x.transpose(0, 2, 1)
    return train_x, train_y, val_x, val_y, test_x, test_y
