import xgboost as xgb
from sklearn.datasets import load_svmlight_file
import numpy as np
import src.lambda_obj_py
from src.lambda_obj_py import lambda_objective
from time import time

data_folder = '../../Data/hw3/'

def load_data(path):
    X_data, y_data, qid_data = load_svmlight_file(path, query_id=True)
    sorted_by_qid_idxs = np.argsort(qid_data)
    qid_data = qid_data[sorted_by_qid_idxs]
    X_data = X_data[sorted_by_qid_idxs]
    y_data = y_data[sorted_by_qid_idxs]
    group_sizes = np.unique(qid_data, return_counts=True)[1]
    return X_data, y_data, qid_data, group_sizes


def train_val_split(X_data, y_data, qid_data, group_sizes, test_size=0.2):
    queries_test_size = int(group_sizes.shape[0] * test_size)
    queries_train_size = group_sizes.shape[0] - queries_test_size
    print(group_sizes.shape[0], queries_test_size, queries_train_size)

    group_train, group_val = group_sizes[:queries_train_size], group_sizes[queries_train_size:]
    print(group_train.shape[0], group_val.shape[0])

    train_x_len = group_train.sum()
    print(X_data.shape[0], train_x_len, X_data.shape[0] - train_x_len)

    X_train, X_val = X_data[:train_x_len], X_data[train_x_len:]
    y_train, y_val = y_data[:train_x_len], y_data[train_x_len:]
    qid_train, qid_val = qid_data[:train_x_len], qid_data[train_x_len:]

    return X_train, y_train, qid_train, group_train, X_val, y_val, qid_val, group_val

X_data, y_data, qid_data, group_sizes = load_data(data_folder +'train_small.txt')
X_train, y_train, qid_train, group_train, \
X_val, y_val, qid_val, group_val = train_val_split(X_data, y_data, qid_data, group_sizes, test_size = 0.2)

dtrain = xgb.DMatrix(data = X_train, label = y_train)
dtrain.set_group(group_train)

dval = xgb.DMatrix(data = X_val, label = y_val)
dval.set_group(group_val)

def save_arr(arr, filename):
    file = open(filename, 'w')
    for v in arr:
        file.write(str(v) + '\n')
    file.close()

def mspe(F:np.array, dtrain:  xgb.DMatrix):
    Y = dtrain.get_label()

    # print("SAVING_ARRAYS:")
    # save_arr(F, 'src/F_test.txt')
    # save_arr(Y, 'src/Y_test.txt')
    # save_arr(group_train, 'src/group_test.txt')
    # print("ARRAYS SAVED")
    # exit(0)
    #
    # return None, None
    #t1 = time()
    grad, hess = lambda_objective(Y, F, 1.0, group_train)
    # print("grad_time(sec): ", time()-t1)

    return grad, hess


print("START TRAIN:")

params = {'objective': 'rank:pairwise', 'eta': 0.1, 'gamma': 1.0,
               'min_child_weight': 0.1, 'max_depth': 6, 'eval_metric': 'ndcg@5'}

# print("ORIGINAL OBJ: ")
# xgb_model = xgb.train(params, dtrain, num_boost_round=2, evals=[(dtrain, 'validation')],
#                      obj= mspe, verbose_eval=True)

print("MY OBJECTIVE")
xgb_model = xgb.train(params, dtrain, num_boost_round=2, evals=[(dtrain, 'validation')],
                     verbose_eval=True)