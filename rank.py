import xgboost as xgb
from sklearn.datasets import load_svmlight_file
import numpy as np
import src.lambda_obj_py
from src.lambda_obj_py import lambda_objective
from time import time
from metrics import calc_ndcg

data_folder = '../../Data/hw3/'



# In[3]:


def load_submission(filename):
    res = {}
    fin = open(filename, 'r')
    fin.readline()
    for l in fin.readlines():
        args = l.split(',')
        args = [x for x in args if len(x) > 0]
        if len(args) < 2:
            continue
        qid = int(args[0])
        xid = int(args[1]) - 1
        res[qid] = res.get(qid, []) + [xid]
    fin.close()
    return res


# In[16]:


def get_submission(pred_qids, preds):
    res = {}
    for qid in np.unique(pred_qids):
        res[qid] = []
        q_doc_idxs = np.argwhere(pred_qids == qid).ravel()
        q_doc_scores = preds[q_doc_idxs]

        sorted_doc_ids = 1 + q_doc_idxs[np.argsort(q_doc_scores)[::-1]]

        for did in sorted_doc_ids:
            # fout.write('{0},{1}\n'.format(qid, did))
            res[qid].append(did)
    return res


# In[4]:


def save_submission(pred_qids, preds, filename):
    fout = open(filename, 'w')
    fout.write('QueryId,DocumentId\n')

    for qid in np.unique(pred_qids):
        q_doc_idxs = np.argwhere(pred_qids == qid).ravel()
        q_doc_scores = preds[q_doc_idxs]


        sorted_doc_ids = 1 + q_doc_idxs[np.argsort(q_doc_scores)[::-1]]
        for did in sorted_doc_ids:
            fout.write('{0},{1}\n'.format(qid, did))

    fout.close()


# In[5]:


def load_data(path):
    X_data, y_data, qid_data = load_svmlight_file(path, query_id=True)
    sorted_by_qid_idxs = np.argsort(qid_data, kind='mergesort')
    print(sorted_by_qid_idxs)
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



X_train, y_train, qid_train, group_train = load_data(data_folder + 'train.txt')

X_test, y_test, qid_test, group_test = load_data(data_folder + 'test.txt')

# In[9]:


dtrain = xgb.DMatrix(data=X_train, label=y_train)
dtrain.set_group(group_train)

# dval = xgb.DMatrix(data = X_val, label = y_val)
# dval.set_group(group_val)

dtest = xgb.DMatrix(data=X_test)
dtest.set_group(group_test)


# In[11]:


def save_arr(arr, filename):
    file = open(filename, 'w')
    for v in arr:
        file.write(str(v) + '\n')
    file.close()


asessors_train_submission = get_submission(qid_train, y_train)
iter_num = 0

def mspe(F: np.array, dtrain: xgb.DMatrix):
    Y = dtrain.get_label()

    t1 = time()
    grad, hess = lambda_objective(Y, F, 1.0, group_train)
    print("grad_time(sec): ", time() - t1)

    return grad, hess


# In[ ]:


# print("START TRAIN:")
#'objective': 'rank:pairwise',
model_name = 'doob_hess-1-500-8-eta-0.3-xxxx'

params = {'objective': 'rank:pairwise', 'eta': 0.1, 'max_depth': 8, 'eval_metric': 'ndcg@5',
          'nthread': 16}

print("MY OBJECTIVE")
evres = dict()
xgb_model = xgb.train(params, dtrain, num_boost_round=500, evals=[(dtrain, 'train')],
                      obj=mspe,
                      evals_result=evres,
                      verbose_eval=True,
                      xgb_model='hess-1-500-8-eta-0.3-xxxx.xgb')

# print("XGBOOST MSE: ")

# params_reg = {'objective': 'reg:linear', 'eta': 0.2, 'max_depth': 10, 'eval_metric': 'ndcg@5'}
# reg_xgb_model = xgb.train(params_reg, dtrain, num_boost_round=20, evals=[(dtrain, 'train')],
#                       verbose_eval=True)


# In[30]:


xgb_model.save_model(model_name+'.xgb')


prediction_test = xgb_model.predict(dtest)
save_submission(qid_test, prediction_test, model_name + '_submission.txt')

np.save(model_name + '_test_pred.np', prediction_test)
#
# my_submission_test = get_submission(qid_test, prediction_test)
# fen_submission = load_submission('fen_submission.txt')

# In[26]:

#
# fen_ndcg = calc_ndcg(my_submission_test, fen_submission, k=5)
# print("FEN NDCG:")
# print("--MEAN: ", fen_ndcg.mean())
# print("--TRUE_RANKED: ", fen_ndcg[fen_ndcg > 0.95].shape[0])
# print("--ALL_RANKED: ", fen_ndcg.shape[0])
# print("--INCORRECT: ", fen_ndcg[fen_ndcg == 0.0].shape[0])
#
#
# prediction_train = xgb_model.predict(dtrain)
# my_train_submission = get_submission(qid_train, prediction_train)
#
#
# asessors_train_submission = get_submission(qid_train, y_train)
# train_ndcg = calc_ndcg(my_train_submission, asessors_train_submission, k =5)
#
# print("MY: ", my_train_submission)
# print("ACESSORS: ", asessors_train_submission)
#
# print("TRAIN NDCG: ")
# print("--MEAN: ", train_ndcg.mean())
# print("--TRUE_RANKED: ", train_ndcg[train_ndcg > 0.95].shape[0])
# print("--ALL_RANKED: ", train_ndcg.shape[0])
# print("--INCORRECT: ", train_ndcg[train_ndcg == 0.0].shape[0])


# for qid in np.unique(qid_train):
#     q_idxs = np.argwhere(qid_train == qid).ravel()
#     my_scores = prediction_train[q_idxs]
#     ass_scores = y_train[q_idxs]
#
#     my_sort_idxs = np.argsort(my_scores)[::-1]
#
#     print('QID: ', qid )
#     print("--MY_SCORES: ", my_scores[my_sort_idxs])
#     print("--ASS_SCORES: ", ass_scores[my_sort_idxs])




