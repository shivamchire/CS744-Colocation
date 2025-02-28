import warnings
warnings.filterwarnings("ignore")
import multiprocessing as mp
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import glog
import pandas as pd

from recoder.model import Recoder
from recoder.data import RecommendationDataset
from recoder.metrics import AveragePrecision, Recall, NDCG
from recoder.nn import DynamicAutoencoder, MatrixFactorization
from recoder.utils import dataframe_to_csr_matrix

data_dir = 'data/ml-20m/pro_sg/'
model_dir = 'models/ml-20m/'

common_params = {
  'user_col': 'uid',
  'item_col': 'sid',
  'inter_col': 'watched',
}

glog.info('Loading Data...')

train_df = pd.read_csv(data_dir + 'train.csv')
val_tr_df = pd.read_csv(data_dir + 'validation_tr.csv')
val_te_df = pd.read_csv(data_dir + 'validation_te.csv')

# uncomment it to train with MatrixFactorization
# train_df = train_df.append(val_tr_df)

train_matrix, item_id_map, _ = dataframe_to_csr_matrix(train_df, **common_params)
val_tr_matrix, _, user_id_map = dataframe_to_csr_matrix(val_tr_df, item_id_map=item_id_map,
                                                        **common_params)
val_te_matrix, _, _ = dataframe_to_csr_matrix(val_te_df, item_id_map=item_id_map,
                                              user_id_map=user_id_map, **common_params)

train_dataset = RecommendationDataset(train_matrix)
val_tr_dataset = RecommendationDataset(val_tr_matrix, val_te_matrix)


use_cuda = True

model = DynamicAutoencoder(hidden_layers=[200], activation_type='tanh',
                           noise_prob=0.5, sparse=False)

# model = MatrixFactorization(embedding_size=200, activation_type='tanh',
#                             dropout_prob=0.5, sparse=False)

trainer = Recoder(model=model, use_cuda=use_cuda, optimizer_type='adam',
                  loss='logistic', user_based=False)

# trainer.init_from_model_file(model_dir + 'bce_ns_d_0.0_n_0.5_200_epoch_50.model')
model_checkpoint = model_dir + 'bce_ns_d_0.0_n_0.5_200'

metrics = [Recall(k=20, normalize=True), Recall(k=50, normalize=True),
           NDCG(k=100)]

try:
  trainer.train(train_dataset=train_dataset, val_dataset=val_tr_dataset,
                batch_size=5, lr=1e-3, weight_decay=2e-5,
                num_epochs=100, negative_sampling=True,
                lr_milestones=[60, 80], num_data_workers=mp.cpu_count() if use_cuda else 0,
                model_checkpoint_prefix=model_checkpoint,
                checkpoint_freq=10, eval_num_recommendations=100,
                metrics=metrics, eval_freq=10)
except (KeyboardInterrupt, SystemExit):
  trainer.save_state(model_checkpoint)
  raise
