import argparse
import logging
import json
import time
import os
import mxnet as mx
from mxnet import gluon, nd, ndarray
from mxnet.metric import MSE
import numpy as np

os.system('pip install pandas')
import pandas as pd

logging.basicConfig(level=logging.DEBUG)

#########
# Globals
#########

##########
# Training
##########

def train(channel_input_dirs, hyperparameters, hosts, current_host, num_gpus, model_dir, **kwargs):
    
    batch_size = hyperparameters.get('batch-size', 1024)
    # get data
    training_dir = channel_input_dirs['train']
    testing_dir = channel_input_dirs['test']
    user_index_dir = channel_input_dirs['user_index']
    item_index_dir = channel_input_dirs['item_index']
    train_iter = load_train_data(training_dir, batch_size)
    test_iter = load_test_data(testing_dir, batch_size)
    df_user_index = load_user_index_data(user_index_dir)
    df_item_index = load_item_index_data(item_index_dir)
    
    # get hyperparameters
    num_embeddings = hyperparameters.get('num-embeddings', 64)
    opt = hyperparameters.get('opt', 'sgd')
    lr = hyperparameters.get('lr', 0.02)
    momentum = hyperparameters.get('momentum', 0.9)
    wd = hyperparameters.get('wd', 0.)
    epochs = hyperparameters.get('epochs', 5)

    # define net
    if num_gpus is None:
        num_gpus = hyperparameters.get('num-gpus', 1)
    if num_gpus > 0:
        ctx = mx.gpu()
    else:
        ctx = mx.cpu()

    net = MFBlock(max_users=df_user_index.shape[0], 
                  max_items=df_item_index.shape[0],
                  num_emb=num_embeddings,
                  dropout_p=0.5)
    
    net.collect_params().initialize(mx.init.Xavier(magnitude=60),
                                    ctx=ctx,
                                    force_reinit=True)
    net.hybridize()

    trainer = gluon.Trainer(net.collect_params(),
                            opt,
                            {'learning_rate': lr,
                             'wd': wd,
                             'momentum': momentum})
    
    # execute
    trained_net = execute(train_iter, test_iter, net, trainer, epochs, ctx, batch_size)
    
    return trained_net, df_user_index, df_item_index


class MFBlock(gluon.HybridBlock):
    def __init__(self, max_users, max_items, num_emb, dropout_p=0.5):
        super(MFBlock, self).__init__()
        
        self.max_users = max_users
        self.max_items = max_items
        self.dropout_p = dropout_p
        self.num_emb = num_emb
        
        with self.name_scope():
            self.user_embeddings = gluon.nn.Embedding(max_users, num_emb)
            self.item_embeddings = gluon.nn.Embedding(max_items, num_emb)

            self.dropout_user = gluon.nn.Dropout(dropout_p)
            self.dropout_item = gluon.nn.Dropout(dropout_p)

            self.dense_user   = gluon.nn.Dense(num_emb, activation='relu')
            self.dense_item = gluon.nn.Dense(num_emb, activation='relu')
            
    def hybrid_forward(self, F, users, items):
        a = self.user_embeddings(users)
        a = self.dense_user(a)
        
        b = self.item_embeddings(items)
        b = self.dense_item(b)

        predictions = self.dropout_user(a) * self.dropout_item(b)      
        predictions = F.sum(predictions, axis=1)

        return predictions

    
def execute(train_iter, test_iter, net, trainer, epochs, ctx, batch_size):
    loss_function = gluon.loss.L2Loss()
    for e in range(epochs):
        print("epoch: {}".format(e))
        for i, (user, item, label) in enumerate(train_iter):

                user = user.as_in_context(ctx)
                item = item.as_in_context(ctx)
                label = label.as_in_context(ctx)

                with mx.autograd.record():
                    output = net(user, item)               
                    loss = loss_function(output, label)
                loss.backward()
                trainer.step(batch_size)

        eval1 = eval_net(train_iter, net, ctx, loss_function, batch_size)
        eval2 = eval_net(test_iter, net, ctx, loss_function, batch_size)
        print(f"EPOCH {e}: MSE TRAINING {eval1}, MSE TEST: {eval2}")
        
    print("end of training")
    return net


def eval_net(data, net, ctx, loss_function, batch_size):
    acc = MSE()
    for i, (user, item, label) in enumerate(data):

            user = user.as_in_context(ctx)
            item = item.as_in_context(ctx)
            label = label.as_in_context(ctx)

            predictions = net(user, item).reshape((batch_size, 1))
            acc.update(preds=[predictions], labels=[label])

    return acc.get()[1]


def save(model_dir, model, df_user_index=None, df_item_index=None):
    import os
    
    os.makedirs(model_dir)
    model.save_parameters('{}/model.params'.format(model_dir))
    f = open('{}/MFBlock.params'.format(model_dir), 'w')
    json.dump({'max_users': net.max_users,
               'max_items': net.max_items,
               'num_emb': net.num_emb,
               'dropout_p': net.dropout_p},
              f)
    f.close()
    df_user_index.to_csv('{}/user_index.csv'.format(model_dir), index=False)
    df_item_index.to_csv('{}/item_index.csv'.format(model_dir), index=False)

    
######
# Data
######

def load_train_data(training_data_file, batch_size):
    df = pd.read_csv(training_data_file, delimiter=',', error_bad_lines=False)
    df = df[['USER_ID', 'ITEM_ID', 'RATING', '_USER_IDX', '_ITEM_IDX']]

    # MXNet data iterators
    data = gluon.data.ArrayDataset(nd.array(df['_USER_IDX'].values, dtype=np.float32), 
                                    nd.array(df['_ITEM_IDX'].values, dtype=np.float32),
                                    nd.array(df['RATING'].values, dtype=np.float32))

    data_iter = gluon.data.DataLoader(data, shuffle=True, num_workers=4, batch_size=batch_size, last_batch='rollover')

    return data_iter


def load_test_data(testing_data_file, batch_size):
    df = pd.read_csv(testing_data_file, delimiter=',', error_bad_lines=False)
    df = df[['USER_ID', 'ITEM_ID', 'RATING', '_USER_IDX', '_ITEM_IDX']]

    # MXNet data iterators
    data = gluon.data.ArrayDataset(nd.array(df['_USER_IDX'].values, dtype=np.float32), 
                                    nd.array(df['_ITEM_IDX'].values, dtype=np.float32),
                                    nd.array(df['RATING'].values, dtype=np.float32))

    data_iter = gluon.data.DataLoader(data, shuffle=True, num_workers=4, batch_size=batch_size, last_batch='rollover')

    return data_iter



def load_user_index_data(user_index_data_file):
    df = pd.read_csv(user_index_data_file, delimiter=',', error_bad_lines=False)
    
    return df


def load_item_index_data(item_index_data_file):
    df = pd.read_csv(item_index_data_file, delimiter=',', error_bad_lines=False)
    
    return df


#########
# Hosting
#########

def model_fn(model_dir):
    """
    Load the gluon model. Called once when hosting service starts.

    :param: model_dir The directory where model files are stored.
    :return: a model (in this case a Gluon network)
    """
    ctx = mx.cpu()
    f = open('{}/MFBlock.params'.format(model_dir), 'r')
    block_params = json.load(f)
    f.close()
    net = MFBlock(max_users=block_params['max_users'], 
                  max_items=block_params['max_items'],
                  num_emb=block_params['num_emb'],
                  dropout_p=block_params['dropout_p'])
    net.load_params('{}/model.params'.format(model_dir), ctx)
    df_user_index = pd.read_csv('{}/user_index.csv'.format(model_dir))
    df_item_index = pd.read_csv('{}/item_index.csv'.format(model_dir))
    return net, df_user_index, df_item_index


def transform_fn(net, data, input_content_type, output_content_type):
    """
    Transform a request using the Gluon model. Called once per request.

    :param net: The Gluon model.
    :param data: The request payload.
    :param input_content_type: The request content type.
    :param output_content_type: The (desired) response content type.
    :return: response payload and content type.
    """
    ctx = mx.cpu()
    parsed = json.loads(data)

    trained_net, df_user_index, df_item_index = net
    users = pd.DataFrame({'USER_ID': parsed['USER_ID']}).merge(df_user_index, how='left')['_USER_IDX'].values
    items = pd.DataFrame({'ITEM_ID': parsed['ITEM_ID']}).merge(df_item_index, how='left')['_ITEM_IDX'].values
    
    predictions = trained_net(nd.array(users).as_in_context(ctx), nd.array(items).as_in_context(ctx))
    response_body = json.dumps(predictions.asnumpy().tolist())

    return response_body, output_content_type


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--num-embeddings', type=int, default=100)
    parser.add_argument('--opt', type=str, default='sgd')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--wd', type=float, default=0)
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--num-gpus', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.02)

    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--test', type=str, default=os.environ['SM_CHANNEL_TEST'])
    parser.add_argument('--user-index', type=str, default=os.environ['SM_CHANNEL_USER_INDEX'])
    parser.add_argument('--item-index', type=str, default=os.environ['SM_CHANNEL_ITEM_INDEX'])
    
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    num_gpus = int(os.environ['SM_NUM_GPUS'])
    channel_input_dirs = {
        'train': args.train,
        'test': args.test,
        'user_index': args.user_index,
        'item_index': args.item_index,
    }
    
    hps = {
        'num-embeddings': args.num_embeddings, 
        'opt': args.opt, 
        'lr': args.lr, 
        'momentum': args.momentum, 
        'wd': args.wd,
        'epochs': args.epochs,
        'num-gpus': args.num_gpus,
        'batch-size': args.batch_size,
    }

    hosts = args.hosts
    current_host = args.current_host
    model_dir = args.model_dir
    
    trained_net, df_user_index, df_item_index = train(
        channel_input_dirs,
        hps,
        hosts,
        current_host,
        num_gpus,
        model_dir,
    )
    
    if current_host == hosts[0]:
        save(model_dir, trained_net, df_user_index, df_item_index)
    
