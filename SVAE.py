import os
import math
from time import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from base.BaseRecommender import BaseRecommender
from dataloader.DataBatcher import DataBatcher
from utils import Tool


class SVAE(BaseRecommender):
    def __init__(self, dataset, model_conf, device):
        super(SVAE, self).__init__(dataset, model_conf)
        self.dataset = dataset
        self.num_users = dataset.num_users
        self.num_items = dataset.num_items

        # model configs
        self.rnn_size = model_conf['rnn_size']
        self.hidden_size = model_conf['hidden_size']
        self.latent_size = model_conf['latent_size']
        # self.total_items = model_conf['total_items']
        self.item_embed_size = model_conf['item_embed_size']
        self.batch_size = model_conf['batch_size']
        self.test_batch_size = model_conf['test_batch_size']
        self.lr = model_conf['learning_rate']
        self.reg = model_conf['weight_decay']

        self.total_anneal_steps = model_conf['total_anneal_steps']
        self.anneal_cap = model_conf['anneal_cap']
        self.anneal = 0.
        self.update_count = 0

        # encoder
        self.enc_linear1 = nn.Linear(self.rnn_size, self.hidden_size)
        nn.init.xavier_normal_(self.enc_linear1.weight)

        # decoder
        self.dec_linear1 = nn.Linear(self.latent_size, self.hidden_size)
        self.dec_linear2 = nn.Linear(self.hidden_size, 1)
        nn.init.xavier_normal_(self.dec_linear1.weight)
        nn.init.xavier_normal_(self.dec_linear2.weight)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.reg)

        self.item_embed = nn.Embedding(self.num_items, self.item_embed_size)

        self.gru = nn.GRU(self.item_embed_size, self.rnn_size, batch_first=True, num_layers=1)

        self.linear1 = nn.Linear(self.hidden_size, 2 * self.latent_size)
        nn.init.xavier_normal_(self.linear1.weight)

        self.activation = nn.Tanh()  # activation for both encoder and decoder

        self.device = device

        self.to(self.device)

    def sample_latent(self, h_enc):
        """
        Return the latent normal sample z ~ N(mu, sigma^2)
        """
        temp_out = self.linear1(h_enc)

        mu = temp_out[:, :self.latent_size]
        log_sigma = temp_out[:, self.latent_size:]

        sigma = torch.exp(log_sigma)
        std_z = torch.from_numpy(np.random.normal(0, 1, size=sigma.size())).float()

        self.z_mean = mu
        self.z_log_sigma = log_sigma

        return mu + sigma * Variable(std_z.to(self.device), requires_grad=False)  # Re-parameterization trick

    def forward(self, x):
        """
        Function to do a forward pass
        :param x: the input
        """
        # print('-'*50)
        in_shape = x.shape  # [bsz x seq_len] = [1 x seq_len]
        # print('Input shape:', in_shape)
        x = x.reshape(-1)  # [seq_len]
        # print('Flatten:', x.shape)

        x = torch.tensor(x).to(self.device).long()
        x = self.item_embed(x)  # [seq_len x embed_size]
        # print('Item embedding:', x.shape)
        x = x.reshape(in_shape[0], in_shape[1], self.item_embed_size)  # [1 x seq_len x embed_size]
        # print('Reshape:', x.shape)

        rnn_out, _ = self.gru(x)  # [1 x seq_len x rnn_size]
        # print('GRU:', rnn_out.shape)
        rnn_out = rnn_out.reshape(in_shape[0] * in_shape[1], self.rnn_size)  # [seq_len x rnn_size]
        # print('Reshape', rnn_out.shape)

        enc_out = self.enc_linear1(rnn_out)  # [seq_len x hidden_size]
        # print('Encoder:', enc_out.shape)
        enc_out = self.activation(enc_out)
        # print('Activation:', enc_out.shape)
        sampled_z = self.sample_latent(enc_out)  # [seq_len x latent_size]
        # print('Sample:', sampled_z.shape)

        dec_out = self.dec_linear1(sampled_z)  # [seq_len x total_items]
        # print('Decoder1:', dec_out.shape)
        dec_out = self.activation(dec_out)
        # print('Activation:', dec_out.shape)
        dec_out = self.dec_linear2(dec_out)
        # print('Decoder2:', dec_out.shape)
        dec_out = dec_out.view(-1, self.num_items)  # [1 x seq_len x total_items]
        # print('Final', dec_out.shape)
        # print('-'*50)

        kl_loss = torch.mean(torch.sum(0.5 * (-self.z_log_sigma + torch.exp(self.z_log_sigma) + self.z_mean ** 2 - 1), -1))

        if self.training:
            return dec_out, kl_loss
        else:
            return dec_out

    def train_model(self, dataset, evaluator, early_stop, logger, config):
        exp_config = config['Experiment']

        num_epochs = exp_config['num_epochs']
        print_step = exp_config['print_step']
        test_step = exp_config['test_step']
        test_from = exp_config['test_from']
        verbose = exp_config['verbose']
        log_dir = logger.log_dir

        # prepare dataset
        # dataset.set_eval_data('valid')
        users = np.arange(self.num_users)

        train_matrix = dataset.train_matrix.toarray()
        train_matrix = torch.FloatTensor(train_matrix)

        # for epoch
        start = time()
        for epoch in range(1, num_epochs + 1):
            self.train()

            epoch_loss = 0.0
            batch_loader = DataBatcher(users, batch_size=self.batch_size, drop_remain=False, shuffle=False)
            num_batches = len(batch_loader)
            # ======================== Train
            epoch_train_start = time()
            for b, batch_idx in enumerate(batch_loader):
                batch_matrix = train_matrix[batch_idx].to(self.device)

                if self.total_anneal_steps > 0:
                    self.anneal = min(self.anneal_cap, 1. * self.update_count / self.total_anneal_steps)
                else:
                    self.anneal = self.anneal_cap

                batch_loss = self.train_model_per_batch(batch_matrix)
                epoch_loss += batch_loss

                if verbose and (b + 1) % verbose == 0:
                    print('batch %d / %d loss = %.4f' % (b + 1, num_batches, batch_loss))
            epoch_train_time = time() - epoch_train_start

            epoch_info = ['epoch=%3d' % epoch, 'loss=%.3f' % epoch_loss, 'train time=%.2f' % epoch_train_time]

            # ======================== Evaluate
            if (epoch >= test_from and epoch % test_step == 0) or epoch == num_epochs:
                self.eval()
                # evaluate model
                epoch_eval_start = time()

                test_score = evaluator.evaluate(self)
                test_score_str = ['%s=%.4f' % (k, test_score[k]) for k in test_score]

                updated, should_stop = early_stop.step(test_score, epoch)

                if should_stop:
                    logger.info('Early stop triggered.')
                    break
                else:
                    # save best parameters
                    if updated:
                        torch.save(self.state_dict(), os.path.join(log_dir, 'best_model.p'))
                        if self.anneal_cap == 1: print(self.anneal)

                epoch_eval_time = time() - epoch_eval_start
                epoch_time = epoch_train_time + epoch_eval_time

                epoch_info += ['epoch time=%.2f (%.2f + %.2f)' % (epoch_time, epoch_train_time, epoch_eval_time)]
                epoch_info += test_score_str
            else:
                epoch_info += ['epoch time=%.2f (%.2f + 0.00)' % (epoch_train_time, epoch_train_time)]

            if epoch % print_step == 0:
                logger.info(', '.join(epoch_info))

        total_train_time = time() - start

        return early_stop.best_score, total_train_time

    def train_model_per_batch(self, batch_matrix, batch_weight=None):
        # zero grad
        self.optimizer.zero_grad()

        # model forwrad
        output, kl_loss = self.forward(batch_matrix)

        # print('Output shape:', output.shape)
        # print('Output sample:', output[0])
        # print('Output sample shape:', output[0].shape)
        # print('Batch matrix shape:', batch_matrix.shape)
        # print('Batch matrix sample:', batch_matrix[0])
        # print('Batch matrix sample shape:', batch_matrix[0].shape)

        if batch_weight is None:
            ce_loss = -(F.log_softmax(output, 1) * batch_matrix).sum(1).mean()
        else:
            ce_loss = -((F.log_softmax(output, 1) * batch_matrix) * batch_weight.view(output.shape[0], -1)).sum(1).mean()

        loss = ce_loss + kl_loss * self.anneal

        # backward
        loss.backward()

        # step
        self.optimizer.step()

        self.update_count += 1

        return loss

    def predict(self, user_ids, eval_pos_matrix, eval_items=None):
        self.eval()
        # print('[SVAE.predict] user_ids:', user_ids)
        # print('[SVAE.predict] eval_pos_matrix:', eval_pos_matrix.shape)
        batch_eval_pos = eval_pos_matrix[user_ids]
        # print('[SVAE.predict] batch_eval_pos:', batch_eval_pos.shape)
        with torch.no_grad():
            eval_input = torch.Tensor(batch_eval_pos.toarray()).to(self.device)
            eval_output = self.forward(eval_input).detach().cpu().numpy()

            if eval_items is not None:
                eval_output[np.logical_not(eval_items)]=float('-inf')
            else:
                eval_output[batch_eval_pos.nonzero()] = float('-inf')
        self.train()
        return eval_output

    def restore(self, log_dir):
        with open(os.path.join(log_dir, 'best_model.p'), 'rb') as f:
            state_dict = torch.load(f)
        self.load_state_dict(state_dict)

    def user_embedding(self, input_matrix):
        with torch.no_grad():
            user_embedding = torch.zeros(self.num_users, self.hidden_size)
            users = np.arange(self.num_users)

            input_matrix = torch.FloatTensor(input_matrix.toarray())

            batch_size = self.test_batch_size
            batch_loader = DataBatcher(users, batch_size=batch_size, drop_remain=False, shuffle=False)
            print(self.device)
            for b, (batch_user_idx) in enumerate(batch_loader):
                batch_matrix = input_matrix[batch_user_idx]
                batch_matrix = torch.Tensor(batch_matrix).to(self.device)

                h = F.dropout(F.normalize(batch_matrix), p=0.1, training=self.training)
                h = nn.Linear(self.num_items, self.hidden_size).to(self.device)(h)
                h = self.activation(h)
                batch_emb = h[:, :self.hidden_size]  # mu

                user_embedding[batch_user_idx] += batch_emb.detach().cpu()

        return user_embedding.detach().cpu().numpy()

    def get_output(self, dataset):
        test_eval_pos, test_eval_target, _ = dataset.test_data()
        num_users = len(test_eval_target)
        num_items = test_eval_pos.shape[1]
        eval_users = np.arange(num_users)
        user_iterator = DataBatcher(eval_users, batch_size=128)
        output = np.zeros((num_users, num_items))
        for batch_user_ids in user_iterator:
            batch_pred = self.predict(batch_user_ids, test_eval_pos)
            output[batch_user_ids] += batch_pred
        return output
