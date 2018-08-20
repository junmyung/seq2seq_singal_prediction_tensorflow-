import pandas as pd
import numpy as np
import os
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from dateutil import parser
import matplotlib.pyplot as plt
import seaborn as sns

class Dataset_Loader(object):
  def __init__(self,data_name,stock_name=None):
    self.data_name = data_name
    self.figure_num = 0
    self.load_dataset(stock_name)
    self.data_returns = []
    self.remove_null_data()
    self.dir_path = "./plot/{}".format(self.data_name)
    if not tf.gfile.Exists(self.dir_path):
      tf.gfile.MakeDirs(self.dir_path)

  def load_dataset(self,stock_name=None):
    if self.data_name == 'power':
      path = './data/household_power_consumption.txt'
      pickle_path = path.replace('txt', 'pickle')
      if os.path.exists(pickle_path):
        self.df = pd.read_pickle(pickle_path)
      else:
        self.df = pd.read_csv(path, sep=';',
                         parse_dates={'dt': ['Date', 'Time']}, infer_datetime_format=True,
                         low_memory=False, na_values=['nan', '?'], index_col='dt')
        self.df.to_pickle(pickle_path)

    elif self.data_name.lower() =='prsa':
      path = './data/PRSA_data_2010.1.1-2014.12.31.csv'
      pickle_path = path.replace('csv', 'pickle')
      if os.path.exists(pickle_path):
        self.df = pd.read_pickle(pickle_path)
      else:
        self.df = pd.read_csv(path,
                              parse_dates={'Datetime':['year', 'month', 'day', 'hour']}, date_parser=lambda *columns: datetime(*map(int,columns)))
        self.df = self.df.set_index(['Datetime'])
        self.df = self.df.drop(['No'],axis=1) ## = del df['No']
        self.df.to_pickle(pickle_path)
      self.df = self.df.rename(columns={'pm2.5':'pm2_5'})

    elif self.data_name.lower() =='stock':
      if stock_name is None:
        stock_name = "_SP500"
      path = "./data/stock_data/{}.csv".format(stock_name)
      pickle_path = "./data/stock_{}.pickle".format(stock_name)
      if os.path.exists(pickle_path):
        self.df = pd.read_pickle(pickle_path)
      else:
        info = pd.read_csv("./data/stock_data/constituents-financials.csv")
        info = info.rename(columns={col: col.lower().replace(' ', '_') for col in info.columns})
        info['file_exists'] = info['symbol'].map(lambda x: os.path.exists("data/stock_data/{}.csv".format(x)))
        info = info[info['file_exists']==True].reset_index(drop=True)
        stock_list = info['symbol'].tolist()
        print("Choose one of {}".format(stock_list))
        print("{} is selected".format(stock_name))
        self.df = pd.read_csv(path, na_values=['nan', '?'], index_col='Date',
                              date_parser=lambda *columns:pd.to_datetime(parser.parse(*map(str,columns))))
        columnstitles = self.df.columns.tolist()
        self.df = self.df.reindex(columns=columnstitles[3:4]+columnstitles[:3]+columnstitles[4:])
        self.df.to_pickle(pickle_path)

  def remove_null_data(self):
    ## finding all columns that have nan:
    droping_list_all = []
    if self.data_name=='prsa':

      temp = pd.get_dummies(self.df['cbwd'], prefix='cbwd')
      self.df = pd.concat([self.df, temp], axis=1)
      del self.df['cbwd'], temp

    for j in range(0, len(self.df.keys())):
      if not self.df.iloc[:, j].notnull().all():
        droping_list_all.append(j)

    print("column list {} have Nan values. Replace with median values".format(droping_list_all))
    # filling nan with median in any columns
    for j in range(0, len(self.df.keys())):
      # self.df.iloc[:, j] = self.df.iloc[:, j].fillna(self.df.iloc[:, j].median())
      self.df.iloc[:, j] = pd.to_numeric(self.df.iloc[:, j], errors='coerce')
      self.df.iloc[:, j] = self.df.iloc[:, j].fillna(self.df.iloc[:, j].median())

    assert(self.df.isnull().sum().sum() == False), "Remove Nan data"


  def resample(self,resample_factor='h'):
    self.df = self.df.resample(resample_factor).mean()
    self.df = self.df.dropna(how='all')


  def normalization(self):
    values = self.df.values
    values = values.astype('float32')
    self.scaler = MinMaxScaler(feature_range=(0,1))
    self.scaled_value = self.scaler.fit_transform(values)
    return self.scaled_value
    # self.scaler.inverse_transform(scaled)

  def plot_samples_sum_mean(self, dataframe, resample_factor='D'):

    # resample data with summation by resample factor and plotting
    plt.figure(self.figure_num)

    self.df[dataframe].resample(resample_factor).sum().plot(title='{} resampled over day for sum'.format(dataframe))
    plt.savefig(os.path.join(self.dir_path,"{}_resample_{}_sum".format(dataframe, resample_factor)))
    plt.figure(self.figure_num)

    self.df[dataframe].resample(resample_factor).mean().plot(title='{} resampled over day'.format(dataframe), color='red')
    plt.savefig(os.path.join(self.dir_path,"{}_resample_{}".format(dataframe, resample_factor)))
    plt.figure(self.figure_num)

    self.df[dataframe].resample(resample_factor).mean().plot(title='{} resampled over day for mean'.format(dataframe),
                                                     color='red')
    plt.tight_layout()
    plt.savefig(os.path.join(self.dir_path,"{}_resample_{}_mean".format(dataframe,resample_factor)))
    plt.close('all')
  def plot_samples_bar(self, dataframe, resample_factor='M'):
    plt.figure(self.figure_num)

    self.df[dataframe].resample(resample_factor).mean().plot(kind='bar')
    plt.xticks(rotation=60)
    plt.ylabel('{}'.format(dataframe))
    plt.title('{} averaged over {})'.format(dataframe,resample_factor))
    plt.savefig(os.path.join(self.dir_path,"{}_resample_{}_mean_bar".format(dataframe, resample_factor)))
    plt.close('all')
  def plot_samples_all(self, resample_factor='D'):
    plt.figure(self.figure_num)
    i = 1
    values = self.df.resample(resample_factor).mean().values
    # plot each column
    plt.figure(figsize=(15, 10))
    for group in range(len(self.df.keys())):
      plt.subplot(len(self.df.keys()), 1, i)
      plt.plot(values[:, group])
      plt.title(self.df.columns[group], y=0.75, loc='right')
      i += 1
    plt.savefig(os.path.join(self.dir_path,"Features_resample_{}_mean".format(resample_factor)), bbox_inches='tight')
    plt.close('all')
  def plot_sample_corr(self,df1,df2):
    if not len(self.data_returns):
      self.data_returns = self.df.pct_change()
    try:
      plt.figure(figsize=(15, 10))
      sns.jointplot(x=df1, y=df2, data=self.data_returns)
      plt.title("{} and {} Corr".format(df1, df2), y=0.01, loc='right')
      plt.savefig(os.path.join(self.dir_path,"{}and{}_correlation".format(df1, df2)))
      plt.close('all')
    except:
      pass

  def mat_corr(self, resample = None):
    # Correlations among columns
    if resample:
      plt.matshow(self.df.resample(resample).mean().corr(method='spearman'), vmax=1, vmin=-1, cmap='PRGn')
      plt.title('Mat Corr resampled over {}'.format(resample), size=10)
    else:
      plt.matshow(self.df.corr(method='spearman'), vmax=1, vmin=-1, cmap='PRGn')
      plt.title('Mat Corr wo resample', size=10)
    plt.colorbar()
    plt.savefig(os.path.join(self.dir_path,"Features_mat_correlation"))
    plt.close('all')
class Input_producer():
  def __init__(self, config, dataset, training=True):
    self.start = 0
    self.config = config
    self.dataset = dataset
    self.batch_size = config['batch_size']
    self.input_length = config['input_length']
    self.output_length = config['output_length']
    self.add_noise = config['add_noise']
    self.max_time_steps = self.input_length+self.output_length

    self.max_iter = dataset.shape[0]-self.max_time_steps
    self.range = np.arange(self.max_iter)
    range_length = int(len(self.range)//self.batch_size*self.batch_size)
    self.range = self.range[:range_length]
    self.max_iter = len(self.range)//self.batch_size
    self.training = training

    if self.training:
      np.random.shuffle(self.range)

  def next_batch(self, placeholder):
    if self.start>=self.max_iter:
      self.start = 0
    pick_batch_list = self.range[self.start*self.batch_size:(self.start+1)*self.batch_size]
    batch_idxs = [list(range(i, i+self.max_time_steps)) for i in pick_batch_list]
    seq = np.take(self.dataset, batch_idxs, axis=0)
    batch_X = seq[:, :-(self.output_length), :]  # total_size X max_time_lenghs X depth
    batch_Y = seq[:, -(self.output_length):, :1]  # total_size X output_size
    if self.training and self.add_noise:
      noise = np.random.normal(0, 0.001, batch_X.shape)
      batch_X = batch_X+noise

    self.start += 1

    feed_dict = {placeholder[0]: batch_X,
                 placeholder[2]: [batch_X.shape[1]]*self.batch_size,
                 placeholder[3]: [batch_Y.shape[1]]*self.batch_size,
                 }
    feed_dict.update({placeholder[1][t]: batch_Y[:, t, :] for t in xrange(self.config['output_length'])})
    return feed_dict


