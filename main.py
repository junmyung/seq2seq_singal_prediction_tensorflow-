from data_utils import Dataset_Loader, Input_producer
from trainer_evaluator import train, evaluate
import tensorflow as tf

def get_config(FLAGS):
  if FLAGS.config=="fixed":
    fixed_config = {'input_length'      : 40,
                    'output_length'     : 1,
                    'init_scale'        : 0.1,
                    'learning_rate'     : 0.001,
                    'num_layers'        : 2,
                    'hidden_size'       : 200,
                    'decay_steps'       : 500,
                    'epoch_size'        : 10000,
                    'keep_prob'         : 1.0,
                    'lr_decay'          : 0.5,
                    'batch_size'        : 512,
                    'cell_type'         : 'lstm',
                    'optimizer'         : 'adam',
                    'use_residual'      : True,
                    'use_dropout'       : True,
                    'max_gradient_norm' : 5.0,
                    'add_noise'          : True,
                    'attention_type'    : 'bahdanau',
                    'is_attention'      : True,
                    'attn_input_feeding': True,
                    }
    return fixed_config
  elif FLAGS.config=="flags":
    return FLAGS.flag_values_dict()
  else:
    raise ValueError("Invalid model: %s", FLAGS.model)

def main():
  flags = tf.flags
  flags.DEFINE_string("config", "flags", "A type of model. Possible options are: fixed, flags,")
  # flags.DEFINE_string("data_path", "/home/jimmy/PycharmProjects/tutorial/data/household_power_consumption.pickle",
  #                     "Where the training/test data is stored.[household_power_consumption.pickle, PRSA_data_2010.1.1-2014.12.31.pickle]")
  flags.DEFINE_string("save_path", "./checkpoints/power_seq2seq",
                      "Model output directory.")  # power_seq2seq PRSA_seq2seq2
  flags.DEFINE_float("train_test_ratio", 0.6, "")
  flags.DEFINE_integer("input_length", 40, "")
  flags.DEFINE_integer("output_length", 5, "")
  flags.DEFINE_float("init_scale", 0.01, "")
  flags.DEFINE_float("learning_rate", 0.001, "")
  flags.DEFINE_float("lr_decay", 0.5, "")
  flags.DEFINE_integer("num_layers", 2, "")
  flags.DEFINE_integer("hidden_size", 200, "")
  flags.DEFINE_integer("decay_steps", 500, "")
  flags.DEFINE_integer("epoch_size", 100, "")
  flags.DEFINE_integer("batch_size", 128, "")
  flags.DEFINE_integer("eval_batch_size", 10, "")
  flags.DEFINE_string("cell_type", "lstm", "['gru','lstm']")
  flags.DEFINE_string("optimizer", "adam", "['adam','adagrad','RMS..']")
  flags.DEFINE_string("attention_type", "bahdanau", "['bahdanau','luong']")
  flags.DEFINE_boolean("use_residual", True, "")
  flags.DEFINE_boolean("is_attention", False, "")
  flags.DEFINE_boolean("attn_input_feeding", True, "")
  flags.DEFINE_boolean("use_dropout", True, "")
  flags.DEFINE_float("max_gradient_norm", 5.0, "")
  flags.DEFINE_boolean("add_noise", False, '')
  flags.DEFINE_boolean("save_plot", True, '')
  flags.DEFINE_boolean("eval_only", False, '')
  FLAGS = flags.FLAGS

  data_loader = Dataset_Loader(data_name='power', stock_name=None)

  data_loader.resample(resample_factor='D')
  df_keys = data_loader.df.keys()
  print("feature names: {}".format(df_keys))
  if FLAGS.save_plot and not FLAGS.eval_only:
    data_loader.plot_samples_sum_mean(df_keys[0])
    data_loader.plot_samples_bar(df_keys[0], resample_factor='M')
    data_loader.plot_samples_bar(df_keys[0], resample_factor='Q')
    data_loader.plot_samples_all(resample_factor='D')
    for i in range(len(df_keys)):
      data_loader.plot_sample_corr(df_keys[i],df_keys[0])
    data_loader.mat_corr()

  scaled_value = data_loader.normalization()  # data_loader.scaler.inverse_transform(scaled)
  print(scaled_value.shape)
  # split train-test dataset
  train_size = int(len(scaled_value)*FLAGS.train_test_ratio)
  train_dataset = scaled_value[:train_size, :]
  test_dataset = scaled_value[train_size:, :]
  del scaled_value
  print("train_dataset shape is {}".format(train_dataset.shape))
  print("test_dataset shape is {}".format(test_dataset.shape))

  # Define Config
  config = get_config(FLAGS)
  config['input_depth'] = train_dataset.shape[-1]
  eval_config = get_config(FLAGS)
  eval_config['input_depth'] = test_dataset.shape[-1]
  if FLAGS.eval_only:
    eval_config['batch_size'] = FLAGS.eval_batch_size
  # Define Input Generator
  train_input_generator = Input_producer(config, train_dataset)
  test_input_generator = Input_producer(eval_config, test_dataset, training=False)
  config['decay_steps'] = train_input_generator.max_iter*config['decay_steps']

  with tf.Graph().as_default():
    if not FLAGS.eval_only:
      train(FLAGS=FLAGS,
            config=config,
            train_input_generator=train_input_generator,
            test_input_generator=test_input_generator,
            graph_show='fix',
            scaler=data_loader.scaler)
    else:
      evaluate(FLAGS=FLAGS,
               config=eval_config,
               test_input_generator=test_input_generator,
               graph_show='move',
               scaler=data_loader.scaler)


if __name__=='__main__':
  main()