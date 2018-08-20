import tensorflow as tf
from tensorflow.contrib.rnn import *
from tensorflow.contrib.seq2seq.python.ops import attention_wrapper

class Seq2seq_model(object):
  def __init__(self, config):
    self.config = config
    self.feed_prev = tf.placeholder(tf.bool, shape=())
    self.cell_type = config['cell_type']  ##
    self.optimizer = config['optimizer']  ##
    self.hidden_units = config['hidden_size']
    self.num_layers = config['num_layers']
    self.attention_type = config['attention_type']  ##
    self.use_residual = config['use_residual']  ##
    self.attn_input_feeding = config['attn_input_feeding']  ##
    self.use_dropout = config['use_dropout']  ##
    self.learning_rate = config['learning_rate']
    self.max_gradient_norm = config['max_gradient_norm']  ##
    self.dtype = tf.float32

    self.build_model()

  def build_model(self):
    self.init_placeholders()
    self.build_encoder()
    self.build_decoder()

  def init_placeholders(self):
    # encoder_inputs: [batch_size, max_time_steps, features]
    self.encoder_inputs = tf.placeholder(dtype=tf.float32,
                                         shape=(None, self.config['input_length'], self.config['input_depth']),
                                         name='encoder_inputs')
    self.encoder_inputs_length = tf.placeholder(dtype=tf.int32, shape=(None,), name='encoder_inputs_length')

    self.batch_size = tf.shape(self.encoder_inputs)[0]

    self.decoder_inputs = [tf.placeholder(dtype=tf.float32,
                                          shape=(None, 1),
                                          name='decoder_inputs_{}'.format(t)) for t in
                           range(self.config['output_length'])]

    self.decoder_inputs_length = tf.placeholder(dtype=tf.int32, shape=(None,), name='decoder_inputs_length')

    # insert _GO symbol in front of each decoder input -> why should _GO symbol be zero? -> last value of encoder input?
    # self.decoder_inputs_train = [tf.zeros_like(self.decoder_inputs[0], dtype=tf.float32, name="GO")]+self.decoder_inputs[:-1]

    ## warning:: self.encoder_inputs[:,-1,:1] -> output value should be located at first column ([:,:,0])
    self.decoder_inputs_train = [self.encoder_inputs[:, -1, :1]]+self.decoder_inputs[:-1]

    self.decoder_targets_train = self.decoder_inputs
    self.keep_prob_placeholder = tf.placeholder(self.dtype, shape=[], name='keep_prob')

  def build_encoder(self):
    print("building encoder..")
    with tf.variable_scope('encoder'):
      # Building encoder_cell
      self.encoder_cell = self.build_encoder_cell()
      input_layer = tf.layers.Dense(self.hidden_units, dtype=self.dtype, name='input_projection')
      self.encoder_inputs_embedded = input_layer(self.encoder_inputs)

      self.encoder_outputs, self.encoder_last_state = tf.nn.dynamic_rnn(
        cell=self.encoder_cell, inputs=self.encoder_inputs_embedded, dtype=self.dtype, time_major=False)

  def build_decoder(self):
    print("building decoder..")
    with tf.variable_scope('decoder'):
      if self.config['is_attention']:
        self.decoder_cell, self.decoder_initial_state = self.build_attention_decoder_cell()
      else:
        self.decoder_cell, self.decoder_initial_state = self.build_decoder_cell()

      input_layer = tf.layers.Dense(self.hidden_units, dtype=self.dtype, name='input_projection')

      output_layer = tf.layers.Dense(units=1, dtype=self.dtype, name='output_projection')

      def _rnn_decoder(decoder_inputs,
                       initial_state,
                       cell,
                       input_fn=input_layer,
                       output_fn=output_layer,
                       feed_previous=False,
                       scope=None):
        with tf.variable_scope(scope or "rnn_decoder"):
          state = initial_state
          outputs = []
          if feed_previous:
            next_input = None
            with tf.variable_scope("loop_function", reuse=tf.AUTO_REUSE):
              for i, inp in enumerate(decoder_inputs):
                if i>0:
                  inp = input_fn(next_input)  # [batch_size, 1] -> [batch_size, hidden_units]
                else:
                  inp = input_fn(inp)  # [batch_size, 1] -> [batch_size, hidden_units]
                output, state = cell(inp, state)  # [batch_size, hidden_units]
                output = output_fn(output)  # [batch_size, hidden_units] -> [batch_size, 1]

                outputs.append(output)
                next_input = output  # [batch_size, 1]
              return outputs, state
          else:
            with tf.variable_scope("loop_function"):
              for i, inp in enumerate(decoder_inputs):
                if i>0:
                  tf.get_variable_scope().reuse_variables()
                inp = input_fn(inp)  # [batch_size, 1] -> [batch_size, hidden_units]
                output, state = cell(inp, state)  # [batch_size, hidden_units]
                output = output_fn(output)  # [batch_size, hidden_units] -> [batch_size, 1]
                outputs.append(output)
              return outputs, state

      self.decoder_logits_train, self.decoder_final_state = tf.cond(self.feed_prev,\
                                                                    lambda :_rnn_decoder(self.decoder_inputs_train,
                                                                         self.decoder_initial_state,
                                                                         self.decoder_cell,
                                                                         input_fn=input_layer,
                                                                         output_fn=output_layer,
                                                                         feed_previous= True),
                                                                    lambda :_rnn_decoder(self.decoder_inputs_train,
                                                                         self.decoder_initial_state,
                                                                         self.decoder_cell,
                                                                         input_fn=input_layer,
                                                                         output_fn=output_layer,
                                                                         feed_previous= False))


      self.loss = tf.losses.mean_squared_error(self.decoder_logits_train, self.decoder_targets_train)

      # Training summary for the current batch_loss
      tf.summary.scalar('loss', self.loss)

      self.build_optimizer()

  def build_optimizer(self):
    self._global_step = tf.train.get_or_create_global_step()
    print("setting optimizer..")
    # Gradients and SGD update operation for training the model
    trainable_params = tf.trainable_variables()
    if self.optimizer.lower()=='adadelta':
      self.opt = tf.train.AdadeltaOptimizer(learning_rate=self.learning_rate)
    elif self.optimizer.lower()=='adam':
      self.opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
    elif self.optimizer.lower()=='rmsprop':
      self.opt = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
    else:
      self.opt = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)

    # Compute gradients of loss w.r.t. all trainable variables
    gradients = tf.gradients(self.loss, trainable_params)

    # Clip gradients by a given maximum_gradient_norm
    clip_gradients, _ = tf.clip_by_global_norm(gradients, self.max_gradient_norm)

    # Update the model
    self._train_op = self.opt.apply_gradients(
      zip(clip_gradients, trainable_params), global_step=self._global_step)

  def build_single_cell(self):
    if (self.cell_type.lower()=='gru'):
      cell_type = GRUCell
    elif (self.cell_type.lower()=='lstm'):
      cell_type = LSTMCell
    cell = cell_type(self.hidden_units)
    if self.use_dropout:
      cell = DropoutWrapper(cell, dtype=self.dtype,
                            output_keep_prob=self.keep_prob_placeholder, )
    if self.use_residual:
      cell = ResidualWrapper(cell)
    return cell

  # Building encoder cell
  def build_encoder_cell(self):
    return MultiRNNCell([self.build_single_cell() for i in xrange(self.num_layers)])

  def build_attention_decoder_cell(self):
    encoder_outputs = self.encoder_outputs
    encoder_last_state = self.encoder_last_state
    encoder_inputs_length = self.encoder_inputs_length

    self.attention_mechanism = attention_wrapper.BahdanauAttention(
      num_units=self.hidden_units, memory=encoder_outputs,
      memory_sequence_length=encoder_inputs_length, )

    if self.attention_type.lower()=='luong':
      self.attention_mechanism = attention_wrapper.LuongAttention(
        num_units=self.hidden_units, memory=encoder_outputs,
        memory_sequence_length=encoder_inputs_length, )

    # # Building decoder_cell
    self.decoder_cell_list = [self.build_single_cell() for i in range(self.num_layers)]

    def attn_decoder_input_fn(inputs, attention):
      if not self.attn_input_feeding:
        return inputs

      # Essential when use_residual=True
      _input_layer = tf.layers.dense(tf.concat([inputs, attention], axis=-1), self.hidden_units,
                                     name='attn_input_feeding')
      return _input_layer

    self.decoder_cell_list[-1] = attention_wrapper.AttentionWrapper(
      cell=self.decoder_cell_list[-1],
      attention_mechanism=self.attention_mechanism,
      attention_layer_size=self.hidden_units,
      cell_input_fn=attn_decoder_input_fn,
      initial_cell_state=encoder_last_state[-1],
      alignment_history=False,
      name='Attention_Wrapper'
      )
    batch_size = self.config['batch_size']
    initial_state = [state for state in encoder_last_state]

    initial_state[-1] = self.decoder_cell_list[-1].zero_state(
      batch_size=batch_size, dtype=self.dtype)
    decoder_initial_state = tuple(initial_state)
    return MultiRNNCell(self.decoder_cell_list), decoder_initial_state

  def build_decoder_cell(self):
    encoder_last_state = self.encoder_last_state
    # # Building decoder_cell
    self.decoder_cell_list = [self.build_single_cell() for i in range(self.num_layers)]

    batch_size = self.config['batch_size']
    initial_state = [state for state in encoder_last_state]

    initial_state[-1] = self.decoder_cell_list[-1].zero_state(
      batch_size=batch_size, dtype=self.dtype)
    decoder_initial_state = tuple(initial_state)
    return MultiRNNCell(self.decoder_cell_list), decoder_initial_state
