import tensorflow as tf
from seq2seq_model import Seq2seq_model
import numpy as np
import matplotlib.pyplot as plt

def train(FLAGS,
          config,
          train_input_generator,
          test_input_generator,
          graph_show='fix',
          scaler=None):
  with tf.variable_scope("Model", reuse=None, initializer=tf.initializers.variance_scaling()):
    seq2seq = Seq2seq_model(config)

  batch_inputs = (seq2seq.encoder_inputs, seq2seq.decoder_inputs, seq2seq.encoder_inputs_length,
                  seq2seq.decoder_inputs_length)

  # Session
  saver = tf.train.Saver(max_to_keep=5)
  checkpoint_saver_hook = tf.train.CheckpointSaverHook(FLAGS.save_path, save_steps=train_input_generator.max_iter,
                                                       saver=saver)
  summary_hook = tf.train.SummarySaverHook(save_secs=120, output_dir=FLAGS.save_path,
                                           scaffold=tf.train.Scaffold(summary_op=tf.summary.merge_all()))
  hooks = [checkpoint_saver_hook, summary_hook]
  print("Starting session")
  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
  with tf.train.SingularMonitoredSession(hooks=hooks, checkpoint_dir=FLAGS.save_path,
                                         config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    train_loss_epoch = []
    test_loss_epoch = []
    for epoch in xrange(config['epoch_size']):
      T_loss = []
      T_test = []
      T_labels = []
      T_outputs = []
      for step in xrange(train_input_generator.max_iter):
        feed = train_input_generator.next_batch(batch_inputs)
        feed[seq2seq.keep_prob_placeholder] = 0.5
        feed[seq2seq.feed_prev] = False
        _, outputs_, labels_, loss_ = sess.run(
          [seq2seq._train_op, seq2seq.decoder_logits_train, seq2seq.decoder_inputs, seq2seq.loss], feed)
        T_loss.append(loss_)
      for step in xrange(test_input_generator.max_iter):
        feed = test_input_generator.next_batch(batch_inputs)
        feed[seq2seq.keep_prob_placeholder] = 1.0
        feed[seq2seq.feed_prev] = True
        outputs_, labels_, loss_ = sess.run(
          [seq2seq.decoder_logits_train, seq2seq.decoder_inputs, seq2seq.loss], feed)

        outputs_ = np.array(outputs_).transpose([1, 0, 2]).reshape([-1, 1])
        labels_ = np.array(labels_).transpose([1, 0, 2]).reshape([-1, 1])

        T_test.append(loss_)
        T_outputs.append(outputs_)
        T_labels.append(labels_)
      print("Epoch:{}/{} Loss:{:.10f} TEST Loss:{:.5f}".format(epoch+1, config['epoch_size'], np.mean(T_loss),
                                                              np.mean(T_test)))
      train_loss_epoch.append(np.mean(T_loss))
      test_loss_epoch.append(np.mean(T_test))

      def plot_loss(path, epoch, train_loss, test_loss):
        axis = np.linspace(1, epoch, epoch)
        label = 'Loss'
        fig = plt.figure()
        plt.title(label)
        plt.plot(axis, np.squeeze(train_loss), 'r', label='Train_loss')
        plt.plot(axis, np.squeeze(test_loss), 'b', label='Test_loss')
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig('{}/loss.pdf'.format(path))
        plt.close(fig)

      plot_loss(FLAGS.save_path, epoch=epoch+1, train_loss=train_loss_epoch, test_loss=test_loss_epoch)

      if scaler is not None:
        T_labels = scaler.inverse_transform(
          np.repeat(np.array(T_labels).reshape([-1, 1]), config['input_depth'], axis=-1))[:, :1]
        T_outputs = scaler.inverse_transform(
          np.repeat(np.array(T_outputs).reshape([-1, 1]), config['input_depth'], axis=-1))[:, :1]
      else:
        T_labels = np.array(T_labels).reshape([-1, 1])
        T_outputs = np.array(T_outputs).reshape([-1, 1])

      if graph_show=='fix':
        plt.clf()
        T_labels = T_labels[0:1000]  # (N, 5000, 1)
        T_outputs = T_outputs[0:1000]  # (N, 5000, 1)
        plt.figure(1, figsize=(12, 5));
        plt.plot(0, 0, 'r', label='True')
        plt.plot(0, 0, 'm', label='Prediction')
        plt.legend(loc='upper left');
        plt.plot(T_labels, '-r.');
        plt.plot(T_outputs, '-m.');
        plt.draw();
        plt.pause(0.00001)

      elif graph_show=='move':
        T_labels = T_labels[:500]
        T_outputs = T_outputs[:500]
        plt.figure(1, figsize=(12, 5));
        plt.ion();  # continuously plot
        plt.plot(0, 0, 'r', label='True')
        plt.plot(0, 0, 'm', label='Prediction')
        plt.legend(loc='upper left');
        for i in range(len(T_outputs)//config['output_length']):
          # plt.xlim(i, i+eval_config['output_length'])
          out_pred = T_outputs[(i)*config['output_length']:(i+1)*config['output_length'], 0]
          out_true = T_labels[(i)*config['output_length']:(i+1)*config['output_length'], 0]
          range_out = np.arange(i, i+config['output_length'])[-config['output_length']:]
          plt.plot(range_out, out_pred, '-m.');
          plt.plot(range_out, out_true, '-r.');
          plt.draw();
          plt.pause(0.00001)
          if i%50==0:
            plt.clf()
            plt.plot(range_out[0], out_true[0], 'b', label='Input')
            plt.plot(range_out[0], out_true[0], 'r', label='True')
            plt.plot(range_out[0], out_true[0], 'm', label='Prediction')
            plt.legend(loc='upper left');


def evaluate(FLAGS,
             config,
             test_input_generator,
             graph_show='pick',
             scaler=None,
             input_and_output=True):
  # EVALUATION ONLY
  with tf.variable_scope("Model", reuse=None, initializer=tf.initializers.variance_scaling()):
    test_model = Seq2seq_model(config)

  test_batch_inputs = (
    test_model.encoder_inputs, test_model.decoder_inputs, test_model.encoder_inputs_length,
    test_model.decoder_inputs_length)

  saver = tf.train.Saver()
  with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(FLAGS.save_path)
    if ckpt and ckpt.model_checkpoint_path:
      # Restores from checkpoint
      saver.restore(sess, ckpt.model_checkpoint_path)
      T_test = []
      T_inputs = []
      T_outputs = []
      T_labels = []
      for step in xrange(test_input_generator.max_iter):
        feed = test_input_generator.next_batch(test_batch_inputs)
        feed[test_model.keep_prob_placeholder] = 1.0
        feed[test_model.feed_prev] = True
        inputs_, outputs_, labels_, loss_ = sess.run(
          [test_model.encoder_inputs, test_model.decoder_logits_train, test_model.decoder_inputs, test_model.loss],
          feed)

        inputs_ = np.reshape(inputs_, [-1, inputs_.shape[-1]])[:, :1]
        outputs_ = np.array(outputs_).transpose([1, 0, 2]).reshape([-1, 1])
        labels_ = np.array(labels_).transpose([1, 0, 2]).reshape([-1, 1])

        T_test.append(loss_)
        T_inputs.append(inputs_)
        T_outputs.append(outputs_)
        T_labels.append(labels_)
      print("Test Loss:{:.5f}".format(np.mean(T_test)))

      if scaler is not None:
        T_inputs = scaler.inverse_transform(
          np.repeat(np.array(T_inputs).reshape([-1, 1]), config['input_depth'], axis=-1))[:, :1]
        T_labels = scaler.inverse_transform(
          np.repeat(np.array(T_labels).reshape([-1, 1]), config['input_depth'], axis=-1))[:, :1]
        T_outputs = scaler.inverse_transform(
          np.repeat(np.array(T_outputs).reshape([-1, 1]), config['input_depth'], axis=-1))[:, :1]
      else:
        T_inputs = np.array(T_inputs).reshape([-1, 1])
        T_labels = np.array(T_labels).reshape([-1, 1])
        T_outputs = np.array(T_outputs).reshape([-1, 1])

      seq_length = config['input_length']+config['output_length']

      if graph_show=='fix':

        T_labels = T_labels[0:1000]  # (N, 5000, 1)
        T_outputs = T_outputs[0:1000]  # (N, 5000, 1)
        plt.figure(1, figsize=(12, 5));
        plt.clf()
        plt.plot(0, 0, 'r', label='True')
        plt.plot(0, 0, 'm', label='Prediction')
        plt.legend(loc='upper left');
        plt.plot(T_labels, '-r.');
        plt.plot(T_outputs, '-m.');
        plt.draw();
        plt.pause(0.01);
        plt.clf()
      elif graph_show=='move':
        if input_and_output:
          ##### INPUT OUTPUT #####
          plt.figure(1, figsize=(12, 5));
          plt.clf()
          plt.ion();  # continuously plot
          plt.plot(0, 0, 'b', label='Input')
          plt.plot(0, 0, 'r', label='True')
          plt.plot(0, 0, 'm', label='Prediction')
          plt.legend(loc='upper left');

          for i in range(len(T_outputs)//config['input_length']):
            plt.xlim(i*seq_length, i*seq_length+2*seq_length)
            inp_ = T_inputs[(i)*config['input_length']:(i+1)*config['input_length'], 0]
            out_pred = T_outputs[(i)*config['output_length']:(i+1)*config['output_length'], 0]
            out_true = T_labels[(i)*config['output_length']:(i+1)*config['output_length'], 0]
            range_inp = np.arange(i*seq_length, i*seq_length+seq_length)[:len(inp_)]
            range_out = np.arange(i*seq_length, i*seq_length+seq_length)[len(inp_):]
            plt.plot(range_inp, inp_, '-b.');
            plt.plot(range_out, out_true, '-r.');
            plt.plot(range_out, out_pred, '-m.');
            plt.legend(loc='upper left');
            plt.draw();
            plt.pause(0.01);
            if i%100==0:
              plt.clf()
              plt.plot(range_out[0], out_true[0], 'b', label='Input')
              plt.plot(range_out[0], out_true[0], 'r', label='True')
              plt.plot(range_out[0], out_true[0], 'm', label='Prediction')
              plt.legend(loc='upper left');

        else:
          ##### OUTPUT ONLY #####
          plt.figure(1, figsize=(12, 5));
          plt.clf()
          plt.ion();  # continuously plot
          plt.plot(0, 0, 'r', label='True')
          plt.plot(0, 0, 'm', label='Prediction')
          plt.legend(loc='upper left');
          for i in range(len(T_outputs)//config['input_length']):
            plt.xlim(i-2*seq_length, i+seq_length)
            out_pred = T_outputs[(i)*config['output_length']:(i+1)*config['output_length'], 0]
            out_true = T_labels[(i)*config['output_length']:(i+1)*config['output_length'], 0]
            range_out = np.arange(i, i+config['output_length'])[-config['output_length']:]
            plt.plot(range_out, out_pred, '-m.');
            plt.plot(range_out, out_true, '-r.');
            plt.draw();
            plt.pause(0.01)
            if i%100==0:
              plt.clf()
              plt.plot(range_out[0], out_true[0], 'b', label='Input')
              plt.plot(range_out[0], out_true[0], 'r', label='True')
              plt.plot(range_out[0], out_true[0], 'm', label='Prediction')
              plt.legend(loc='upper left');

      elif graph_show=='pick':
        config['batch_size'] = 1
        with tf.name_scope("Test"):
          with tf.variable_scope("Model", reuse=True):
            test_model = Seq2seq_model(config)

        saver = tf.train.Saver()
        with tf.Session() as sess:
          ckpt = tf.train.get_checkpoint_state(FLAGS.save_path)
          if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)
            seq_length = config['input_length']+config['output_length']
            indx = np.random.choice(test_input_generator.dataset.shape[0]-seq_length-1, seq_length)[0]
            input_pick = np.expand_dims(test_input_generator.dataset[indx:indx+config['input_length'], :], axis=0)
            output_pick = np.expand_dims(
              test_input_generator.dataset[
              indx+config['input_length']:indx+config['input_length']+config['output_length'],
              :][:, :1], axis=0)
            # output_pick = [output_pick[:, t, :] for t in xrange(config['output_length'])]

            feed = {test_model.encoder_inputs       : input_pick,
                    test_model.encoder_inputs_length: [config['input_length']],
                    test_model.decoder_inputs_length: [config['output_length']]}
            feed.update({test_model.decoder_inputs[t]: output_pick[:, t, :] for t in xrange(config['output_length'])})
            feed[test_model.keep_prob_placeholder] = 1.0
            inputs_, outputs_, labels_, loss_ = sess.run(
              [test_model.encoder_inputs, test_model.decoder_logits_train, test_model.decoder_inputs,
               test_model.loss], feed)
            inputs_ = np.reshape(inputs_, [-1, inputs_.shape[-1]])[:, :1]
            outputs_ = np.array(outputs_).transpose([1, 0, 2]).reshape([-1, 1])
            labels_ = np.array(labels_).transpose([1, 0, 2]).reshape([-1, 1])

            plt.figure(1, figsize=(12, 5));
            plt.plot(0, 0, 'b', label='Input')
            plt.plot(0, 0, 'r', label='True')
            plt.plot(0, 0, 'm', label='Prediction')
            plt.legend(loc='upper left');
            range_inp = np.arange(0, seq_length)[:len(inputs_)]
            range_out = np.arange(0, seq_length)[len(inputs_):]
            plt.plot(range_inp, inputs_, '-b.');
            plt.plot(range_out, labels_, '-r.');
            plt.plot(range_out, outputs_, '-m.');
            # plt.draw();
            # plt.pause(0.00001)
            plt.show()

