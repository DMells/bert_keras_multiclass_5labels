import tensorflow as tf
from tensorflow.keras import layers
import os
from transformers import TFBertForSequenceClassification, TFAlbertForSequenceClassification, TFAlbertModel, TFDistilBertModel, DistilBertConfig, BertConfig, TFBertModel
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.models import Model
import numpy as np
import sys
# from tqdm.keras import TqdmCallback
from tensorflow.python.keras.callbacks import TensorBoard
import smart_open
import time
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import subprocess

def compile(args, n_classes):
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        learn_rate = 4e-3

        bert = 'bert-base-uncased'

        model = TFBertModel.from_pretrained(bert, trainable=False)

        input_ids_layer = Input(shape=(args.max_seq_len,), dtype=np.int32)
        input_mask_layer = Input(shape=(args.max_seq_len,), dtype=np.int32)


        bert_layer = model([input_ids_layer, input_mask_layer])[0]

        X = tf.keras.layers.GlobalMaxPool1D()(bert_layer)

        output = Dense(n_classes)(X)

        if args.train:
            output = BatchNormalization()(output)
        else:
            output = BatchNormalization(trainable=False)(output)

        output = Activation('softmax')(output)

        model_ = Model(inputs=[input_ids_layer, input_mask_layer], outputs=output)

        optimizer = tf.keras.optimizers.Adam(learn_rate)

        loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

        metric = tf.keras.metrics.CategoricalAccuracy('accuracy')

        model_.compile(optimizer=optimizer, loss=loss, metrics=[metric])

        return model_


def fit(model_, args, n_classes, train_dataset, val_dataset):
    tensorboard = TensorBoard(
        log_dir='s3://sn-classification/Tenders/Logs/{}'.format(time.strftime("%Y%m%d-%H%M%S")),
        update_freq='epoch',
        histogram_freq=1,
        # write_grads=True,
        # write_images=True
    )

    train_x = [train_dataset[0], train_dataset[1]]
    train_y = train_dataset[3]
    val_x = [val_dataset[0], val_dataset[1]]
    val_y = val_dataset[3]

    earlystop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, min_delta=0.01, patience=6)

    rootdir = os.path.dirname(os.path.abspath(__file__))
    model_name = args.model_save_prefix + time.strftime("%Y%m%d-%H%M%S") + '-model.h5'
    savepath = os.path.join(rootdir, args.model_save_dir, args.type, model_name)

    modelcheckpoint = ModelCheckpoint(savepath, monitor='val_loss', mode='min', save_best_only=True, save_weights_only=True)
    cbs = CallBacks(val_x, val_y)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss',patience=3, factor=0.1, verbose=1, mode='min')

    model_.fit(train_x, train_y, validation_data=(val_x, val_y), batch_size=args.batch, epochs=args.epochs,
               callbacks=[cbs, reduce_lr, earlystop, modelcheckpoint])

    if sys.platform == 'linux':
        print("Transferring to gs://bert_models1/" + str(model_name))
        cmd = "gsutil -m cp " + str(savepath) + " gs://bert_models1"
        p = subprocess.Popen(cmd, shell=True)
        p.wait()

    return model_


class CallBacks(tf.keras.callbacks.Callback):
    def __init__(self, val_x, val_y):
        super(CallBacks, self).__init__()
        self.val_x = val_x
        self.val_y = val_y

    def on_epoch_end(self, epoch, logs={}):

        y_scores = self.model.predict(self.val_x)
        y_pred = tf.argmax(y_scores, axis=1)

        y_true = tf.argmax(self.val_y, axis=1)
        f1score = f1_score(y_true, y_pred, average='weighted')
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')

        sk_auc = roc_auc_score(y_true, y_scores, multi_class='ovr')

        # Logs contains all the different scores (loss, val_acc etc), so need to add our new f1 score to it.
        # Logs then gets returned back to the tuner and the max f1_score will be chosen as the best.
        logs['f1_score'] = f1score
        logs['precision'] = precision
        logs['recall'] = recall
        logs['auc'] = sk_auc


def save_model(model,tokenizer, args):
    rootdir = os.path.dirname(os.path.abspath(__file__))
    model_name = args.model_save_prefix + time.strftime("%Y%m%d-%H%M%S")+'-model.h5'
    path = os.path.join(rootdir, args.model_save_dir, args.type, model_name)
    model.save_weights(path)

    # If running on gcp instance
    if sys.platform =='linux':
        print("Transferring to gs://bert_models1/"+str(model_name))
        cmd = "gsutil -m cp "+str(path)+" gs://bert_models1"
        p = subprocess.Popen(cmd, shell=True)
        p.wait()


def load_model(args, model):
    rootdir = os.path.dirname(os.path.abspath(__file__))
    model.load_weights(os.path.join(rootdir, args.model_load_dir, args.type, args.model_load_name))


def evaluate(model, test_dataset):
    result = model.evaluate(x=test_dataset[:2], y=test_dataset[2], batch_size=20)
    print(result)
