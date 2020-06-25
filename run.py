import tensorflow as tf
import os
import pandas as pd
from dotenv import load_dotenv, find_dotenv
import argparse
import model_design
import preprocess
import predict_matrix
from datetime import datetime
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer
from sklearn.metrics import classification_report, confusion_matrix
load_dotenv(find_dotenv())
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--predict_test', action='store_true')
    parser.add_argument('--predict_production', action='store_true')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--skip_le', action='store_true')
    parser.add_argument('--resample', default=None, type=int)
    parser.add_argument('--max_seq_len', default=512, type=int)
    parser.add_argument('--load', type=str)
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--batch', default=32, type=int)
    parser.add_argument('--model_save_dir', default='Saved_Models', type=str)
    parser.add_argument('--model_load_dir', default='Saved_Models', type=str)
    parser.add_argument('--model_save_prefix', default='', type=str)
    parser.add_argument('--model_load_name', default='', type=str)
    parser.add_argument('--infile',
                        default='s3://sn-classification/Tenders/Training_Data/cpv_tenders_10k_min10wrds.csv')
    parser.add_argument('--type', default='Tenders', type=str, help="Spend or Tenders")
    parser.add_argument('--min_word_count', default=2, type=int)
    parser.add_argument('--minrowsperlabel', default=None, type=int)
    parser.add_argument('--resamplereplace', action='store_true', help='modifies replace param in resample - use if want to oversample small groups')
    args = parser.parse_args()
    return args


def main(args):
    print(tf.__version__)
    df = preprocess.import_training_data(args)
    print("Importing Data - Done")

    df = preprocess.get_top_labels(args, df)

    df['text'] = df['text'].apply(preprocess.preprocess_text)

    le = LabelBinarizer()
    le.fit(df['label'])
    train, test, val = preprocess.split_data(df)
    y_train = le.transform(train['label'])
    y_test = le.transform(test['label'])
    y_val = le.transform(val['label'])

    tokenizer = preprocess.initialise_bert_tokenizer(args)

    n_classes = len(df.label.unique())

    if args.train:
        x_train, x_test, x_val = preprocess.get_model_inputs(train, tokenizer, args, test, val)

        train_dataset = preprocess.reshape_inputs(x_train, y_train, args)
        test_dataset = preprocess.reshape_inputs(x_test, y_test, args)
        val_dataset = preprocess.reshape_inputs(x_val, y_val, args)

        print(tf.config.experimental.list_physical_devices('GPU'))

        compiled_model = model_design.compile(args, n_classes)
        model = model_design.fit(compiled_model, args, n_classes, train_dataset, val_dataset)

    if args.evaluate:
        if not model:
            model = model_design.compile(args, n_classes)
            model_design.load_model(args, model)
            print("Loading model - Done")
            x_train, x_test, x_val = preprocess.get_model_inputs(train, tokenizer, args, test, val)
            test_dataset = preprocess.reshape_inputs(x_test, y_test, args)

        model_design.evaluate(model, test_dataset)

    if args.predict_test:
        try:
            model
        except NameError:
            x_train, x_test, x_val = preprocess.get_model_inputs(train, tokenizer, args, test, val)
            test_dataset = preprocess.reshape_inputs(x_test, y_test, args)
            model = model_design.compile(args, n_classes)
            model_design.load_model(args, model)
        test_sample = [test_dataset[0][:100], test_dataset[1][:100], test_dataset[2][:100]]
        predictions = model.predict(test_sample[:2])
        inv_preds = le.inverse_transform(predictions)
        true_inv = le.inverse_transform(test_sample[2])
        cm = confusion_matrix(true_inv, inv_preds)
        print(cm)
        print(predict_matrix.plot_confusion_matrix(cm))


if __name__ == '__main__':
    args = get_args()
    if not args.train:
        assert args.model_load_name, "Not in train mode, must specify model to load via --model_load_name"
    main(args)