from smart_open import open
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np
from transformers import *
from sklearn.utils import resample

import re
import math
import random
import pickle
from sklearn.utils import class_weight
from tqdm import tqdm
from html import unescape
from imblearn.over_sampling import SMOTE
from collections import Counter
from nltk.corpus import stopwords
from nltk import word_tokenize

def import_training_data(args):
    with open(args.infile, encoding="utf-8") as f:
        df = pd.read_csv(f, error_bad_lines=False)
    # df = df.sample(frac=1)
    df = df.rename(columns={'title_description': 'text'})
    df = df.rename(columns={'cpv': 'label'})
    # Shuffle with seed (so generates same shuffle every time)
    df = df.sample(n=len(df), random_state=42)
    df = df.dropna(how='any')
    df = df.astype({'label': np.int32})
    assert not df.isnull().values.any()
    return df


def get_top_labels(df):
    df = df[df.label.isin([48000000, 80000000, 85000000, 45000000, 90000000])]
    return df


def preprocess_text(sentence):
    # Convert to lowercase
    sentence = sentence.lower()
    # Remove punctuation and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)
    # Remove double spaces (left by punct removal)
    sentence = re.sub(r'\s+', ' ', sentence)

    new_stopwords = ['service', 'services', 'contract', 'contracts', 'solution', 'solutions', 'county', 'supplier',
                     'suppliers', 'district', 'council', 'borough', 'management', 'provider', 'providers',
                     'provision', 'provisions', 'projects', 'project', 'contractor', 'contractors', 'systems', 'system',
                     'proposal', 'proposals', 'requirement', 'requirements', 'support',
                     'programme', 'work', 'required', 'deliver', 'delivery', 'provide', 'provision', 'framework',
                     'agreement', 'https', 'http', 'multiquote', 'tender', 'document', 'site']
    stop_words = set(stopwords.words('english'))
    stop_words.update(new_stopwords)

    sentence = [w for w in sentence.split(" ") if not w in stop_words]
    sentence = ' '.join(w for w in sentence)

    return sentence


def split_data(df):
    # Stratifying ensures that no matter how big the dataset is we have a full representation of labels in both
    # training and test sets.
    train, test = train_test_split(df, test_size=0.30, shuffle=True, stratify=df['label'], random_state=10)
    train, val = train_test_split(train, test_size=0.1, shuffle=True, stratify=train['label'], random_state=10)
    return train, test, val


def initialise_bert_tokenizer(args):
    # Load the tokenizer.
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', add_special_tokens=True, do_lower_case=True,
                                              max_length=args.max_seq_len, pad_to_max_length=True)
    return tokenizer


def create_input_array(df, tokenizer, args):
    # https://mccormickml.com/2019/07/22/BERT-fine-tuning/
    # Tokenize all of the sentences and map the tokens to thier word IDs.
    sentences = df.text.values

    input_ids = []
    attention_masks = []
    token_type_ids = []

    # For every sentence...
    for sent in tqdm(sentences):
        # `encode_plus` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        #   (5) Pad or truncate the sentence to `max_length`
        #   (6) Create attention masks for [PAD] tokens.
        encoded_dict = tokenizer.encode_plus(
            sent,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=args.max_seq_len,  # Pad & truncate all sentences.
            pad_to_max_length=True,
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors='tf',  # Return tf tensors.
            return_token_type_ids=True,
            padding_side='right'
        )

        # Add the encoded sentence to the list.
        input_ids.append(encoded_dict['input_ids'])

        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])

        token_type_ids.append(encoded_dict['token_type_ids'])

    input_ids = tf.convert_to_tensor(input_ids)
    attention_masks = tf.convert_to_tensor(attention_masks)
    token_type_ids = tf.convert_to_tensor(token_type_ids)

    return input_ids, attention_masks, token_type_ids


def get_model_inputs(train, tokenizer, args, test=None, val=None):
    if args.train:
        train_inputs = [create_input_array(train[:], tokenizer=tokenizer, args=args)]
        test_inputs = [create_input_array(test[:], tokenizer=tokenizer, args=args)]
        val_inputs = [create_input_array(val[:], tokenizer=tokenizer, args=args)]
    if args.evaluate or args.predict_test:
        if not args.train:
            train_inputs = []
            test_inputs = [create_input_array(test[:], tokenizer=tokenizer, args=args)]
            val_inputs = []
    if args.predict_production :
        # In predict mode, train just = df but keep naming for consistency of return statement
        train_inputs = [create_input_array(train[:], tokenizer=tokenizer, args=args)]
        test_inputs = []
        val_inputs = []

    return train_inputs, test_inputs, val_inputs


def reshape_inputs(inputs,y, args):

    ids = inputs[0][0]
    masks = inputs[0][1]
    token_types = inputs[0][2]

    ids = tf.reshape(ids, (-1, args.max_seq_len))
    print("Input ids shape: ", ids.shape)
    masks = tf.reshape(masks, (-1, args.max_seq_len))
    print("Input Masks shape: ", masks.shape)
    token_types = tf.reshape(token_types, (-1, args.max_seq_len))
    print("Token type ids shape: ", token_types.shape)
    labels = tf.convert_to_tensor(y[:])

    ids=ids.numpy()
    masks = masks.numpy()
    token_types = token_types.numpy()
    labels = labels.numpy()

    return [ids, masks, token_types, labels]
