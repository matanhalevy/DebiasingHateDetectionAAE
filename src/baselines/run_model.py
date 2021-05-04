import tensorflow_model_remediation.min_diff as md
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text
from official.nlp import optimization  # to create AdamW optmizer
import tensorflow_model_analysis as tfma
import tensorflow_data_validation as tfdv
from tensorflow_model_analysis.addons.fairness.post_export_metrics import fairness_indicators
from tensorflow_model_analysis.addons.fairness.view import widget_view
import os
import logging
import random
import argparse
import json

from sklearn.utils import class_weight
# import tfrecorder #https://github.com/google/tensorflow-recorder

from utils.utils import tfidf_vectorize, logistic_regression_model, strip_punc_hp, \
    f1_from_prec_recall, compute_fairness_metrics, glove_vectorize, train_plot, \
    prepare_mindiff_ds, plot_cm, get_basic_metrics

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--is_local",
                        action='store_true',
                        help="Is it being run locally")
    parser.add_argument("--is_gcloud",
                        action='store_true',
                        help="is it run in colab or diyis server")
    parser.add_argument("--is_gridsearch",
                        action='store_true',
                        help="Is it being run in colab or locally")
    parser.add_argument("--retrain",
                        action='store_true',
                        help="If directory exists, do you want to retrain")
    parser.add_argument("--task_name",
                        default="ngram",
                        type=str,
                        required=True,
                        choices=['ngram', 'tf_idf', 'glove', 'bert', 'bert_mindiff'],
                        help="The name of the task to train.")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--dataset',
                        type=str,
                        required=True,
                        choices=['hate', 'harassment', 'davidson', 'founta', 'waseem', 'golbeck'])
    parser.add_argument('--batch_size',
                        type=int,
                        default=32,
                        help="batch size for model training")
    parser.add_argument('--epochs',
                        type=int,
                        default=1000,
                        help="epochs to train model")
    parser.add_argument('--max_seq_length',
                        type=int,
                        default=128,
                        help="max sequence length for embeddings")
    parser.add_argument('--learning_rate',
                        type=float,
                        default=2e-5,
                        help="learning_rate")
    parser.add_argument('--reg_strength',
                        type=float,
                        default=0,
                        help="L2 Regularization strength"
                        )
    parser.add_argument('--class_weight',
                        type=str,
                        default="none",
                        help="balanced, none, ten"
                        )
    parser.add_argument('--min_diff_weight',
                        type=float,
                        default=1.5,
                        required=False,
                        help="Min Diff regularization strength",)
    parser.add_argument("--output_dir",
                        type=str,
                        default=None,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    # if true, use test data instead of val data
    parser.add_argument("--test", action='store_true')

    args = parser.parse_args()

    LEARNING_RATE = args.learning_rate
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    MAX_SEQ_LENGTH = args.max_seq_length
    REG_STRENGTH = args.reg_strength
    CLASS_WEIGHT = args.class_weight
    MIN_DIFF_WEIGHT = args.min_diff_weight

    dataset_name = args.dataset.lower()
    task_name = args.task_name.lower()

    #CW_{CLASS_WEIGHT}_RS_{strip_punc_hp(REG_STRENGTH)}_
    affix = f'{dataset_name}_LR_{strip_punc_hp(LEARNING_RATE)}_BS_{str(BATCH_SIZE)}_EP_{str(EPOCHS)}_MSL_{str(MAX_SEQ_LENGTH)}'
    if task_name == 'bert_mindiff':
        affix = affix + f'_MD_{strip_punc_hp(MIN_DIFF_WEIGHT)}'


    OUTPUT_DIR = f'{args.output_dir}/{affix}' if args.is_gridsearch else f'{args.output_dir}/{dataset_name}_{args.seed}'
    # retrain if directory doesnt exist or flag to retrain is true
    should_retrain = args.retrain or not os.path.exists(f'{OUTPUT_DIR}/{task_name}') #todo revisit

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # save configs
    f = open(os.path.join(OUTPUT_DIR, 'args.json'), 'w')
    json.dump(args.__dict__, f, indent=4)
    f.close()

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    random.seed(args.seed)

    is_local = args.is_local
    is_gcloud = args.is_gcloud
    local_data_prefix = '../../data/twitter_datasets/'
    diyi_data_path_prefix = '/nethome/mhalevy3/HateSpeech/benchmarking/data/twitter_datasets/'
    gcloud_data_path_prefix = 'drive/MyDrive/HateSpeech/benchmarking/data/twitter/'
    cloud_data_path_prefix = gcloud_data_path_prefix if is_gcloud else diyi_data_path_prefix

    data_prefix = local_data_prefix if is_local else cloud_data_path_prefix

    # tuple of folder name containing the dev/test/training dataset, the label, and the text feature
    data_map = {
        'hate': ('combined_hate/', 'is_hate', 'cleaned_tweet'),
        'harassment': ('combined_harassment/', 'is_harassment', 'cleaned_tweet'),
        'davidson': ('davidson/', 'is_harassment', 'cleaned_tweet'),
        'founta': ('founta/', 'is_harassment', 'cleaned_tweet'),
        'waseem': ('waseem/', 'is_hate', 'cleaned_tweet'),
        'golbeck': ('golbeck/', 'is_harassment', 'cleaned_tweet'),
    }
    class_weight_map = {
        'balanced': 'balanced',
        'none': None,
        'ten': {0: 0.1, 1: 1},
    }
    data_folder, LABEL, TEXT_FEATURE = data_map.get(dataset_name)
    data_path = data_prefix + data_folder

    dev_pd = pd.read_csv(f'{data_path}dev.csv', index_col=None).dropna()
    train_pd = pd.read_csv(f'{data_path}train.csv', index_col=None).dropna()
    test_pd = pd.read_csv(f'{data_path}test.csv', index_col=None).dropna()

    class_weights = class_weight.compute_class_weight(class_weight_map.get(CLASS_WEIGHT, None),
                                                      np.unique(train_pd[LABEL].values),
                                                      train_pd[LABEL].values)
    class_weight_dict = dict(enumerate(class_weights))



    def eval_min_diff_bert(train_data,
                           validation_data,
                           test_data,
                           path_to_bert=f'{OUTPUT_DIR}/bert',
                           learning_rate=LEARNING_RATE,
                           batch_size=BATCH_SIZE,
                           epochs=EPOCHS,
                           min_diff_weight=MIN_DIFF_WEIGHT,
                           ):

        train_ds_main, train_ds_unpriv, train_ds_priv, (dev_examples, y_dev), (
            test_examples, y_test) = prepare_mindiff_ds(train_data=train_data,
                                                        validation_data=validation_data,
                                                        test_data=test_data,
                                                        unpriv_label='is_aae_06',
                                                        text_feature=TEXT_FEATURE,
                                                        label=LABEL,
                                                        batch_size=batch_size,
                                                        max_seq_length=MAX_SEQ_LENGTH)

        dev_ds = tf.data.Dataset.from_tensor_slices((dev_examples,
                                                     y_dev.reshape(-1, 1) * 1.0)
                                                    ).batch(batch_size)
        test_ds = tf.data.Dataset.from_tensor_slices((test_examples,
                                                      y_test.reshape(-1, 1) * 1.0)
                                                     ).batch(batch_size)
        validate_tfrecord_file = test_ds if args.test else dev_ds

        if args.do_train and should_retrain:

            bert_model = tf.keras.models.load_model(path_to_bert, compile=False)
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            bert_model.compile(optimizer='adam', loss='binary_crossentropy')

            # Create the dataset that will be passed to the MinDiffModel during training.
            dataset = md.keras.utils.input_utils.pack_min_diff_data(
                train_ds_main, train_ds_unpriv, train_ds_priv
            )

            # Wrap the original model in a MinDiffModel, passing in one of the MinDiff
            # losses and using the set loss_weight.
            min_diff_loss = md.losses.MMDLoss()
            model = md.keras.MinDiffModel(bert_model,
                                          min_diff_loss,
                                          min_diff_weight)

            # Compile the model normally after wrapping the original model.  Note that
            # this means we use the baseline's model's loss here.
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            loss = tf.keras.losses.BinaryCrossentropy()
            # early stopping if validation loss does not decrease in 2 consecutive tries.
            callbacks = [
                tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5),
                tf.keras.callbacks.TensorBoard(log_dir="logs")
            ]
            model.compile(optimizer=optimizer, loss=loss, metrics=[tf.metrics.BinaryAccuracy()])

            history = model.fit(dataset,
                                validation_data=dev_ds,
                                epochs=epochs,
                                callbacks=callbacks,
                                verbose=2,
                                batch_size=batch_size,
                                )
            model.save_original_model(f'{OUTPUT_DIR}/{task_name}', save_format='tf')

            history = history.history
            val_acc = history["val_binary_accuracy"][-1]
            val_loss = history["val_loss"][-1]
            min_diff_loss = history["min_diff_loss"][-1]
            print(f'Validation accuracy: {val_acc}, loss: {val_loss}, mindiff loss: {min_diff_loss}')
            train_plot(history, OUTPUT_DIR, task_name, acc='binary_accuracy')

        else:
            model = tf.keras.models.load_model(f'{OUTPUT_DIR}/{task_name}')

        eval_examples, eval_labels = (test_examples, y_test) if args.test else (dev_examples, y_dev)
        in_group_06 = test_data['is_aae_06'].values if args.test else validation_data['is_aae_06'].values
        in_group_08 = test_data['is_aae_08'].values if args.test else validation_data['is_aae_08'].values

        validate(model=model,
                 eval_examples=eval_examples,
                 eval_labels=eval_labels,
                 in_group_labels_06=in_group_06,
                 in_group_labels_08=in_group_08)

    def eval_bert(train_data,
                  validation_data,
                  test_data,
                  learning_rate=LEARNING_RATE,
                  batch_size=BATCH_SIZE,
                  epochs=EPOCHS,
                  ):

        train_ds = tf.data.Dataset.from_tensor_slices((train_data[TEXT_FEATURE].values,
                                                       train_data[LABEL].values.reshape(-1, 1) * 1.0)
                                                      ).batch(batch_size)

        dev_ds = tf.data.Dataset.from_tensor_slices((validation_data[TEXT_FEATURE].values,
                                                     validation_data[LABEL].values.reshape(-1, 1) * 1.0)
                                                    ).batch(batch_size)
        test_ds = tf.data.Dataset.from_tensor_slices((test_data[TEXT_FEATURE].values,
                                                      test_data[LABEL].values.reshape(-1, 1) * 1.0)
                                                     ).batch(batch_size)
        validate_tfrecord_file = test_ds if args.test else dev_ds

        if args.do_train and should_retrain:
            # note not supported in windows
            bert_preprocess_name = "bert_en_uncased_preprocess"
            bert_model_name = "bert_en_uncased_L-12_H-768_A-12"
            bert_model = hub.KerasLayer(f"https://tfhub.dev/tensorflow/{bert_model_name}/3", trainable=True)
            bert_preprocess = hub.KerasLayer(f"https://tfhub.dev/tensorflow/{bert_preprocess_name}/3")

            def build_classifier_model():
                text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
                preprocessing_layer = hub.KerasLayer(f"https://tfhub.dev/tensorflow/{bert_preprocess_name}/3",
                                                     name='preprocessing')
                encoder_inputs = preprocessing_layer(text_input)
                encoder = hub.KerasLayer(f"https://tfhub.dev/tensorflow/{bert_model_name}/3", trainable=True,
                                         name='BERT_encoder')
                outputs = encoder(encoder_inputs)
                net = outputs['pooled_output']
                net = tf.keras.layers.Dropout(0.1)(net)
                net = tf.keras.layers.Dense(1, activation=None, name='classifier')(net)
                return tf.keras.Model(text_input, net)

            model = build_classifier_model()
            steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()
            num_train_steps = steps_per_epoch * epochs
            num_warmup_steps = int(0.1 * num_train_steps)

            optimizer = optimization.create_optimizer(init_lr=learning_rate,
                                                      num_train_steps=num_train_steps,
                                                      num_warmup_steps=num_warmup_steps,
                                                      optimizer_type='adamw')

            # adamw is adam with weight decay used in bert
            model.compile(optimizer=optimizer,
                          loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                          metrics=[tf.metrics.BinaryAccuracy()])
            history = model.fit(x=train_ds,
                                validation_data=dev_ds,
                                epochs=epochs)

            # Save model.
            tf.keras.models.save_model(model, filepath=f'{OUTPUT_DIR}/{task_name}')
            # Print results.
            history = history.history
            val_acc = history["val_binary_accuracy"][-1]
            val_loss = history["val_loss"][-1]
            print(f'Validation accuracy: {val_acc}, loss: {val_loss}')
            train_plot(history, OUTPUT_DIR, task_name, acc="binary_accuracy")
        else:
            steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()
            num_train_steps = steps_per_epoch * epochs
            num_warmup_steps = int(0.1 * num_train_steps)
            model = tf.keras.models.load_model(f'{OUTPUT_DIR}/{task_name}', compile=False)
            optimizer = optimization.create_optimizer(init_lr=learning_rate,
                                                      num_train_steps=num_train_steps,
                                                      num_warmup_steps=num_warmup_steps,
                                                      optimizer_type='adamw')

            # adamw is adam with weight decay used in bert
            model.compile(optimizer=optimizer,
                          loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                          metrics=[tf.metrics.BinaryAccuracy()])

        eval_examples, eval_labels = (test_data[TEXT_FEATURE].values, test_data[LABEL].values) if args.test else (validation_data[TEXT_FEATURE].values, validation_data[LABEL].values)
        in_group_06 = test_data['is_aae_06'].values if args.test else validation_data['is_aae_06'].values
        in_group_08 = test_data['is_aae_08'].values if args.test else validation_data['is_aae_08'].values

        validate(model=model,
                 eval_examples=eval_examples,
                 eval_labels=eval_labels,
                 in_group_labels_06=in_group_06,
                 in_group_labels_08=in_group_08)

    def eval_ngram_logreg(train_data,
                          validation_data,
                          test_data,
                          learning_rate=LEARNING_RATE,
                          epochs=EPOCHS,
                          batch_size=BATCH_SIZE,
                          tf_idf=True,
                          ngram_range=(1, 2)):
        """

        :param train_data: pandas dataframe of the training data
        :param validation_data: pandas dataframe of the validation data
        :param test_data: pandas dataframe of the test data
        :param learning_rate: float, learning rate for training model.
        :param epochs: int, number of epochs.
        :param batch_size: int, number of samples per batch.
        :param tf_idf: bool, whether to encode tf-idf or n-gram
        :return:
        """
        train_texts, y_train = train_data[TEXT_FEATURE].values, train_data[LABEL].values
        dev_texts, y_dev = validation_data[TEXT_FEATURE].values, validation_data[LABEL].values
        test_texts, y_test = test_data[TEXT_FEATURE].values, test_data[LABEL].values

        x_train, x_dev, x_test = tfidf_vectorize(train_texts=train_texts,
                                                 train_labels=y_train,
                                                 val_texts=dev_texts,
                                                 test_texts=test_texts,
                                                 tf_idf=tf_idf,
                                                 ngram_range=ngram_range)
        if args.do_train and should_retrain:
            model = logistic_regression_model(x_train.shape[1],reg_strength=REG_STRENGTH)
            optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
            loss = tf.keras.losses.BinaryCrossentropy()
            model.compile(loss='bce',
                          optimizer=optimizer,
                          metrics=['acc', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(),
                                   tf.keras.metrics.AUC()]
                          )
            # early stopping if validation loss does not decrease in 2 consecutive tries.
            callbacks = [
                tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10),
                tf.keras.callbacks.TensorBoard(log_dir="logs")
            ]

            ## tensorboard todo: explore
            ## % load_ext
            ## tensorboard
            ## % tensorboard - -logdir
            ## logs
            ##
            history = model.fit(x_train,
                                y_train,
                                epochs=epochs,
                                callbacks=callbacks,
                                validation_data=(x_dev, y_dev),
                                verbose=2,  # once per epoch
                                batch_size=batch_size,
                                class_weight=class_weight_dict,
                                )

            # Print results.
            history = history.history
            val_acc = history["val_acc"][-1]
            val_loss = history["val_loss"][-1]
            print(f'Validation accuracy: {val_acc}, loss: {val_loss}')
            train_plot(history, OUTPUT_DIR, task_name)

            # Save model.
            model.save(f'{OUTPUT_DIR}/{task_name}_logreg.h5')
        else:
            model = tf.keras.models.load_model(f'{OUTPUT_DIR}/{task_name}_logreg.h5')

        eval_examples, eval_labels = (x_test, y_test) if args.test else (x_dev, y_dev)
        in_group_06 = test_data['is_aae_06'].values if args.test else validation_data['is_aae_06'].values
        in_group_08 = test_data['is_aae_08'].values if args.test else validation_data['is_aae_08'].values

        validate(model=model,
                 eval_examples=eval_examples,
                 eval_labels=eval_labels,
                 in_group_labels_06=in_group_06,
                 in_group_labels_08=in_group_08)

    def eval_glove_logreg(train_data,
                          validation_data,
                          test_data,
                          learning_rate=LEARNING_RATE,
                          epochs=EPOCHS,
                          batch_size=BATCH_SIZE):
        local_glove_prefix = '../../models/glove/'
        diyi_glove_prefix = '/nethome/mhalevy3/HateSpeech/benchmarking/models/glove/'
        gcloud_glove_path_prefix = 'drive/MyDrive/HateSpeech/benchmarking/models/glove/'
        cloud_glove_prefix = gcloud_glove_path_prefix if is_gcloud else diyi_glove_prefix

        glove_prefix = local_glove_prefix if is_local else cloud_glove_prefix
        path_to_glove_file = glove_prefix + 'glove.6B.100d.txt'

        train_texts, y_train = train_data[TEXT_FEATURE].values, train_data[LABEL].values
        dev_texts, y_dev = validation_data[TEXT_FEATURE].values, validation_data[LABEL].values
        test_texts, y_test = test_data[TEXT_FEATURE].values, test_data[LABEL].values

        x_train, x_dev, x_test, embedding_layer = glove_vectorize(train_texts=train_texts,
                                                                  val_texts=dev_texts,
                                                                  test_texts=test_texts,
                                                                  path_to_glove_file=path_to_glove_file,
                                                                  )

        if args.do_train and should_retrain:
            int_sequences_input = tf.keras.Input(shape=(None,), dtype="int64")
            # embedded_sequences = embedding_layer(int_sequences_input)
            model = logistic_regression_model(x_train.shape[1], embedding_layer)
            optimizer = tf.keras.optimizers.Adam(lr=learning_rate)

            # todo: https://github.com/keras-team/keras/issues/11749
            # try with custom metrics on output instead like f1
            model.compile(loss='bce',
                          optimizer=optimizer,
                          metrics=['acc', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(),
                                   tf.keras.metrics.AUC()]
                          )
            # early stopping if validation loss does not decrease in 10 consecutive tries.
            callbacks = [
                tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10),
                tf.keras.callbacks.TensorBoard(log_dir="logs")
            ]

            history = model.fit(x_train,
                                y_train,
                                epochs=epochs,
                                callbacks=callbacks,
                                validation_data=(x_dev, y_dev),
                                verbose=2,  # once per epoch
                                batch_size=batch_size,
                                )

            # Print results.
            history = history.history
            val_acc = history["val_acc"][-1]
            val_loss = history["val_loss"][-1]
            print(f'Validation accuracy: {val_acc}, loss: {val_loss}')
            train_plot(history, OUTPUT_DIR, task_name)

            # Save model.
            model.save(f'{OUTPUT_DIR}/{task_name}_logreg.h5')
        else:
            model = tf.keras.models.load_model(f'{OUTPUT_DIR}/{task_name}_logreg.h5')

        eval_examples, eval_labels = (x_test, y_test) if args.test else (x_dev, y_dev)
        in_group_06 = test_data['is_aae_06'].values if args.test else validation_data['is_aae_06'].values
        in_group_08 = test_data['is_aae_08'].values if args.test else validation_data['is_aae_08'].values

        validate(model=model,
                 eval_examples=eval_examples,
                 eval_labels=eval_labels,
                 in_group_labels_06=in_group_06,
                 in_group_labels_08=in_group_08)

        # how to make end to end model :
        # string_input = tf.keras.Input(shape=(1,), dtype="string")
        # x = vectorizer(string_input)
        # preds = model(x)
        # end_to_end_model = keras.Model(string_input, preds)

        # probabilities = end_to_end_model.predict(
        #     [["this message is about computer graphics and 3D modeling"]]
        # )
        #
        # class_names[np.argmax(probabilities[0])]

    def validate(model, eval_examples, eval_labels, in_group_labels_06, in_group_labels_08):
        eval = model.evaluate(eval_examples, eval_labels)
        if task_name == 'bert_mindiff' or task_name == 'bert':
            y_hat_test_pr = tf.sigmoid(model.predict(eval_examples)).numpy().flatten()
            y_hat_test = np.round(y_hat_test_pr).flatten()
            prec, recall, auc_roc = get_basic_metrics(eval_labels, y_hat_test, y_hat_test_pr)
            result = {
                'loss': eval[0],
                'acc': eval[1],
                'precision': prec,
                'recall': recall,
                'auc_roc': auc_roc,
            }
        else:
            # corresponds with metrics in model.compile
            y_hat_test = np.round(model.predict(eval_examples)).flatten()
            metrics_list = ["loss", "acc", "precision", "recall", "auc_roc"]
            result = dict(zip(metrics_list, eval))
        result['f1'] = f1_from_prec_recall(result['precision'], result['recall'])
        result = compute_fairness_metrics(metrics_dict=result,
                                          preds=y_hat_test,
                                          in_group_labels_08=in_group_labels_08,
                                          in_group_labels_06=in_group_labels_06,
                                          true_labels=eval_labels,
                                          )
        result['loss'] = eval[0]
        split = 'dev' if not args.test else 'test'

        output_eval_file = os.path.join(OUTPUT_DIR, f"eval_results_{split}_{args.task_name}")
        with open(output_eval_file, "w", encoding="utf-8") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

        plot_cm(OUTPUT_DIR, task_name, eval_labels, y_hat_test, split)

    bert_local_prefix = './'
    diyi_bert_prefix = '/nethome/mhalevy3/HateSpeech/benchmarking/baselines/'
    gcloud_bert_prefix = 'drive/MyDrive/HateSpeech/benchmarking/baselines/'
    bert_cloud_prefix = gcloud_bert_prefix if is_gcloud else diyi_bert_prefix
    bert_prefix = bert_local_prefix if is_local else bert_cloud_prefix

    bert_affix_map = {
        'davidson': 'davidson_LR_5e05_BS_64_EP_5_MSL_128',
        'founta': 'founta_LR_2e05_BS_64_EP_1_MSL_128',
        'golbeck': 'golbeck_LR_2e05_BS_64_EP_3_MSL_128',
        'harassment': 'harassment_LR_5e05_BS_32_EP_1_MSL_128',
        'hate': 'hate_LR_2e05_BS_64_EP_3_MSL_128'
    }
    bert_path = f'{bert_prefix}runs/bert_gs/{bert_affix_map.get(dataset_name)}/bert'

    params = {
        'ngram': {'train_data': train_pd, 'validation_data': dev_pd, 'test_data': test_pd, 'tf_idf': False,
                  'ngram_range': (1, 1)},
        'tf_idf': {'train_data': train_pd, 'validation_data': dev_pd, 'test_data': test_pd, 'tf_idf': True,
                   'ngram_range': (2, 2)},
        'glove': {'train_data': train_pd, 'validation_data': dev_pd, 'test_data': test_pd},
        'bert': {'train_data': train_pd, 'validation_data': dev_pd, 'test_data': test_pd},
        'bert_mindiff': {'train_data': train_pd, 'validation_data': dev_pd, 'test_data': test_pd,
                         'path_to_bert': bert_path},
    }

    kwargs = params.get(task_name, None)
    if task_name == 'ngram' or task_name == 'tf_idf':
        eval_ngram_logreg(**kwargs)
    elif task_name == 'glove':
        eval_glove_logreg(**kwargs)
    elif task_name == 'bert':
        eval_bert(**kwargs)
    elif task_name == 'bert_mindiff':
        eval_min_diff_bert(**kwargs)


if __name__ == "__main__":
    main()
