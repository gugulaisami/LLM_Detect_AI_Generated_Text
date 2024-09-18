import sys
import gc
import traceback  # 引入 traceback 模块

import pandas as pd
import psutil
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)
from datasets import Dataset
from transformers import PreTrainedTokenizerFast

# 读取测试数据、样本提交文件和原始训练数据
test = pd.read_csv('llm-detect-ai-generated-text/test_essays.csv')
sub = pd.read_csv('llm-detect-ai-generated-text/sample_submission.csv')
org_train = pd.read_csv('llm-detect-ai-generated-text/train_essays.csv')
train = pd.read_csv("archive/train_v2_drcat_02.csv", sep=',')

# 删除重复的训练数据
train = train.drop_duplicates(subset=['text'])
train.reset_index(drop=True, inplace=True)

LOWERCASE = False
VOCAB_SIZE = 30522

# Creating Byte-Pair Encoding tokenizer
raw_tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
raw_tokenizer.normalizer = normalizers.Sequence([normalizers.NFC()] + [normalizers.Lowercase()] if LOWERCASE else [])
raw_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
trainer = trainers.BpeTrainer(vocab_size=VOCAB_SIZE, special_tokens=special_tokens)
dataset = Dataset.from_pandas(test[['text']])
def train_corp_iter():
    for i in range(0, len(dataset), 1000):
        yield dataset[i : i + 1000]["text"]
raw_tokenizer.train_from_iterator(train_corp_iter(), trainer=trainer)
tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=raw_tokenizer,
    unk_token="[UNK]",
    pad_token="[PAD]",
    cls_token="[CLS]",
    sep_token="[SEP]",
    mask_token="[MASK]",
)

# 分词化测试集文本
tokenized_texts_test = []
for text in tqdm(test['text'].tolist()):
    tokens = tokenizer.tokenize(text)
    tokenized_texts_test.append(tokens)
    # print(f"Original Text: {text}")
    # print(f"Tokenized Text: {tokens}")
    # print("-" * 50)

# 分词化训练集文本
tokenized_texts_train = []
for text in tqdm(train['text'].tolist()):
    tokens = tokenizer.tokenize(text)
    tokenized_texts_train.append(tokens)

    # print(f"Original Text: {text}")
    # print(f"Tokenized Text: {tokens}")
    # print("-" * 50)

# 分词化原始训练集文本
tokenized_texts_train_org = []
for text in tqdm(org_train['text'].tolist()):
    tokens = tokenizer.tokenize(text)
    tokenized_texts_train_org.append(tokens)
    # print(f"Original Text: {text}")
    # print(f"Tokenized Text: {tokens}")
    # print("-" * 50)

# 定义虚拟函数（在这里未被使用）
def dummy(text):
    return text
vectorizer = TfidfVectorizer(ngram_range=(3, 5), lowercase=False, sublinear_tf=True, analyzer = 'word',
    tokenizer = dummy,
    preprocessor = dummy,
    token_pattern = None, strip_accents='unicode')

vectorizer.fit(tokenized_texts_test)

# Getting vocab
vocab = vectorizer.vocabulary_

print(vocab)

vectorizer = TfidfVectorizer(ngram_range=(3, 5), lowercase=False, sublinear_tf=True,
                             analyzer='word', tokenizer=dummy, preprocessor=dummy,
                             token_pattern=None, strip_accents='unicode',
                             min_df=1, max_df=0.95)  # 调整 min_df 和 max_df 的值

# tf_train = vectorizer.fit_transform(tokenized_texts_train)
# tf_train_org = vectorizer.fit_transform(tokenized_texts_train_org)
# tf_test = vectorizer.transform(tokenized_texts_test)

# 在训练集上拟合 vectorizer
vectorizer.fit(tokenized_texts_train)

tf_train = vectorizer.transform(tokenized_texts_train)
tf_train_org = vectorizer.transform(tokenized_texts_train_org)
tf_test = vectorizer.transform(tokenized_texts_test)

# print("TF-IDF Train Sparse Matrix Info:")
# print("Sparse Matrix Shape:", tf_train.shape)
# print("Number of Non-Zero elements:", tf_train.nnz)

print("Train Labels Distribution:")
#print(train['generated'].value_counts())
print(train['label'].value_counts())

# 打印出训练集中每一列的独立值数量
print(train.nunique())

# 释放内存
del vectorizer
gc.collect()

# 获取训练集的标签
y_train = train['label'].values
y_train_org = org_train['generated'].values

# 如果测试集样本数小于等于5，直接生成空的提交文件
if len(test.text.values) < 3:
    sub.to_csv('submission.csv', index=False)
    print(len(test.text.values))
else:
    clf = MultinomialNB(alpha=0.02)
    #     clf2 = MultinomialNB(alpha=0.01)
    sgd_model = SGDClassifier(max_iter=8000, tol=1e-4, loss="modified_huber")
    p6 = {'n_iter': 2500, 'verbose': -1, 'objective': 'cross_entropy', 'metric': 'auc',
          'learning_rate': 0.05081909898961407, 'colsample_bytree': 0.726023996436955,
          'colsample_bynode': 0.5803681307354022, 'lambda_l1': 8.562963348932286,
          'lambda_l2': 4.893256185259296, 'min_data_in_leaf': 115, 'max_depth': 23, 'max_bin': 898}
    lgb = LGBMClassifier(**p6)
    cat = CatBoostClassifier(iterations=2000,
                             verbose=0,
                             l2_leaf_reg=6.6591278779517808,
                             learning_rate=0.005599066836106983,
                             subsample=0.4,
                             allow_const_label=True, loss_function='CrossEntropy')
    weights = [0.07, 0.31, 0.31, 0.31]

    # 获取系统内存信息
    memory_info = psutil.virtual_memory()

    # 打印内存信息
    print(f"Total Memory: {memory_info.total / (1024 ** 3):.2f} GB")
    print(f"Available Memory: {memory_info.available / (1024 ** 3):.2f} GB")
    print(f"Used Memory: {memory_info.used / (1024 ** 3):.2f} GB")
    print(f"Memory Usage Percentage: {memory_info.percent:.2f}%")


    def fit_ensemble(tf_train, y_train):
        clf = MultinomialNB(alpha=0.02)
        sgd_model = SGDClassifier(max_iter=8000, tol=1e-4, loss="modified_huber")
        p6 = {'n_iter': 2500, 'verbose': -1, 'objective': 'cross_entropy', 'metric': 'auc',
              'learning_rate': 0.05081909898961407, 'colsample_bytree': 0.726023996436955,
              'colsample_bynode': 0.5803681307354022, 'lambda_l1': 8.562963348932286,
              'lambda_l2': 4.893256185259296, 'min_data_in_leaf': 115, 'max_depth': 23, 'max_bin': 898}
        lgb = LGBMClassifier(**p6)
        cat = CatBoostClassifier(iterations=2000,
                                 verbose=0,
                                 l2_leaf_reg=6.6591278779517808,
                                 learning_rate=0.005599066836106983,
                                 subsample=0.4,
                                 allow_const_label=True, loss_function='CrossEntropy')
        weights = [0.07, 0.31, 0.31, 0.31]

        ensemble = VotingClassifier(estimators=[('mnb', clf),
                                                ('sgd', sgd_model),
                                                ('lgb', lgb),
                                                ('cat', cat)
                                                ],
                                    weights=weights, voting='soft', n_jobs=1)

        # try:
        #     # 将CatBoost模型的训练放在try块中
        #     ensemble.fit(tf_train, y_train)
        # except joblib.externals.loky.process_executor.TerminatedWorkerError as e:
        #     # 捕获异常并打印详细错误信息
        #     print("TerminatedWorkerError occurred:")
        #     print(e)
        #     traceback.print_exc(file=sys.stdout)

        ensemble.fit(tf_train, y_train)
        gc.collect()
        final_preds = ensemble.predict_proba(tf_test)[:, 1]
        sub['generated'] = final_preds
        sub.to_csv('submission.csv', index=False)
        sub

        # 在训练集train上计算正确率
        train_preds = ensemble.predict(tf_train)
        train_accuracy = accuracy_score(y_train, train_preds)
        print(f"Training Accuracy (train): {train_accuracy}")

        # 在训练集train_org上计算正确率
        train_preds = ensemble.predict(tf_train_org)
        train_accuracy_org = accuracy_score(y_train_org, train_preds)
        print(f"Training Accuracy (org_train): {train_accuracy_org}")

        # 在测试集上进行预测
        print("Making predictions on the test set...")
        test_preds = ensemble.predict(tf_test)

        # 生成提交文件
        print("Generating submission file...")
        submission = pd.DataFrame({'id': test['id'], 'label': test_preds})
        submission.to_csv('submission.csv', index=False)

        # 释放内存
        del ensemble
        gc.collect()

    # 调用异常处理函数
    fit_ensemble(tf_train, y_train)



