#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np
import re
import os

from nlp import Dataset


# # Load Training Data - EmpatheticDialogues

TRAIN_PATH = 'empatheticdialogues/train.csv'
VAL_PATH = 'empatheticdialogues/valid.csv'
TEST_PATH = 'empatheticdialogues/test.csv'

def load_data(path):
  '''dataframe from data path'''
  file = open(path).readlines()
  f = [line.strip().split(",")[:8] for line in file] # remove the retrieval data 
  df = pd.DataFrame(f[1:], columns=f[0])

  df['prompt'] = df['prompt'].str.replace("_comma_", ",")
  df['utterance'] = df['utterance'].str.replace("_comma_", ",")

  return df


def get_context_sent(df):
  contexts = []
  sentences = []
  for did, df_i in df.groupby('conv_id'):
    
    prompt = df_i['prompt'].values[0]
    history = [prompt]
    for sent in df_i['utterance'].tolist():
      contexts.append(' '.join(history))
      sentences.append(sent)

      # update context
      history.append(sent)

  return pd.DataFrame(zip(contexts, sentences), columns=['contexts', 'sentences'])
  

train_data = get_context_sent(load_data(TRAIN_PATH))
val_data = get_context_sent(load_data(VAL_PATH))
test_data = get_context_sent(load_data(TEST_PATH))
print(len(train_data), len(val_data), len(test_data))



# # BERT2GPT2

from transformers import BertTokenizer, GPT2Tokenizer, EncoderDecoderModel, Trainer, TrainingArguments

model = EncoderDecoderModel.from_encoder_decoder_pretrained("bert-base-uncased", "gpt2")
model.decoder.config.use_cache = False

# BERT Encoder
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_tokenizer.bos_token = bert_tokenizer.cls_token # CLS token will work as BOS token
bert_tokenizer.eos_token = bert_tokenizer.sep_token # SEP token will work as EOS token


# GPT2 Decoder
def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
  '''make sure GPT2 appends EOS in begin and end'''
  outputs = [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
  return outputs

GPT2Tokenizer.build_inputs_with_special_tokens = build_inputs_with_special_tokens
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# set pad_token_id to unk_token_id -> be careful here as unk_token_id == eos_token_id == bos_token_id
gpt2_tokenizer.pad_token = gpt2_tokenizer.unk_token

# set decoding params
model.config.decoder_start_token_id = gpt2_tokenizer.bos_token_id
model.config.eos_token_id = gpt2_tokenizer.eos_token_id
model.config.max_length = 142
model.config.min_length = 5
model.config.no_repeat_ngram_size = 3
model.early_stopping = True
model.length_penalty = 2.0
model.num_beams = 4



encoder_length = 512
decoder_length = 128
batch_size = 16

def map_to_encoder_decoder_inputs(batch):
    inputs = bert_tokenizer(batch["contexts"], padding="max_length", truncation=True, max_length=encoder_length)
    outputs = gpt2_tokenizer(batch["sentences"], padding="max_length", truncation=True, max_length=decoder_length)

    batch["input_ids"] = inputs.input_ids
    batch["attention_mask"] = inputs.attention_mask
    batch["decoder_input_ids"] = outputs.input_ids
    batch["labels"] = outputs.input_ids.copy()
    batch["decoder_attention_mask"] = outputs.attention_mask

    # complicated list comprehension here because pad_token_id alone is not good enough to know whether label should be excluded or not
    batch["labels"] = [
        [-100 if mask == 0 else token for mask, token in mask_and_tokens] for mask_and_tokens in [zip(masks, labels) for masks, labels in zip(batch["decoder_attention_mask"], batch["labels"])]
    ]

    assert all([len(x) == encoder_length for x in inputs.input_ids])
    assert all([len(x) == decoder_length for x in outputs.input_ids])

    return batch


# ## Evaluation

from datasets import load_metric
bleu_metric = load_metric('bleu')

def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    # all unnecessary tokens are removed
    pred_str = gpt2_tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = gpt2_tokenizer.eos_token_id
    label_str = gpt2_tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    score = bleu_metric.compute(predictions=pred_str, 
                              references=[[s] for s in label_str],)

    return {'bleu': score['bleu']}


# # Tokenize Data

# make train dataset ready
dataset = Dataset.from_pandas(train_data)
train_dataset = dataset.map(
    map_to_encoder_decoder_inputs, batched=True, batch_size=batch_size, remove_columns=["contexts", "sentences"],
)
train_dataset.set_format(
    type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
)

# same for validation dataset
dataset_val = Dataset.from_pandas(val_data)
val_dataset = dataset_val.map(
    map_to_encoder_decoder_inputs, batched=True, batch_size=batch_size, remove_columns=["contexts", "sentences"],
)
val_dataset.set_format(
    type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
)


# # Training

training_args = TrainingArguments(
    output_dir="./",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    # predict_from_generate=True,
    # evaluate_during_training=True,
    do_train=True,
    do_eval=True,
    logging_steps=1000,
    save_steps=1000,
    eval_steps=1000,
    overwrite_output_dir=True,
    warmup_steps=2000,
    save_total_limit=10,
    fp16=True,
)

# instantiate trainer
trainer = Trainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# start training
trainer.train()


torch.save(model)


# # Test

model.to("cuda")

def generate_summary(batch):
    # Tokenizer will automatically set [BOS] <text> [EOS]
    # cut off at BERT max length 512
    inputs = bert_tokenizer(batch["contexts"], padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    input_ids = inputs.input_ids.to("cuda")
    attention_mask = inputs.attention_mask.to("cuda")

    outputs = model.generate(input_ids, attention_mask=attention_mask)

    # all special tokens including will be removed
    output_str = gpt2_tokenizer.batch_decode(outputs, skip_special_tokens=True)

    batch["pred"] = output_str

    return batch

test_dataset = Dataset.from_pandas(test_data)
results = test_dataset.map(generate_summary, batched=True, batch_size=batch_size, remove_columns=["contexts"])

pred_str = results["pred"]
label_str = results["sentences"]

score = bleu_metric.compute(predictions=pred_str, 
                              references=[[s] for s in label_str],)
print(score['bleu'])




