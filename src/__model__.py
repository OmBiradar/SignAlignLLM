import os
from random import *
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from transformers import LlamaModel
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import Seq2SeqLMOutput

# Config helper classes


class SignRepJustEmbeddingsConfig(PretrainedConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.text_vocab_size = kwargs.get("text_vocab_size", 32000)
        self.sign_vocab_size = kwargs.get("sign_vocab_size", 60000)
        self.sign_pad_token_id = kwargs.get("sign_pad_token_id", 60001)
        self.use_gloss = kwargs.get("use_gloss", True)
        self.llama_model_path = kwargs.get("llama_model_path", "./local_llama_weights")
        self.just_embeddings = True


class SignRepConfig(PretrainedConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.text_vocab_size = kwargs.get("text_vocab_size", 32000)
        self.sign_vocab_size = kwargs.get("sign_vocab_size", 60000)
        self.sign_pad_token_id = kwargs.get("sign_pad_token_id", 60001)
        self.use_gloss = kwargs.get("use_gloss", True)
        self.llama_model_path = kwargs.get("llama_model_path", "./local_llama_weights")


# Primary model classes


class SignRepJustEmbeddings(PreTrainedModel):
    def __init__(self, config):
        super(SignRepJustEmbeddings, self).__init__(config)
        self.llama = LlamaModel.from_pretrained(config.llama_model_path)
        for param in self.llama.parameters():
            param.requires_grad = False

        self.padding_idx = config.sign_pad_token_id
        self.embedding = nn.Embedding(
            config.sign_vocab_size,
            self.llama.config.hidden_size,
            padding_idx=self.padding_idx,
        )
        nn.init.xavier_uniform_(self.embedding.weight)
        self.use_gloss = config.use_gloss

    def forward(self, skel, text, gloss):
        if self.use_gloss:
            sign_emb = self.embedding(gloss)
        else:
            sign_emb = self.embedding(skel)

        text_emb = self.llama.embed_tokens(text)
        return sign_emb, text_emb

    def get_input_embeddings(self):
        return self.embedding


class SignRep(PreTrainedModel):
    def __init__(self, config):
        super(SignRep, self).__init__(config)
        self.llama = LlamaModel.from_pretrained(config.llama_model_path)
        for param in self.llama.parameters():
            param.requires_grad = False

        self.padding_idx = config.sign_pad_token_id
        self.embedding = nn.Embedding(
            config.sign_vocab_size,
            self.llama.config.hidden_size,
            padding_idx=self.padding_idx,
        )
        nn.init.xavier_uniform_(self.embedding.weight)
        self.use_gloss = config.use_gloss

    def forward(self, skel, text, gloss):
        if self.use_gloss:
            sign_emb = self.embedding(gloss)
        else:
            sign_emb = self.embedding(skel)

        rand_layer = randint(0, self.llama.config.num_hidden_layers - 1)
        sign_rep = self.llama(
            inputs_embeds=sign_emb, output_hidden_states=True
        ).hidden_states[rand_layer]
        text_rep = self.llama(text, output_hidden_states=True).hidden_states[rand_layer]

        return sign_rep, text_rep

    def get_input_embeddings(self):
        return self.embedding


# Task-specific model classes


class SignRepJustEmbeddingsForTextGeneration(SignRepJustEmbeddings):
    config_class = SignRepJustEmbeddingsConfig

    def __init__(self, config):
        super().__init__(config)
        for param in self.llama.parameters():
            param.requires_grad = False
        for param in self.embedding.parameters():
            param.requires_grad = False

        self.lm_head = nn.Linear(
            self.llama.config.hidden_size, config.text_vocab_size, bias=False
        )
        for param in self.lm_head.parameters():
            param.requires_grad = True
        nn.init.xavier_uniform_(self.lm_head.weight)

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, **kwargs):
        text_rep, _ = super().forward(kwargs["skel"], kwargs["text"], kwargs["gloss"])
        logits = self.lm_head(text_rep)
        outputs = Seq2SeqLMOutput(logits=logits)
        return outputs

    def prepare_inputs_for_generation(
        self, input_ids, past=None, attention_mask=None, **kwargs
    ):
        return {"skel": input_ids, "text": input_ids, "gloss": input_ids}


class SignRepForTextGeneration(SignRep):
    config_class = SignRepConfig

    def __init__(self, config):
        super().__init__(config)
        for param in self.llama.parameters():
            param.requires_grad = False
        for param in self.embedding.parameters():
            param.requires_grad = False

        self.lm_head = nn.Linear(
            self.llama.config.hidden_size, config.text_vocab_size, bias=False
        )
        nn.init.xavier_uniform_(self.lm_head.weight)

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, **kwargs):
        text_rep, _ = super().forward(kwargs["skel"], kwargs["text"], kwargs["gloss"])
        logits = self.lm_head(text_rep)
        outputs = Seq2SeqLMOutput(logits=logits)
        return outputs

    def prepare_inputs_for_generation(
        self, input_ids, past=None, attention_mask=None, **kwargs
    ):
        return {"skel": input_ids, "text": input_ids, "gloss": input_ids}
