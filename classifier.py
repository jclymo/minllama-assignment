
import torch
import torch.nn.functional as F

# change it with respect to the original model
from config import LlamaConfig
from llama import load_pretrained
from tokenizer import Tokenizer

class LlamaZeroShotClassifier(torch.nn.Module):
	def __init__(self, config: LlamaConfig, tokenizer: Tokenizer, label_names: list[str]):
		super(LlamaZeroShotClassifier, self).__init__()
		self.num_labels = config.num_labels
		self.llama = load_pretrained(config.pretrained_model_path)
		# Zero-shot classification does not require updating llama paramters.
		for param in self.llama.parameters():
			param.requires_grad = False
		assert len(label_names) == self.num_labels
		self.tokenizer = tokenizer
		self.label_name_ids = [tokenizer.encode(label, bos=False, eos=False) for label in label_names]


	def forward(self, input_ids):
		# compute the completion probability of each label string
		logits, _ = self.llama(input_ids)
		log_probabilities = F.log_softmax(logits, dim=-1)
		label_probabilities = torch.zeros((log_probabilities.shape[0], self.num_labels), device=log_probabilities.device)
		for i, label_token_ids in enumerate(self.label_name_ids):
			total_log_prob = torch.sum(log_probabilities[:, :, label_token_ids], axis=-1)
			label_probabilities[:, i] = total_log_prob[:, 0]
		return label_probabilities

class LlamaEmbeddingClassifier(torch.nn.Module):
	def __init__(self, config):
		super(LlamaEmbeddingClassifier, self).__init__()
		self.num_labels = config.num_labels
		self.pad_id = config.pad_id
		self.llama = load_pretrained(config.pretrained_model_path)
		# If we use pretrain mode, we freeze Llama parameters.
		for param in self.llama.parameters():
			if config.option == 'pretrain':
				param.requires_grad = False
			elif config.option == 'finetune':
				param.requires_grad = True

		self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
		self.classifier_head = torch.nn.Linear(self.llama.config.dim, self.num_labels)

	def forward(self, input_ids):
		'''
		1) Find the hidden state after the final token of the input sequence
		2) Apply dropout (self.dropout) to the hidden state at training time to mitigate
		   overfitting.
		3) Pass this through the classifier head (self.classifier_head), which will return
		   logits (unnormalized probabilities) over all classes.
		4) Take the log-softmax of the logits and return log-probabilities over all classes.
		'''
		# input_ids has shape (bs, seqlen)
		logits, hidden_states = self.llama(input_ids)
		
		# last_tok = (input_ids != self.pad_id).sum(dim=1) - 2
		# # final_hidden_states = hidden_states[:, -1, :]
		# final_hidden_states = hidden_states[torch.arange(hidden_states.shape[0]), last_tok, :]
		# final_hidden_states = self.dropout(final_hidden_states)
		# logits = self.classifier_head(final_hidden_states)

		# mask = (input_ids != self.pad_id).float()
		# expanded_mask = mask.unsqueeze(-1).expand_as(hidden_states)
		# masked_hidden_states = hidden_states.clone()
		# masked_hidden_states[expanded_mask == 0] = -1e9
		# pooled_hidden_states = torch.max(masked_hidden_states, dim=1)[0]
		# pooled_hidden_states = self.dropout(pooled_hidden_states)
		# logits = self.classifier_head(pooled_hidden_states)

		mask = (input_ids != self.pad_id).float()
		expanded_mask = mask.unsqueeze(-1).expand_as(hidden_states)
		masked_hidden_states = hidden_states * expanded_mask
		pooled_hidden_states = torch.sum(masked_hidden_states, dim=1)
		token_counts = torch.sum(mask, dim=1, keepdim=True) + 1e-9
		pooled_hidden_states = pooled_hidden_states / token_counts
		pooled_hidden_states = self.dropout(pooled_hidden_states)
		logits = self.classifier_head(pooled_hidden_states)

		# mask = (input_ids != self.pad_id).float()
		# expanded_mask = mask.unsqueeze(-1).expand_as(hidden_states)
		# masked_hidden_states = hidden_states * expanded_mask
		# pooled_hidden_states = torch.sum(masked_hidden_states, dim=1)
		# logits = self.classifier_head(pooled_hidden_states)

		log_probs = F.log_softmax(logits, dim=-1)
		return log_probs
