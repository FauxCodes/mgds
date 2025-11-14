from typing import List

import torch
from transformers import CLIPTokenizer, T5Tokenizer, T5TokenizerFast, GemmaTokenizer, LlamaTokenizer, Qwen2Tokenizer

from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule


class Tokenize(
    PipelineModule,
    RandomAccessPipelineModule,
):
    def __init__(
            self,
            in_name: str,
            tokens_out_name: str,
            mask_out_name: str,
            tokenizer: CLIPTokenizer | T5Tokenizer | T5TokenizerFast | GemmaTokenizer | LlamaTokenizer | Qwen2Tokenizer,
            max_token_length: int | None,
            format_text: str | None = None,
            additional_format_text_tokens: int | None = None,
            expand_mask: int = 0,
            expand_clip: bool = False,
    ):
        super(Tokenize, self).__init__()
        self.in_name = in_name
        self.tokens_out_name = tokens_out_name
        self.mask_out_name = mask_out_name
        self.tokenizer = tokenizer
        self.max_token_length = max_token_length
        self.format_text = format_text
        self.additional_format_text_tokens = additional_format_text_tokens
        self.expand_mask = expand_mask
        self.expand_clip = expand_clip

    def length(self) -> int:
        return self._get_previous_length(self.in_name)

    def get_inputs(self) -> list[str]:
        return [self.in_name]

    def get_outputs(self) -> list[str]:
        return [self.tokens_out_name, self.mask_out_name]

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        text = self._get_previous_item(variation, self.in_name, index)

        if self.expand_clip:
            tokenizer_output = tokenize_chunked(text, self.tokenizer)
        elif self.format_text is not None:
            text = self.format_text.format(text)
            tokenizer_output = self.tokenizer(
                text,
                padding='max_length',
                truncation=True,
                max_length=self.max_token_length + self.additional_format_text_tokens,
                return_tensors="pt",
            )
        else:
            tokenizer_output = self.tokenizer(
                text,
                padding='max_length',
                truncation=True,
                max_length=self.max_token_length,
                return_tensors="pt",
            )

        tokens = tokenizer_output.input_ids.to(self.pipeline.device)
        mask = tokenizer_output.attention_mask.to(self.pipeline.device)

        tokens = tokens.squeeze(dim=0)
        mask = mask.squeeze(dim=0)

        #unmask n tokens:
        if self.expand_mask > 0:
            masked_idx = (mask == 0).nonzero(as_tuple=True)[0]
            mask[masked_idx[:self.expand_mask]] = 1 #dtype is long

        return {
            self.tokens_out_name: tokens,
            self.mask_out_name: mask,
        }

def tokenize_chunked(text: str, tokenizer: CLIPTokenizer, chunk_size: int = 75) -> dict:
    chunks = _chunk_prompt(text, tokenizer, chunk_size)

    tokenized_chunks = []
    for chunk in chunks:
        tokenized = tokenizer(
            chunk,
            max_length=chunk_size + 2,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        tokenized_chunks.append(tokenized)

    return {
        'input_ids': torch.cat([chunk['input_ids'] for chunk in tokenized_chunks], dim=0),
        'attention_mask': torch.cat([chunk['attention_mask'] for chunk in tokenized_chunks], dim=0),
        'num_chunks': len(tokenized_chunks)
    }

def _chunk_prompt(text: str, tokenizer: CLIPTokenizer, chunk_size: int = 75) -> List[str]:
    tokens = tokenizer.encode(text)
    content_tokens = tokens[1:-1] if tokens[0] == tokenizer.bos_token_id else tokens

    chunks = []
    curr_chunk = []

    for token in content_tokens:
        if len(curr_chunk) < chunk_size:
            curr_chunk.append(token)
        else:
            chunk_text = tokenizer.decode(curr_chunk)
            if ', ' in chunk_text:
                parts = chunk_text.rsplit(', ', 1)
                if len(parts) == 2:
                    first_part, remaining = parts
                    if first_part:
                        chunks.append(first_part + ',')
                    curr_chunk = tokenizer.encode(remaining)[1:-1]
                    curr_chunk.append(token)
                    continue

            chunks.append(chunk_text)
            curr_chunk = [token]

    if curr_chunk:
        chunks.append(tokenizer.decode(curr_chunk))

    return chunks