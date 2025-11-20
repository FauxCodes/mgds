from contextlib import nullcontext
from typing import List

import torch
from torch import Tensor
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule


class EncodeClipText(
    PipelineModule,
    RandomAccessPipelineModule,
):
    def __init__(
            self,
            in_name: str,
            tokens_attention_mask_in_name: str | None,
            hidden_state_out_name: str,
            pooled_out_name: str | None,
            text_encoder: CLIPTextModel | CLIPTextModelWithProjection,
            tokenizer: CLIPTokenizer,
            add_layer_norm: bool,
            hidden_state_output_index: int | None = None,
            autocast_contexts: list[torch.autocast | None] = None,
            dtype: torch.dtype | None = None,
            expand_token_limit: bool = False,
            expanded_chunk_size: int = 0
    ):
        super(EncodeClipText, self).__init__()
        self.in_name = in_name
        self.tokens_attention_mask_in_name = tokens_attention_mask_in_name
        self.hidden_state_out_name = hidden_state_out_name
        self.pooled_out_name = pooled_out_name
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.add_layer_norm = add_layer_norm
        self.hidden_state_output_index = hidden_state_output_index
        self.expand_token_limit = expand_token_limit
        self.expanded_chunk_size = expanded_chunk_size

        self.autocast_contexts = [nullcontext()] if autocast_contexts is None else autocast_contexts
        self.dtype = dtype

    def length(self) -> int:
        return self._get_previous_length(self.in_name)

    def get_inputs(self) -> list[str]:
        return [self.in_name]

    def get_outputs(self) -> list[str]:
        if self.pooled_out_name:
            return [self.hidden_state_out_name, self.pooled_out_name]
        else:
            return [self.hidden_state_out_name]

    def encode_text_long(self, variation: int, index: int, requested_name: str = None) -> dict:

        tokens = self._get_previous_item(variation, self.in_name, index)

        token_groups = self._group_tokens2(tokens, self.tokenizer)
        # token_groups = self._group_tokens(tokens)

        # TODO: figure out how to handle layer norms... only made this with SDXL in mind and it isn't used there

        with self._all_contexts(self.autocast_contexts):
            text_encoder_output = self.text_encoder(
                token_groups,
                attention_mask=None,
                output_hidden_states=True,
                return_dict=True,
            )
        hidden_state = text_encoder_output.hidden_states[self.hidden_state_output_index]
        # transform (chunk_count, chunk_size + 2, N) -> (chunk_count * chunk_size, N)
        hidden_state_content = hidden_state[:,1:-1,:]
        hidden_state_content = hidden_state_content.reshape((-1, hidden_state_content.shape[-1]))
        # slice hidden_state for <BOS> and <EOS> tokens, reshape to (1,N)
        # assemble <BOS> <content> <EOS> and reshape to (chunk_size * chunk_count + 2,N)
        hidden_state = torch.cat([
            hidden_state[0,0,:].unsqueeze(0),
            hidden_state_content.unsqueeze(0)[0,:,:],
            hidden_state[0,-1,:].unsqueeze(0)
        ]).squeeze()

        pooled_state = None
        if self.pooled_out_name:
            if (hasattr(text_encoder_output, "text_embeds")):
                pooled_state = text_encoder_output.text_embeds
                pooled_state = pooled_state.mean(dim=0).reshape((1,pooled_state.shape[-1]))
            elif hasattr(text_encoder_output, "pooler_output"):
                pooled_state = text_encoder_output.pooler_output.mean(dim=0)
        else:
            pooled_state = None
        pooled_state = None if pooled_state is None else pooled_state.squeeze()
        return {
            self.hidden_state_out_name: hidden_state,
            self.pooled_out_name: pooled_state
        }

    def _group_tokens(self, tokens: Tensor):
        stripped_tokens = tokens[1:-1]  # slice off <EOS> and <BOS> tokens
        chunk_count = stripped_tokens.shape[0] // self.expanded_chunk_size
        # reshape (1,N)->(C,expanded_chunk_size), where C is the number of chunks, N is a multiple of expanded_chunk_size
        stripped_tokens = stripped_tokens.reshape(chunk_count, self.expanded_chunk_size)
        token_groups = []
        for i in range(0, chunk_count):
            # reassemble each chunk to be <BOS> <chunk_content> <EOS>
            chunk = (
                tokens[0].unsqueeze(0),
                stripped_tokens[i, :],
                tokens[-1].unsqueeze(0)
            )
            token_groups.append(torch.cat(chunk))
        return torch.stack(token_groups)

    def _group_tokens2(self, tokens: Tensor, tokenizer: CLIPTokenizer, clip_chunk_size: int = 75):
        if tokens.dim() == 2:
            tokens = tokens.squeeze(0)
        text = tokenizer.decode(tokens, skip_special_tokens=True)
        chunks = _chunk_prompt(text, tokenizer)
        return tokenize_chunked(chunks, tokenizer, clip_chunk_size).to(self.pipeline.device)

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        if not self.add_layer_norm and self.expand_token_limit and self.expanded_chunk_size != 0:
            return self.encode_text_long(variation, index, requested_name)

        tokens = self._get_previous_item(variation, self.in_name, index)
        tokens = tokens.unsqueeze(0)

        if self.tokens_attention_mask_in_name is not None:
            tokens_attention_mask = self._get_previous_item(variation, self.tokens_attention_mask_in_name, index)
            tokens_attention_mask = tokens_attention_mask.unsqueeze(0)
        else:
            tokens_attention_mask = None

        with self._all_contexts(self.autocast_contexts):
            if tokens_attention_mask is not None and self.dtype:
                tokens_attention_mask = tokens_attention_mask.to(dtype=self.dtype)

            text_encoder_output = self.text_encoder(
                tokens,
                attention_mask=tokens_attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )

        hidden_states = text_encoder_output.hidden_states
        if self.pooled_out_name:
            if hasattr(text_encoder_output, "text_embeds"):
                pooled_state = text_encoder_output.text_embeds
            if hasattr(text_encoder_output, "pooler_output"):
                pooled_state = text_encoder_output.pooler_output

        else:
            pooled_state = None

        hidden_states = [hidden_state.squeeze(dim=0) for hidden_state in hidden_states]
        pooled_state = None if pooled_state is None else pooled_state.squeeze(dim=0)

        hidden_state = hidden_states[self.hidden_state_output_index]

        if self.add_layer_norm:
            with self._all_contexts(self.autocast_contexts):
                final_layer_norm = self.text_encoder.text_model.final_layer_norm
                hidden_state = final_layer_norm(
                    hidden_state
                )

        return {
            self.hidden_state_out_name: hidden_state,
            self.pooled_out_name: pooled_state,
        }


def tokenize_chunked(chunks: List[str], tokenizer: CLIPTokenizer, chunk_size: int = 75) -> Tensor:
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

    return torch.cat([chunk['input_ids'] for chunk in tokenized_chunks], dim=0)

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
