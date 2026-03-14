from contextlib import nullcontext

import torch
from transformers import CLIPTextModel, CLIPTextModelWithProjection

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
            add_layer_norm: bool,
            hidden_state_output_index: int | None = None,
            autocast_contexts: list[torch.autocast | None] = None,
            dtype: torch.dtype | None = None,
    ):
        super(EncodeClipText, self).__init__()
        self.in_name = in_name
        self.tokens_attention_mask_in_name = tokens_attention_mask_in_name
        self.hidden_state_out_name = hidden_state_out_name
        self.pooled_out_name = pooled_out_name
        self.text_encoder = text_encoder
        self.add_layer_norm = add_layer_norm
        self.hidden_state_output_index = hidden_state_output_index

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

    def get_item(self, variation: int, index: int, requested_name: str = None, chunk_if_needed: bool = False, chunk_size: int = 75) -> dict:
        tokens = self._get_previous_item(variation, self.in_name, index)

        if self.tokens_attention_mask_in_name is not None:
            tokens_attention_mask = self._get_previous_item(variation, self.tokens_attention_mask_in_name, index)
        else:
            tokens_attention_mask = None

        if chunk_if_needed and tokens.shape[0] > 77:
            return self._get_item_chunked(tokens, tokens_attention_mask, chunk_size)
        else:
            return self._get_item_single(tokens, tokens_attention_mask)

    def _get_item_single(self, tokens: torch.Tensor, tokens_attention_mask: torch.Tensor | None) -> dict:
        tokens = tokens.unsqueeze(0)
        if tokens_attention_mask is not None:
            tokens_attention_mask = tokens_attention_mask.unsqueeze(0)

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

    def _get_item_chunked(self, tokens: torch.Tensor, tokens_attention_mask: torch.Tensor | None, chunk_size: int) -> dict:
        # split tokens into chunks of chunk_size tokens, and add BOS/EOS to each
        bos_token = tokens[0]
        eos_token = tokens[-1]

        # remove BOS and EOS
        tokens = tokens[1:-1]
        if tokens_attention_mask is not None:
            tokens_attention_mask = tokens_attention_mask[1:-1]

        # split into chunks of chunk_size
        input_id_chunks = tokens.split(chunk_size)
        if tokens_attention_mask is not None:
            attention_mask_chunks = tokens_attention_mask.split(chunk_size)
        else:
            attention_mask_chunks = [None] * len(input_id_chunks)

        last_input_id_chunk = input_id_chunks[-1]
        if len(last_input_id_chunk) < chunk_size:
            # pad last chunk with EOS
            input_id_chunks = list(input_id_chunks)
            input_id_chunks[-1] = torch.cat([
                last_input_id_chunk,
                torch.full((chunk_size - len(last_input_id_chunk),), eos_token, dtype=tokens.dtype, device=tokens.device)
            ])
            if tokens_attention_mask is not None:
                attention_mask_chunks = list(attention_mask_chunks)
                last_attention_mask_chunk = attention_mask_chunks[-1]
                attention_mask_chunks[-1] = torch.cat([
                    last_attention_mask_chunk,
                    torch.full((chunk_size - len(last_attention_mask_chunk),), 0, dtype=tokens_attention_mask.dtype, device=tokens_attention_mask.device)
                ])

        # add BOS and EOS to each chunk
        input_id_chunks = [torch.cat([bos_token.unsqueeze(0), chunk, eos_token.unsqueeze(0)]) for chunk in input_id_chunks]
        if tokens_attention_mask is not None:
            attention_mask_chunks = [torch.cat([torch.ones(1, dtype=tokens_attention_mask.dtype, device=tokens_attention_mask.device), chunk, torch.ones(1, dtype=tokens_attention_mask.dtype, device=tokens_attention_mask.device)]) for chunk in attention_mask_chunks]

        hidden_states = []
        pooled_states = []

        for chunk_tokens, chunk_attention_mask in zip(input_id_chunks, attention_mask_chunks):
            res = self._get_item_single(chunk_tokens, chunk_attention_mask)
            hidden_states.append(res[self.hidden_state_out_name])
            if self.pooled_out_name:
                pooled_states.append(res[self.pooled_out_name])

        hidden_state = torch.cat(hidden_states, dim=0)
        pooled_state = torch.mean(torch.stack(pooled_states), dim=0) if self.pooled_out_name else None

        return {
            self.hidden_state_out_name: hidden_state,
            self.pooled_out_name: pooled_state,
        }
