import argparse
import copy
from datetime import datetime

from datasets import load_dataset
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import set_seed, GPT2Tokenizer
from transformers.models.gpt2.modeling_gpt2 import *


class CustomGPT2Model(GPT2Model):
    """
    Trick of Customizing original pretrained model, is as below :
     1. Inherit pretrained model (like CustomGPT2Model, inheriting GPT2Model)
     2. Do function shadowing on purpose on Inherited pretrained model (like forward method)
     3. Use Inherited pretrained model as you are using original pretrained model
    """

    def forward(
            self,
            input_ids=None,
            past_key_values=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        # GPT2Attention mask.
        if attention_mask is not None:
            if batch_size <= 0:
                raise ValueError("batch_size has to be defined and > 0")
            attention_mask = attention_mask.view(batch_size, -1)
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask[:, None, None, :]

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.add_cross_attention and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)

        output_shape = input_shape + (hidden_states.size(-1),)

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):

            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure layer_past is on same device as hidden_states (might not be correct)
                if layer_past is not None:
                    layer_past = tuple(past_state.to(hidden_states.device) for past_state in layer_past)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if isinstance(head_mask, torch.Tensor):
                    head_mask = head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:

                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, use_cache, output_attentions)

                    return custom_forward

                outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    None,
                    attention_mask,
                    head_mask[i],
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (outputs[3 if use_cache else 2],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, presents, all_hidden_states, all_self_attentions, all_cross_attentions]
                if v is not None
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class Gpt2ForClassification(nn.Module):
    def __init__(self, gpt2_model_name, num_labels):
        super(Gpt2ForClassification, self).__init__()

        self.hidden_size = GPT2Config.from_pretrained(gpt2_model_name).hidden_size
        self.gpt2 = CustomGPT2Model.from_pretrained(gpt2_model_name)
        self.linear = nn.Linear(in_features=self.hidden_size, out_features=num_labels)

    def forward(self, input_ids, attention_mask):
        gpt2_out, _ = self.gpt2(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)

        batch_size = gpt2_out.shape[0]
        gpt2_out_last_indices = attention_mask.squeeze().sum(dim=-1) - 1
        gpt2_out = gpt2_out[[i for i in range(batch_size)], gpt2_out_last_indices]

        linear_output = self.linear(gpt2_out)
        return linear_output


def inference(batch, model):
    predict = model(batch['input_ids'], batch['attention_mask'])
    return predict


def validate_model(device, dataloader, model, loss_fn):
    model.to(device)
    loss_fn.to(device)

    loss = 0
    correct_val = 0
    data_len = len(dataloader)

    model.eval()
    with torch.no_grad():
        for _, batch in enumerate(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            predict = inference(batch, model)
            loss += loss_fn(predict, batch['labels'])
            correct_val += (predict.argmax(dim=1) == batch['labels']).sum().item()

    current_acc = correct_val / data_len
    return loss, current_acc


def train_model(epochs, device, train_dataloader, validation_dataloader, model, loss_fn, optimizer):
    model.to(device)
    loss_fn.to(device)

    best_acc = 0
    best_model = None

    for t in range(epochs):
        model.train()
        for i, batch in enumerate(tqdm(train_dataloader)):
            batch = {k: v.to(device) for k, v in batch.items()}

            optimizer.zero_grad()
            predict = inference(batch, model)
            loss = loss_fn(predict, batch['labels'])
            loss.backward()
            optimizer.step()

        current_loss, current_acc = validate_model(device, validation_dataloader, model, loss_fn)
        if best_acc < current_acc:
            best_acc = current_acc
            best_model = copy.deepcopy(model)

        print(f"\nValidation loss / current_acc / best_acc : {current_loss}, {current_acc}, {best_acc}")

    return best_model


def main():
    # Parser --
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='gpt2', type=str)  # should be gpt2-xxx
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--seq_max_length', default=128, type=int)
    parser.add_argument('--epochs', default=1, type=int)  # TODO dev
    parser.add_argument('--lr', default=1e-5, type=float)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--seed', default=4885, type=int)

    args = parser.parse_known_args()[0]
    setattr(args, 'device', f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu')
    setattr(args, 'time', datetime.now().strftime('%Y%m%d-%H:%M:%S'))

    print('[List of arguments]')
    for a in args.__dict__:
        print(f'{a}: {args.__dict__[a]}')

    # Device --
    device = args.device

    # Hyper parameter --
    set_seed(args.seed)
    learning_rate = args.lr
    batch_size = args.batch_size
    epochs = args.epochs
    seq_max_length = args.seq_max_length
    model_name = args.model_name

    # Dataset --
    train_dataset = load_dataset('glue', 'sst2', split="train")
    validation_dataset = load_dataset('glue', 'sst2', split="validation")
    dataset_num_labels = 2

    # Prepare tokenizer, dataloader, model, loss function, optimizer, etc --
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    def encode_input(examples):
        return tokenizer(examples['sentence'], max_length=seq_max_length, truncation=True, padding='max_length')

    def format_output(examples):
        return {'labels': examples['label']}

    train_dataset = train_dataset.map(encode_input, batched=True)
    train_dataset = train_dataset.map(format_output, batched=True)
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    validation_dataset = validation_dataset.map(encode_input, batched=True)
    validation_dataset = validation_dataset.map(format_output, batched=True)
    validation_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size)

    model = Gpt2ForClassification(model_name, dataset_num_labels)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    model = train_model(epochs, device, train_dataloader, validation_dataloader, model, loss_fn, optimizer)
    best_loss, best_acc = validate_model(device, validation_dataloader, model, loss_fn)
    print(f"\nValidation best_loss / best_acc with best model: {best_loss} / {best_acc}")


if __name__ == '__main__':
    main()
