import torch
import math
from torch import nn

import torch.nn.functional as F

from transformers.activations import get_activation

from .utils import init_weights

def _mask(logits, mask):
    return mask * logits - 1e3 * (1 - mask)

# VarMisuse -----------------------------------------------------------------

class _LocRepairPointerHead(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.prediction = nn.Linear(config.hidden_size, 2)

        self.apply(init_weights)

    def forward(self, input_states):
        hidden = self.dense(input_states)
        hidden = get_activation("gelu")(hidden)
        logits = self.prediction(hidden)
        logits = logits.transpose(2, 1)
        return logits


class VarMisuseBaseModel(nn.Module):

    def __init__(self, config, encoder):
        super().__init__()

        self.config = config
        self.encoder = encoder

    @torch.no_grad()
    def score(self, logits, labels):
        probs = nn.Softmax(dim = 2)(logits)

        # Location metrics
        loc_predict = probs[:, 0, :]
        loc_labels =  labels[:, 0, :]

        locate = loc_predict.argmax(dim=1)
        locate = torch.nn.functional.one_hot(locate, num_classes=loc_predict.shape[1]).float()
        locate_acc = (locate * loc_labels).sum(dim=1)

        buggy_labels = 1 - loc_labels[:, 0]

        # Buggy classification
        false_alarms = 1 - ((1 - buggy_labels)*locate_acc).sum() / ((1 - buggy_labels).sum() + 1e-9)
        bug_acc      = (buggy_labels * locate_acc).sum() / (buggy_labels.sum() + 1e-9)

        # Classification
        cls_predict = loc_predict[:, 0].round()
        cls_labels  = loc_labels[:, 0]
        cls_acc = (cls_predict * cls_labels).mean() + ((1 - cls_predict) * buggy_labels).mean()

        #Repair pointer
        rep_probs = probs[:, 1, :]
        rep_labels = labels[:, 1, :]

        target_probs   = (rep_labels * rep_probs).sum(dim=-1)
        target_predict = target_probs.round()
        target_acc = (target_predict * buggy_labels).sum() / (1e-9 + buggy_labels.sum())

        joint_acc = (buggy_labels * locate_acc * target_predict).sum() / (1e-9 + buggy_labels.sum())

        return {
            "classification_acc": cls_acc.item(),
            "localization_acc": locate_acc.mean().item(),
            "bug_acc": bug_acc.item(),
            "false_alarm_rate": false_alarms.item(),
            "repair_acc": target_acc.item(),
            "loc_repair_acc": joint_acc.item(),
            "avg_prediction": cls_predict.mean().item()
        }

    def loc_repair_acc(self, tokens, position_ids = None, labels = None):
        pass
    
    def forward(self, tokens, token_mask = None, position_ids = None, labels = None):

        prediction = self.loc_repair_logits(tokens, position_ids, labels)
        
        # Mask prediction
        if token_mask is not None:
            token_mask = token_mask.float().unsqueeze(1).expand_as(prediction)
            prediction = _mask(prediction, token_mask)

        # Calculate a loss if necessary
        if labels is not None:
            log_probs = nn.LogSoftmax(dim=2)(prediction)
            norm = labels.sum(dim=-1, keepdim = True)

            per_token_loss = (-labels * log_probs) / (norm + 1e-9)
            per_example_loss = per_token_loss.sum(dim=-1)

            per_task_loss = per_example_loss.mean(dim = 0)
            return per_task_loss.sum(), prediction


        return prediction


class VarMisuseModel(VarMisuseBaseModel):

    def __init__(self, config, encoder):
        super().__init__(config, encoder)

        self.head = _LocRepairPointerHead(config)

    def loc_repair_logits(self, tokens, position_ids = None, labels = None):
        attention_mask = tokens.sum(dim=2).clamp_(0, 1)

        encoding, _ = self.encoder(
            tokens = tokens,
            attention_mask = attention_mask.bool(),
            position_ids = position_ids
        )

        return self.head(encoding)



class _LocalizationHead(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.ff_in = nn.Linear(config.hidden_size + config.embedding_size, config.embedding_size)
        self.ff_out = nn.Linear(config.embedding_size, 1)

        self.apply(init_weights)
    
    def forward(self, context_embed, token_embed):
        
        hidden = torch.cat([context_embed, token_embed], dim=-1)
        hidden = self.ff_in(hidden)
        hidden = nn.Tanh()(hidden)
        logits = self.ff_out(hidden)

        return logits.transpose(2, 1)


class _CopyRepairHead(nn.Module):

    def __init__(self, config):
        super().__init__()
    
    def forward(self, error_tokens, hidden):
        repair_logits = torch.bmm(error_tokens.unsqueeze(1), hidden.transpose(2, 1))
        repair_logits /= math.sqrt(error_tokens.shape[1]) # Stabilizes loss

        return repair_logits


class VarMisuseTargetedModel(VarMisuseBaseModel):

    def __init__(self, config, encoder):
        super().__init__(config, encoder)

        self.loc_head    = _LocalizationHead(config)
        self.repair_head = _CopyRepairHead(config)

    def loc_repair_logits(self, tokens, position_ids = None, labels = None):
        attention_mask = tokens.sum(dim=2).clamp_(0, 1)

        context, tokens = self.encoder(
            tokens = tokens,
            attention_mask = attention_mask.bool(),
            position_ids = position_ids
        )

        loc_logits = self.loc_head(context, tokens)
        
        if labels is not None: # We are training
            locate_mask = labels[:, 0, :].bool()
        else: # We are at inference
            locate = loc_logits.argmax(dim=1)
            locate_mask = F.one_hot(locate, num_classes=locate.shape[1]).bool()

        error_hidden = context[locate_mask]
        repair_logits = self.repair_head(error_hidden, context)
     
        assert loc_logits.shape == repair_logits.shape, "Location shape differs from repair shape (%s != %s)" % (str(loc_logits.shape), str(repair_logits.shape))
       
        return torch.cat([loc_logits, repair_logits], dim = 1)


class _FFN(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.dense  = nn.Linear(config.hidden_size, config.hidden_size)
        self.ln     = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.output = nn.Linear(config.hidden_size, config.decoder_vocab_size, bias = False)

        self.apply(init_weights)
    
    def forward(self, hidden):
        hidden = self.dense(hidden)
        hidden = get_activation("gelu")(hidden)
        hidden = self.ln(hidden)
        return self.output(hidden)


class _PointerHead(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.decoder = _FFN(config)


    def forward(self, error_tokens, hidden, pointer_labels = None, error_repair_labels = None, token_mask = None):

        # Equivalent to attention distribution in Pointer Gen
        repair_logits = torch.bmm(error_tokens.unsqueeze(1), hidden.transpose(2, 1))
        repair_logits /= math.sqrt(error_tokens.shape[1])

        if token_mask is not None: repair_logits = _mask(repair_logits, token_mask.unsqueeze(1))

        if error_repair_labels is not None:
            vocab_logits  = self.decoder(error_tokens)

            common_logits = torch.cat([repair_logits.squeeze(), vocab_logits], dim = 1)
            common_logits = -nn.LogSoftmax(dim = 1)(common_logits)

            ohe_labels = F.one_hot(error_repair_labels, num_classes=self.config.decoder_vocab_size)
            ohe_labels[:, 1] = 0

            labels = torch.cat([pointer_labels, ohe_labels], dim=1)
            norm   = labels.sum(dim=-1)
            
            repair_loss = (common_logits * labels).sum(dim = -1) / (norm + 1e-9)
            repair_loss = repair_loss.mean()
        
            return repair_loss, repair_logits

        return None, repair_logits



class VarMisusePointerModel(VarMisuseBaseModel):

    def __init__(self, config, encoder):
        super().__init__(config, encoder)

        self.loc_head    = _LocalizationHead(config)
        self.repair_head = _PointerHead(config)

    def forward(self, tokens, token_mask = None, position_ids = None, labels = None):
        
        attention_mask = tokens.sum(dim=2).clamp_(0, 1)

        context, tokens = self.encoder(
            tokens = tokens,
            attention_mask = attention_mask.bool(),
            position_ids = position_ids
        )

        loc_logits = self.loc_head(context, tokens)

        error_repair_labels = None
        
        if labels is not None: # We are training
            locate_mask = labels[:, 0, :].bool()

            error_repair_labels = labels[:, 2, :]
            error_repair_labels = error_repair_labels[locate_mask]
        else: # We are at inference
            locate = loc_logits.argmax(dim=1)
            locate_mask = F.one_hot(locate, num_classes=locate.shape[1]).bool()

        error_hidden = context[locate_mask]
        repair_loss, repair_logits = self.repair_head(error_hidden, context,
                                                         labels[:, 1, :], error_repair_labels, token_mask)

        assert loc_logits.shape == repair_logits.shape, "Location shape differs from repair shape (%s != %s)" % (str(loc_logits.shape), str(repair_logits.shape))

        prediction = torch.cat([loc_logits, repair_logits], dim = 1)

        # Calculate loss for localization

        # Mask prediction
        if token_mask is not None:
            prediction = _mask(loc_logits, token_mask)

        # Calculate a loss if necessary
        if labels is not None:
            loc_labels = labels[:, 0, :]
            log_probs = nn.LogSoftmax(dim=1)(loc_logits.squeeze())
            norm = loc_labels.sum(dim=-1, keepdim = True)

            per_token_loss = (-loc_labels * log_probs) / (norm + 1e-9)
            loc_loss = per_token_loss.sum(dim=-1)
            loc_loss = loc_loss.mean()

            return loc_loss + repair_loss, prediction


        return prediction


# General model that works with inner repairs and localization --------------------------------


class _LocateHead(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.ffn_in  = nn.Linear(2 * config.hidden_size, config.hidden_size)
        self.ln      = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.ffn_out = nn.Linear(config.hidden_size, 1) 

        self.apply(init_weights)

    def forward(self, context_embed, token_embed, token_mask = None, labels = None):
        assert context_embed.shape[1] == token_embed.shape[1]

        # Localization prediction --------------------------------

        diff_vector = token_embed - context_embed
        diff_vector = torch.cat([context_embed, diff_vector], dim = 2)
        hidden = self.ffn_in(diff_vector)
        hidden = get_activation("gelu")(hidden)
        hidden = self.ln(hidden)
        hidden = self.ffn_out(hidden)
        hidden = hidden.squeeze()

        if token_mask is not None: hidden = _mask(hidden, token_mask)

        # Loss calculation ---------------------------------------

        if labels is not None:
            locate_labels = labels[:, 0, :]
            log_probs = nn.LogSoftmax(dim=1)(hidden)
            loss = (-locate_labels * log_probs).sum(dim=1)
            
            return loss.mean(), hidden

        return None, hidden


class _RepairHead(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        if config.decoder_vocab_size > 0: # We have a target vocab
            self.decoder = nn.Linear(config.hidden_size, config.decoder_vocab_size, bias = False)
            self.apply(init_weights)

    def forward(self, error_embed, context_embed, token_mask = None, labels = None, target_labels = None):
        
        # Compute a local pointer --------------------------------

        repair_logits = torch.bmm(error_embed.unsqueeze(1), context_embed.transpose(2, 1)).squeeze()
        repair_logits /= math.sqrt(error_embed.shape[1])

        if token_mask is not None: repair_logits = _mask(repair_logits, token_mask)

        if labels is not None:
            repair_labels = labels[:, 1, :]

        # Compute a global vocab index ---------------------------

        if hasattr(self, "decoder"):
            decoder_logits = self.decoder(error_embed)
            repair_logits  = torch.cat([repair_logits, decoder_logits], dim = 1)

            if labels is not None and target_labels is not None:
                ohe_labels = F.one_hot(target_labels, num_classes=self.config.decoder_vocab_size)
                ohe_labels[:, 1] = 0
                repair_labels = torch.cat([repair_labels, ohe_labels], dim = 1)

        # Loss computation ---------------------------------------

        if labels is not None:
            repair_log_probs = nn.LogSoftmax(dim = 1)(repair_logits)
            norm = repair_labels.sum(dim = -1).clamp_(0, 1)

            # Collect log probs
            # log sum_(t_i = w)(P(t_i)) = log sum_(t_i = w)(exp log P(t_i))
            #      = LSE(log P(t_i))
            repair_log_probs = _mask(repair_log_probs, repair_labels)
            per_example_loss = -norm * torch.logsumexp(repair_log_probs, dim = 1)

            return per_example_loss.mean(), repair_logits
        
        return repair_logits



class LocateRepairModel(nn.Module):

    def __init__(self, config, encoder):
        super().__init__()

        self.config = config
        self.encoder = encoder

        self.locate_head = _LocateHead(config)
        self.repair_head = _RepairHead(config)

    @torch.no_grad()
    def score(self, logits, labels):
        locate_logits, repair_logits = logits

        # Score for localization
        loc_predict = nn.Softmax(dim = 1)(locate_logits)
        loc_labels =  labels[:, 0, :]

        locate = loc_predict.argmax(dim=1)
        locate = torch.nn.functional.one_hot(locate, num_classes=loc_predict.shape[1]).float()
        locate_acc = (locate * loc_labels).sum(dim=1)

        buggy_labels = 1 - loc_labels[:, 0]

        # Buggy classification
        false_alarms = 1 - ((1 - buggy_labels)*locate_acc).sum() / ((1 - buggy_labels).sum() + 1e-9)
        bug_acc      = (buggy_labels * locate_acc).sum() / (buggy_labels.sum() + 1e-9)

        # Classification
        cls_predict = loc_predict[:, 0].round()
        cls_labels  = loc_labels[:, 0]
        cls_acc = (cls_predict * cls_labels).mean() + ((1 - cls_predict) * buggy_labels).mean()

        # Repair scores
        rep_probs = nn.Softmax(dim = 1)(repair_logits)
        rep_labels = labels[:, 1, :]

        if rep_probs.shape[1] != rep_labels.shape[1]:
            target_labels = labels[:, 2, :]
            target_labels = target_labels[loc_labels.bool()]
            ohe_labels = F.one_hot(target_labels, num_classes=self.config.decoder_vocab_size)
            ohe_labels[:, 1] = 0
            rep_labels = torch.cat([rep_labels, ohe_labels], dim = 1)

        target_probs   = (rep_labels * rep_probs).sum(dim=-1)
        target_predict = target_probs.round()
        target_acc = (target_predict * buggy_labels).sum() / (1e-9 + buggy_labels.sum())

        joint_acc = (buggy_labels * locate_acc * target_predict).sum() / (1e-9 + buggy_labels.sum())

        return {
            "classification_acc": cls_acc.item(),
            "localization_acc": locate_acc.mean().item(),
            "bug_acc": bug_acc.item(),
            "false_alarm_rate": false_alarms.item(),
            "repair_acc": target_acc.item(),
            "loc_repair_acc": joint_acc.item(),
            "avg_prediction": cls_predict.mean().item()
        }


    def forward(self, tokens, token_mask = None, position_ids = None, labels = None):

        attention_mask = tokens.sum(dim=2).clamp_(0, 1)

        context_embed, token_embed = self.encoder(
            tokens = tokens,
            attention_mask = attention_mask.bool(),
            position_ids = position_ids
        )

        locate_loss, locate_logits = self.locate_head(context_embed, 
                                                        token_embed,
                                                        token_mask,
                                                        labels)

        # Either use the gold localization or the predicted to get the error position

        error_repair_labels = None
        
        if labels is not None: # We are training
            locate_mask = labels[:, 0, :].bool()

            error_repair_labels = labels[:, 2, :]
            error_repair_labels = error_repair_labels[locate_mask]
        else: # We are at inference
            locate = locate_logits.argmax(dim=1)
            locate_mask = F.one_hot(locate, num_classes=locate.shape[1]).bool()

        error_hidden = context_embed[locate_mask]

        # ----------------------------------------------------------------

        repair_loss, repair_logits = self.repair_head(
            error_hidden,
            context_embed,
            token_mask,
            labels,
            error_repair_labels
        )

        if labels is not None:
            return locate_loss + repair_loss, (locate_logits, repair_logits)

        return (locate_logits, repair_logits)
