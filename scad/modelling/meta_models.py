import torch
from torch import nn

def _mask(logits, mask):
    return mask * logits - 1e3 * (1 - mask)

# VarMisuse -----------------------------------------------------------------

class VarMisuseModel(nn.Module):

    def __init__(self, config, encoder):
        super().__init__()

        self.config = config
        self.encoder = encoder

        self.prediction = nn.Linear(config.hidden_size, 2)

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
        cls_predict = loc_predict[:, 0]
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

    def forward(self, tokens, token_mask = None, position_ids = None, labels = None):

        attention_mask = tokens.sum(dim=2).clamp_(0, 1)

        encoding, _ = self.encoder(
            tokens = tokens,
            attention_mask = attention_mask.bool(),
            position_ids = position_ids
        )

        prediction = self.prediction(encoding)
        prediction = prediction.transpose(-2, -1)

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

            loc_loss = per_example_loss[:, 0].mean()

            repair_loss_norm = norm[:, 1, 0].clamp_(0, 1).sum()
            repair_loss = per_example_loss[:, 1].sum() / (repair_loss_norm + 1e-9)
        
            return loc_loss + repair_loss, prediction

        return prediction