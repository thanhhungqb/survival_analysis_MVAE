"""
Implement some function related to survival analysis
"""
import torch
import torch.nn as nn
from pycox.models.loss import NLLLogistiHazardLoss
from pycox.preprocessing.label_transforms import LabTransDiscreteTime


class LossAELogHaz(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.loss_surv = NLLLogistiHazardLoss()

    def forward(self, phi, target_loghaz):
        idx_durations, events = target_loghaz
        loss_surv = self.loss_surv(phi, idx_durations, events)

        return loss_surv


class LossLogHazInd(nn.Module):
    """
    Task Haz + Individual reg, size (T+1)
    """

    def __init__(self, alpha=0.5, **kwargs):
        super().__init__()
        self.loss_surv = NLLLogistiHazardLoss()
        self.loss_ind = nn.MSELoss()
        self.alpha = alpha

    def forward(self, phi, target_loghaz):
        phi, ind_sv = phi[:, :-1], phi[:, -1]

        idx_durations, events = target_loghaz[:, 0], target_loghaz[:, 1]
        idx_durations = idx_durations.type(torch.int64)
        t_ind_sv, t_org_event = target_loghaz[:, 2], target_loghaz[:, 3]

        loss_surv = self.loss_surv(phi, idx_durations, events)

        # mse only for event
        with torch.no_grad():
            n_sample = phi.size()[0]
            device = phi.device
            zeros = torch.tensor([0.] * n_sample).to(device)
            ones = torch.tensor([1.] * n_sample).to(device)
            hs = torch.where(t_org_event > 0, ones, zeros)
        se = (ind_sv - t_ind_sv) * (ind_sv - t_ind_sv)
        loss_mse = torch.sum(se * hs) / torch.sum(hs)

        return loss_surv * self.alpha + loss_mse * (1 - self.alpha)


class LabelTransform:
    def __init__(self, train_durations, train_events, num_durations=10, **kwargs):
        labtrans = LabTransDiscreteTime(num_durations)

        # fit transform with train label
        y_train_surv = labtrans.fit_transform(train_durations, train_events)
        _ = y_train_surv

        self.trans = labtrans

    def __call__(self, durations, events, **kwargs):
        """
        If list of durations and events, then return a list (batch mode).
        If only one element given for each, then, working with single element, serving as
        label_fn.
        :param durations:
        :param events:
        :param kwargs:
        :return: form (array([10]), array([1.], dtype=float32)), should convert to correct label before pass
        """
        return self.trans.transform(durations, events)


def make_label_transform_pipe(**config):
    train_df = config['train_df']
    train_durations, train_events = [train_df[o] for o in config['y_names']]
    params = {**config, 'train_durations': train_durations, 'train_events': train_events}
    label_trans = LabelTransform(**params)
    config['labtrans'] = label_trans.trans

    config['labtrans_cuts'] = label_trans.trans.idu.cuts
    return config


# surv post process for predict, see input form `prlab.torch.utils.default_post_process_fn`
def surv_ppp_merge_hazard_st_fn(batch_tensor, **config):
    """
    output from MultiDecoderVAE with only one second_decoder
    [[bs, n_hazard+1]]"""
    x = batch_tensor[0].cpu().detach().numpy().tolist()

    return x
