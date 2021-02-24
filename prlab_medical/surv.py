"""
Implement some function related to survival analysis
"""
import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtuples as tt
from lifelines.utils import concordance_index
from pycox.models import LogisticHazard
from pycox.models.loss import NLLLogistiHazardLoss
from pycox.preprocessing.label_transforms import LabTransDiscreteTime

from prlab.torch.utils import cumsum_rev
from prlab_medical.data_loader import event_norm


class LogisticHazardE(LogisticHazard):
    """
    Extend LogisticHazard to implement pmf
    see `pycox.models.loss.nll_logistic_hazard`
    phi {torch.tensor} -- Estimates in (-inf, inf), where hazard = sigmoid(phi).
    """

    def __init__(self, net, optimizer=None, device=None, duration_index=None, loss=None, **kwargs):
        super().__init__(net=net, optimizer=optimizer, loss=loss, device=device, duration_index=duration_index)

    def predict_pmf(self, input, batch_size=8224, numpy=None, eval_=True, to_cpu=False,
                    num_workers=0, epsilon=1e-7):
        hazard = self.predict_hazard(input, batch_size, False, eval_, to_cpu, num_workers)
        surv = (1 - hazard).add(epsilon).log().cumsum(1).exp()

        pmf = hazard * surv
        return tt.utils.array_or_tensor(pmf, numpy, input)


class MSELossFilter:
    """
    see `MSELossE`
    Filter of mse, target = [bs, 2], the second value is filter 1/0
    Using in censoring data.
    If data is non-censoring => 1 => count
    if data is censoring => 0 => count if right-censoring and flag and pred < taget
    """

    def __init__(self, **kwargs):
        self.base = nn.MSELoss
        self.right_censoring_loss = kwargs.get('right_censoring_loss', False)

    def __call__(self, pred, target, **kwargs):
        return self.forward(pred, target, **kwargs)

    def forward(self, pred, target, **kwargs):
        ret = F.mse_loss(pred, target[:, 0:1], reduction='none')
        events = target[:, 1]

        device = target.device
        with torch.no_grad():
            n_sample = target.size()[0]
            zeros = torch.tensor([0.] * n_sample).to(device)
            ones = torch.tensor([1.] * n_sample).to(device)
            have_count = torch.where(events > 0, ones, zeros)
            flag_lt = torch.where(pred < target[:, 0], ones, zeros)
            hs3 = torch.where(have_count + flag_lt > 0, ones, zeros)
            have_count = hs3 if self.right_censoring_loss else have_count

        loss = torch.sum(ret * have_count) / torch.sum(have_count)

        return loss.mean()


class MSELossWithRightCensoring(MSELossFilter):
    def __init__(self, **kwargs):
        super(MSELossWithRightCensoring, self).__init__(**{**kwargs, 'right_censoring_loss': True})


class LossAELogHaz(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.loss_surv = NLLLogistiHazardLoss()

    def forward(self, phi, target_loghaz):
        idx_durations, events = target_loghaz[:, 0], target_loghaz[:, 1]
        idx_durations = idx_durations.type(torch.int64)
        loss_surv = self.loss_surv(phi, idx_durations, events)

        return loss_surv


class LossLogHazInd(nn.Module):
    """
    Task Haz + Individual reg, size (T+1)
    phi {torch.tensor} -- Estimates in (-inf, inf), where hazard = sigmoid(phi)
    # see `pycox.models.loss.nll_logistic_hazard`
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


class LossHazClsInd(LossLogHazInd):
    """
    Loss include several part: NLLLogistiHazardLoss, cross entropy, MSE of ind reg and MSE of ind infer from f_j

    Task Haz + Individual reg, size (T+1)
    Note: different meaning of phi with LossLogHazInd as bellow
        phi {torch.tensor} -- Estimates in (-inf, inf), and softmax => f_j = p(T=t_j)
        S_j = S(t_j) = p(T >= t_j) = cumsum(f_j) # reverse
        h_j = h(t_j) = p(T = t_j | T >= t_j) = f_j / S_j
    """

    def __init__(self, alpha=0.5, **kwargs):
        super().__init__(alpha=alpha, **kwargs)

        cuts = kwargs['cuts']  # using fixed cuts values
        self.cuts = cuts  # 0: c0-c1, 1: c1-c2, ..., i: ci-c(i+1)...
        self.mid_values = [0.5 * (cuts[i] + cuts[i + 1]) for i in range(len(cuts) - 1)]
        self.mid_values.append(cuts[-1] + 0.5 * (cuts[-2] + cuts[-1]))  # last value is base on previous
        self.mid_values = torch.tensor(self.mid_values)

    def forward(self, phi, target_loghaz):
        phi, ind_sv = phi[:, :-1], phi[:, -1]
        device = phi.device
        esp = 1e-6

        idx_durations, events = target_loghaz[:, 0], target_loghaz[:, 1]
        idx_durations = idx_durations.type(torch.int64)
        t_ind_sv, t_org_event = target_loghaz[:, 2], target_loghaz[:, 3]

        f_j = torch.softmax(phi, dim=-1)
        s_j = cumsum_rev(f_j) + esp
        h_j = f_j / s_j

        # expected survival time from f_j: \sum(f_j * mid_values_j)
        values_dev = self.mid_values.to(device=device)  # move to correct device
        ind_sv_e = torch.sum(f_j * values_dev, dim=-1)  # bs

        # x_phi = rev_sigmoid(h_j) because NLLLogistiHazardLoss need values before sigmoid
        h_j = h_j + esp
        x_phi = torch.log(h_j) - torch.log(1 - h_j)
        loss_surv = self.loss_surv(x_phi, idx_durations, events)
        loss_surv_ce = None  # TODO CE(f_j, idx_durations)

        # mse only for event
        with torch.no_grad():
            n_sample = phi.size()[0]
            zeros = torch.tensor([0.] * n_sample).to(device)
            ones = torch.tensor([1.] * n_sample).to(device)
            hs = torch.where(t_org_event > 0, ones, zeros)

        se = (ind_sv - t_ind_sv) * (ind_sv - t_ind_sv)
        loss_mse = torch.sum(se * hs) / torch.sum(hs)

        se_e = (ind_sv_e - t_ind_sv) * (ind_sv_e - t_ind_sv)
        loss_mse_e = torch.sum(se_e * hs) / torch.sum(hs)

        # TODO loss = loss_surv + loss_surv_ce + loss_mse + loss_mse_e
        # loss = loss_surv * self.alpha + loss_mse * (1 - self.alpha)
        loss = loss_surv * self.alpha + 0.5 * (loss_mse + loss_mse_e) * (1 - self.alpha)
        # loss = 0.5 * (loss_mse + loss_mse_e)
        # loss = loss_surv

        return loss


class LabelTransform:
    def __init__(self, train_durations=None, train_events=None, cuts=None, num_durations=10, **kwargs):
        """
        :param train_durations:
        :param train_events:
        :param cuts: list, if given then three others are not need anymore
        :param num_durations:
        :param kwargs:
        """

        # if cuts given (list of cut points), then all other values are not need
        if cuts is not None:
            self.trans = LabTransDiscreteTime(cuts=cuts)

            # a, b = np.array([20, 400, 5, 50000]), np.array([0, 0, 1, 1])
            # print('test', self.trans.transform(a, b))
            # exit(0)
            #
            return None

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


def train_valid_filter(**config):
    """
    Train and valid filter based on the column with 0/1 for omit/keep.
    Should be run after train_test_split_fold
    :param config: train_df, may be have valid_df
    :return: updated train_df, valid_df with only 1 value in column name filter_col_name
    """
    filter_col_name = config['filter_col_name']

    df = config['train_df']
    config['train_df'] = df[df[filter_col_name] == 1]

    if config.get('valid_df') is not None:
        df = config['valid_df']
        config['valid_df'] = df[df[filter_col_name] == 1]
    return config


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
    ele = batch_tensor[0].cpu().detach()
    hazards = torch.sigmoid_(ele[:, :-1]).numpy().tolist()
    ind_st = ele[:, -1].numpy().tolist()
    x = zip(hazards, ind_st)

    # for easy to use, separated n_hazard and survival time predict and named it
    def fn(one):
        return {'hazard': one[0], 'survival time': one[1]}

    ret = [fn(one) for one in x]

    return ret


# surv post process for predict, see input form `prlab.torch.utils.default_post_process_fn`
def surv_ppp_merge_hazard_sm_st_fn(batch_tensor, **config):
    """
    output from MultiDecoderVAE with only one second_decoder
    [[bs, n_hazard+1]]"""
    esp = 1e-6
    ele = batch_tensor[0].cpu().detach()
    # hazards = torch.sigmoid_(ele[:, :-1]).numpy().tolist()
    phi = ele[:, :-1]
    f_j = torch.softmax(phi, dim=-1)
    s_j = cumsum_rev(f_j) + esp
    h_j = f_j / s_j

    # x_phi = rev_sigmoid(h_j) because NLLLogistiHazardLoss need values before sigmoid
    h_j = h_j + esp
    x_phi = torch.log(h_j) - torch.log(1 - h_j)

    ind_st = ele[:, -1].numpy().tolist()

    # expected survival time from f_j: \sum(f_j * mid_values_j)
    # values_dev = config['loss_func'].second_loss[0].mid_values  # TODO fix hard code here
    # ind_sv_e = torch.sum(f_j * values_dev, dim=-1)  # bs
    # ind_sv_e = ind_sv_e.numpy().tolist()

    x_phi = x_phi.numpy().tolist()
    x = zip(x_phi, ind_st, ind_st)

    # for easy to use, separated n_hazard and survival time predict and named it
    def fn(one):
        return {'hazard': one[0], 'survival time 2': one[2], 'survival time': one[1]}

    ret = [fn(one) for one in x]

    return ret


def surv_ppp_merge_hazard_st_single_fn(batch_tensor, **config):
    """ just override surv_ppp_merge_hazard_sm_st_fn with only 1 element (as DNN)"""
    return surv_ppp_merge_hazard_sm_st_fn([batch_tensor, None], **config)


def surv_ppp_merge_hazard_sm_st_fn2(batch_tensor, **config):
    """
    output from MultiDecoderVAE with only one second_decoder
    [[bs, n_hazard+1]]"""
    esp = 1e-6

    ind_st = batch_tensor[1].cpu().detach()

    # hazards = torch.sigmoid_(ele[:, :-1]).numpy().tolist()
    phi = batch_tensor[0].cpu().detach()
    f_j = torch.softmax(phi, dim=-1)
    s_j = cumsum_rev(f_j) + esp
    h_j = f_j / s_j

    # x_phi = rev_sigmoid(h_j) because NLLLogistiHazardLoss need values before sigmoid
    h_j = h_j + esp
    x_phi = torch.log(h_j) - torch.log(1 - h_j)

    # ind_st = ele[:, -1].numpy().tolist()

    # expected survival time from f_j: \sum(f_j * mid_values_j)
    # values_dev = config['loss_func'].second_loss[0].mid_values  # TODO fix hard code here
    # ind_sv_e = torch.sum(f_j * values_dev, dim=-1)  # bs
    # ind_sv_e = ind_sv_e.numpy().tolist()

    ind_st = ind_st.numpy().tolist()
    x_phi = x_phi.numpy().tolist()
    x = zip(x_phi, ind_st, ind_st)

    # for easy to use, separated n_hazard and survival time predict and named it
    def fn(one):
        return {'hazard': one[0], 'survival time 2': one[2], 'survival time': one[1][0]}

    ret = [fn(one) for one in x]

    return ret


# report for survival
def report_survival_time(**config):
    # run after make prediction and save to file, e.g. predict.csv
    # compare with test.csv, metric maybe MAE, CI, ...
    # column name should be fixed (customable later)
    test_filename, predict_filename = 'test.csv', 'predict.csv'
    pid_name = 'pid'
    predict_header_names = ['hazard', 'survival time', 'pid']

    gt = pd.read_csv(config['cp'] / test_filename)
    pred = pd.read_csv(config['cp'] / predict_filename)

    gt = gt[config['y_names'] + [pid_name]]
    pred = pred[predict_header_names]

    gt = gt.rename(columns={config['y_names'][0]: 'gt_sv', config['y_names'][1]: 'event', 'pid': pid_name})
    pred = pred.rename(
        columns={predict_header_names[0]: 'pred_hazard', predict_header_names[1]: 'pred_sv', 'pid': pid_name})

    # merge two df to one by pid
    merge_df = pd.concat([gt, pred], axis=1, join='inner')
    merge_df['event'] = event_norm(merge_df['event'])

    try:
        merge_df.to_csv(config['cp'] / 'merge_gt_pred.csv')
    except:
        pass

    # for CI
    hazard = merge_df['pred_hazard']
    hazard = [[float(i) for i in o.strip()[1:-1].replace(',', '').split()] for o in hazard]
    hazard

    trans = config['labtrans']
    events = np.array((merge_df['event']))
    gt_idx, _ = trans.transform(merge_df['gt_sv'], events)
    pred_idx, _ = trans.transform(merge_df['pred_sv'], events)
    ci_val = concordance_index(gt_idx, pred_idx, event_observed=events)

    config['out'] = {
        'MAE': mae_non_censoring_only(merge_df),
        'MSE': mse_non_censoring_only(merge_df),
        'default_mae_mse': default_mae_mse(merge_df),
        'CI': ci_val
    }

    return config


def surv2interpolation(s, sub=10):
    """ s tensor [bs,out_dim]
        see `pycox.models.interpolation.InterpolateDiscrete._surv_const_pdf`
    """
    n, m = s.shape
    device = s.device
    diff = (s[:, 1:] - s[:, :-1]).contiguous().view(-1, 1).repeat(1, sub).view(n, -1)
    rho = torch.linspace(0, 1, sub + 1, device=device)[:-1].contiguous().repeat(n, m - 1)
    s_prev = s[:, :-1].contiguous().view(-1, 1).repeat(1, sub).view(n, -1)
    surv = torch.zeros(n, int((m - 1) * sub + 1))
    surv[:, :-1] = diff * rho + s_prev
    surv[:, -1] = s[:, -1]
    return surv.cpu().detach()


def mae_non_censoring_only(df):
    """
    dataframe with ground truth and predict with fixed names
    """
    ae = abs(df['gt_sv'] - df['pred_sv'])
    return sum(ae * df['event']) / sum(df['event'])


def mse_non_censoring_only(df):
    """
    dataframe with ground truth and predict with fixed names
    """
    se = (df['gt_sv'] - df['pred_sv']) * (df['gt_sv'] - df['pred_sv'])
    ret = sum(se * df['event']) / sum(df['event'])
    ret = math.sqrt(ret)
    return ret


def default_mae_mse(df):
    mean = sum(df['gt_sv'] * df['event']) / sum(df['event'])

    # mse
    se = (df['gt_sv'] - mean) * (df['gt_sv'] - mean)
    mse = sum(se * df['event']) / sum(df['event'])

    # mae
    ae = abs(df['gt_sv'] - mean)

    ret = sum(ae * df['event']) / sum(df['event']), math.sqrt(mse)

    return ret


def report_file_to_df(data_lines, lines=None, labels=None):
    """
    From report file to dataframe.
    Usage:
        df_mv = report_file_to_df(lines, (102, 112), labels=['', 'x', '', '','','CI'])
        df_mv['CI'] = pd.to_numeric(df_mv['CI'])
        ...

    :param data_lines: read line by line from report, separated by \t
    :param lines: pair start and stop line count from zero
    :param labels: list
    :return: 
    """
    s, e = lines if lines is not None else (0, -1)
    selected_lines = data_lines[s:e]
    selected_lines = [o.strip().split('\t') for o in selected_lines]

    df = pd.DataFrame(selected_lines)

    if labels is not None: df.columns = labels
    return df
