"""
Implement some function related to survival analysis
"""
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtuples as tt
from lifelines.utils import concordance_index
from pycox.evaluation import EvalSurv
from pycox.models import LogisticHazard, CoxCC
from pycox.models.loss import NLLLogistiHazardLoss
from pycox.preprocessing.label_transforms import LabTransDiscreteTime
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper

from prlab.model.vae import MultiDecoderVAE
from prlab_medical.data_loader import data_loader_to_df_mix
from prlab_medical.data_loader import event_norm

from pycox.models import CoxTime, CoxPH
from pycox.models.cox_time import MLPVanillaCoxTime


class SurvFromMVAE(nn.Module):
    """
    from MVAE to get only surv (phi)
    This predict version support for only single x instead x_cat, x_cont
    Data input are in form of [cat,..., cat, cont, cont]
    Should separated cat anc cont before pass to model.
    To adapt with `pycox.models.logistic_hazard.LogisticHazard` and input from dataframe.
    Note: order of the input should be fix as above
    """

    def __init__(self, mvae_net, **kwargs):
        super(SurvFromMVAE, self).__init__()

        self.mvae_net = mvae_net
        self.n_cat = len(kwargs['cat_names'])

    def predict(self, x, **kwargs):
        x_cat, x_cont = x[:, :self.n_cat], x[:, self.n_cat:]
        x_cat = x_cat.long()

        store_mode = self.mvae_net.output_mode
        self.mvae_net.output_mode = MultiDecoderVAE.TEST_MODE
        ret = self.mvae_net(x_cat, x_cont, **kwargs)
        phi = ret[1][0]
        self.mvae_net.output_mode = store_mode
        phi = phi.contiguous()
        return phi
        # return self.mvae_net.forward(x_cat, x_cont, **kwargs)

    def __repr__(self):
        return f"SurvFromMVAE ( {str(self.mvae_net)} )"


class SurvFromDNN(nn.Module):
    def __init__(self, dnn, **kwargs):
        super(SurvFromDNN, self).__init__()
        self.dnn = dnn
        self.sep = kwargs.get('dnn_output_sep', [-1])
        self.n_cat = len(kwargs['cat_names'])

    def predict(self, x, **kwargs):
        x_cat, x_cont = x[:, :self.n_cat], x[:, self.n_cat:]
        x_cat = x_cat.long()

        ret = self.dnn(x_cat, x_cont, **kwargs)
        return ret[:, :self.sep[0]].contiguous()


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


class MSELossFilter(nn.Module):
    """
    see `MSELossE`
    Filter of mse, target = [bs, 2], the second value is filter 1/0
    Using in censoring data.
    If data is non-censoring => 1 => count
    if data is censoring => 0 => count if right-censoring and flag and pred < taget
    """

    def __init__(self, **kwargs):
        super(MSELossFilter, self).__init__()
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
    Note: same meaning of phi with LossLogHazInd (raw of hazard, before sigmoid)
    surv (S) and pmf (f) are inference from hazard
    """

    def __init__(self, alpha=0.5, **kwargs):
        super().__init__(alpha=alpha, **kwargs)

        self.loss_ind = MSELossFilter(**kwargs)

        cuts = kwargs['cuts']  # using fixed cuts values
        self.cuts = cuts  # 0: c0-c1, 1: c1-c2, ..., i: ci-c(i+1)...
        self.mid_values = [0.5 * (cuts[i] + cuts[i + 1]) for i in range(len(cuts) - 1)]
        self.mid_values.append(cuts[-1] + 0.5 * (cuts[-1] - cuts[-2]))  # fixed last value is base on previous
        self.mid_values = torch.tensor(self.mid_values)

    def forward(self, phi, target_loghaz):
        phi, ind_sv = phi[:, :-1], phi[:, -1]
        device = phi.device

        idx_durations, events = target_loghaz[:, 0], target_loghaz[:, 1]
        idx_durations = idx_durations.type(torch.int64)
        t_ind_sv, t_org_event = target_loghaz[:, 2], target_loghaz[:, 3]

        h_j, s_j, f_j = hsf_from_phi(phi=phi)

        # expected survival time from f_j: \sum(f_j * mid_values_j)
        values_dev = self.mid_values.to(device=device)  # move to correct device
        ind_sv_e = torch.sum(f_j * values_dev, dim=-1)  # bs

        loss_surv = self.loss_surv(phi, idx_durations, events)
        loss_surv_ce = None  # TODO CE(f_j, idx_durations)

        # mse only for event and/or right-censoring
        loss_mse = self.loss_ind(ind_sv, target=target_loghaz[2:])
        loss_mse_e = self.loss_ind(ind_sv_e, target=target_loghaz[2:])

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


def cox_based_pipe(**config):
    """
    Training with CoxTime or CoxPh
    :param config:
    :return:
    """
    df_train, df_val, df_test = config['train_df'], config['valid_df'], config['test_df']

    def is_from_cat(name):
        for cat in config['cat_names']:
            if cat in name:
                return True

        return False

    cols_standardize = config['cont_names']
    cols_leave = [o for o in df_train.columns if is_from_cat(o)]  # from cat to multi-column

    is_normalize_cont = config.get('is_normalize_cont', False)
    standardize = [([col], StandardScaler() if is_normalize_cont else None) for col in cols_standardize]
    leave = [(col, None) for col in cols_leave]

    x_mapper = DataFrameMapper(standardize + leave)

    x_train = x_mapper.fit_transform(df_train).astype('float32')
    x_val = x_mapper.transform(df_val).astype('float32')
    x_test = x_mapper.transform(df_test).astype('float32')

    labtrans = CoxTime.label_transform()

    # get_target = lambda df: (df['Survival.time'].values, df['Deadstatus.event'].values)
    def get_target1(df):
        df[df['Deadstatus.event'] == 9] = 0
        return df['Survival.time'].values, df['Deadstatus.event'].values

    y_train = labtrans.fit_transform(*get_target1(df_train))
    y_val = labtrans.transform(*get_target1(df_val))
    durations_test, events_test = get_target1(df_test)
    val = tt.tuplefy(x_val, y_val)

    # there are two model to get, CoxTime and CoxPH
    in_features = x_train.shape[1]
    out_features = config.get('out_features', 1)
    num_nodes = config.get('layers', [128, 128])
    batch_norm = config.get('batch_norm', False)
    dropout = config.get('dropout', 0.1)
    output_bias = config.get('output_bias', False)
    if config['model'] == 'CoxTime':
        net = MLPVanillaCoxTime(in_features, num_nodes, batch_norm, dropout)
        model = CoxTime(net, tt.optim.Adam, labtrans=labtrans)

    elif config['model'] == 'CoxCC':
        net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm,
                                      dropout, output_bias=output_bias)
        model = CoxCC(net, tt.optim.Adam)
        
    else:
        # CoxPH here
        net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm,
                                      dropout, output_bias=output_bias)
        model = CoxPH(net, tt.optim.Adam)

    # lrfinder = model.lr_finder(x_train, y_train, config['bs'], tolerance=2)
    # _ = lrfinder.plot()
    # lrfinder.get_best_lr()

    model.optimizer.set_lr(config.get('lr', 1e-2))

    callbacks = [tt.callbacks.EarlyStopping()]
    verbose = True

    model.fit(x_train, y_train, config['bs'], config['epochs'], callbacks, verbose,
              val_data=val.repeat(10).cat())

    # _ = log.plot()
    model.partial_log_likelihood(*val).mean()
    _ = model.compute_baseline_hazards()

    surv = model.predict_surv_df(x_test)
    durations_test, events_test = durations_test.astype(np.int32), events_test.astype(np.int32)

    config.update({
        'surv': surv,
        'durations_test': durations_test,
        'events_test': events_test
    })

    return report_from_surv(**config)


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
    ele = batch_tensor[0].cpu().detach()
    # hazards = torch.sigmoid_(ele[:, :-1]).numpy().tolist()
    phi = ele[:, :-1]
    ind_st = ele[:, -1].numpy().tolist()

    # expected survival time
    ind_sv_e = expectation_of_life(phi=phi, **config).numpy().tolist()

    phi = phi.numpy().tolist()
    x = zip(phi, ind_st, ind_sv_e)

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
    ind_st = batch_tensor[1].cpu().detach()
    phi = batch_tensor[0].cpu().detach()

    # expected survival time
    ind_sv_e = expectation_of_life(phi, **config).numpy().tolist()

    ind_st = ind_st.numpy().tolist()
    phi = phi.numpy().tolist()
    x = zip(phi, ind_st, ind_sv_e)

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


def report_from_dnn(**config):
    model = LogisticHazardE(
        net=SurvFromDNN(dnn=config['model'], **config),
        optimizer=tt.optim.Adam(0.01),
        duration_index=config['labtrans'].cuts,
        loss=NLLLogistiHazardLoss()
    )

    out = report_from_logistic_hazard(**{**config, 'model': model})
    config['out'] = out['out']
    return config


def report_from_mvae(**config):
    # this model for report only, infer from MVAE
    model = LogisticHazardE(
        net=SurvFromMVAE(mvae_net=config['model'], **config),
        optimizer=tt.optim.Adam(0.01),
        duration_index=config['labtrans'].cuts,
        loss=NLLLogistiHazardLoss()
    )

    out = report_from_logistic_hazard(**{**config, 'model': model})
    config['out'] = out['out']
    return config


def report_from_logistic_hazard(**config):
    """
    See more about report: https://github.com/havakv/pycox/blob/master/examples/03_network_architectures.ipynb
    :param config:
    :return:
    """
    # get report for some basic info MAE, MSE, ...
    xout = report_survival_time(**config)['out']

    model = config['model']
    df_test = data_loader_to_df_mix(config['data_test'], **config)

    # custom header for our task, TODO fix hard code number of cat and cont
    cols_leave = [f"cat_{i}" for i in range(5)] + [f"cont_{i}" for i in range(6)]
    cols_standardize = []

    standardize = [([col], StandardScaler()) for col in cols_standardize]
    leave = [(col, None) for col in cols_leave]

    x_mapper = DataFrameMapper(standardize + leave)

    # we do not need it and it does not affect results
    x_mapper.fit_transform(df_test).astype('float32')  # x_train
    x_test = x_mapper.transform(df_test).astype('float32')

    durations_test, events_test = df_test['label_other_0'].values, df_test['event'].values

    surv = model.interpolate(10).predict_surv_df(x_test)

    out = report_from_surv(**{
        **config,
        'surv': surv,
        'durations_test': durations_test,
        'events_test': events_test
    })['out']

    xout.update(out)
    config['out'] = xout
    return config


def report_from_surv(**config):
    """
    Report from surv predicted and test data in *_test
    :param config:
    :return:
    """
    # should pass from outside
    surv = config['surv']
    durations_test, events_test = config['durations_test'], config['events_test']

    # draw below graph to file and make value to configure or file
    sample_case_ids = config.get('sample_case_ids', [37, 29, 0, 9])
    cp = Path(config.get('cp', '.'))
    surv.iloc[:, sample_case_ids].plot(drawstyle='steps-post')
    plt.ylabel('S(t | x)')
    _ = plt.xlabel('Time')
    plt.savefig(cp / 'sample_cases.pdf', transparent=True, bbox_inches='tight', pad_inches=0)
    plt.clf()

    out = {'MAE': 0, 'MSE': 0, 'default_mae_mse': 0}
    # evaluateion
    ev = EvalSurv(surv, durations_test, events_test, censor_surv='km')
    out['CI'] = ev.concordance_td('antolini')

    # Brier Score
    time_grid = np.linspace(durations_test.min(), durations_test.max(), 100)
    ev.brier_score(time_grid).plot()
    plt.ylabel('Brier score'), plt.xlabel('Time')
    plt.savefig(cp / 'Brier-score.pdf', transparent=True, bbox_inches='tight', pad_inches=0)
    plt.clf()

    # Negative binomial log-likelihood
    ev.nbll(time_grid).plot()
    plt.ylabel('NBLL'), plt.xlabel('Time')
    plt.savefig(cp / 'NBLL.pdf', transparent=True, bbox_inches='tight', pad_inches=0)

    # Integrated scores
    out['brier_score'] = ev.integrated_brier_score(time_grid)
    out['nbll'] = ev.integrated_nbll(time_grid)

    config['out'] = out
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
    return surv


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


def hsf_from_phi(phi, eps=1e-7):
    """
    :param phi: phi {torch.tensor} -- Estimates in (-inf, inf), where hazard = sigmoid(phi).
    :param eps:
    :return: hazard, survival function and pmf
    """
    h_j = torch.sigmoid(phi)
    s_j = (1 - h_j).add(eps).log().cumsum(1).exp()
    f_j = h_j * s_j
    return h_j, s_j, f_j


def expectation_of_life(phi, **config):
    """
    expectation of life from f_j (or s_j)
    :param phi: phi {torch.tensor} -- Estimates in (-inf, inf), where hazard = sigmoid(phi).
    :param config:
    :return: tensor [bs, 1] or [bs]
    """
    h_j, s_j, f_j = hsf_from_phi(phi=phi)
    dev = phi.device

    mid_values = config.get('mid_values')
    if mid_values is None:
        xtmp = config['cuts'][-1] * 2 - config['cuts'][-2]
        mid_values = list((np.array(config['cuts']) + np.array(config['cuts'][1:] + [xtmp])) / 2)

    mid_values = torch.tensor(mid_values, device=dev)

    exp = torch.sum(f_j * mid_values, dim=-1)  # bs
    return exp
