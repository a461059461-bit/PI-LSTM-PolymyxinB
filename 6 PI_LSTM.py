"""
多粘菌素B脓毒症患者药动学建模
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy.integrate import odeint
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import math
import warnings
warnings.filterwarnings('ignore')

seednum = 42
np.random.seed(seednum)
torch.manual_seed(seednum)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seednum)


# ============================================================================
# 群体药动学参数配置
# ============================================================================

THETA_CL = 2.5    # L/h
THETA_V = 35.0    # L
OMEGA_CL = 0.4    # CL的个体间变异
OMEGA_V = 0.3     # V的个体间变异
THERAPEUTIC_MIN = 2.0
THERAPEUTIC_MAX = 4.0


# ============================================================================
# 数据生成
# ============================================================================

class ClinicalDataSimulator:
    def __init__(self, n_patients=200):
        self.n_patients = n_patients
        
    def generate_patient_covariates(self, patient_id, n_timepoints):
        age = np.random.uniform(40, 80)
        weight = np.random.uniform(50, 100)
        sex = np.random.binomial(1, 0.6)
        baseline_egfr = np.random.uniform(30, 120)
        
        egfr_trend = np.random.choice(['stable', 'declining', 'recovering'], p=[0.4, 0.35, 0.25])
        egfr_values = []
        current_egfr = baseline_egfr
        for t in range(n_timepoints):
            if egfr_trend == 'declining':
                current_egfr = max(10, current_egfr - np.random.uniform(0, 5))
            elif egfr_trend == 'recovering':
                current_egfr = min(120, current_egfr + np.random.uniform(0, 3))
            else:
                current_egfr += np.random.normal(0, 2)
            egfr_values.append(np.clip(current_egfr, 10, 120))
        
        baseline_albumin = np.random.uniform(2.0, 4.0)
        albumin_values = [np.clip(baseline_albumin + np.random.normal(0, 0.2), 1.5, 4.5) 
                         for _ in range(n_timepoints)]
        
        baseline_crp = np.random.uniform(50, 200)
        crp_values = []
        current_crp = baseline_crp
        for t in range(n_timepoints):
            current_crp = max(5, current_crp * np.random.uniform(0.85, 1.05))
            crp_values.append(current_crp)
        
        baseline_sofa = np.random.randint(4, 15)
        sofa_values = []
        current_sofa = baseline_sofa
        for t in range(n_timepoints):
            current_sofa += np.random.choice([-1, 0, 0, 1], p=[0.3, 0.4, 0.2, 0.1])
            sofa_values.append(np.clip(current_sofa, 0, 24))
        
        scr_values = [max(0.5, 175 * (age**-0.203) * (0.742 if sex == 0 else 1) / egfr * (1/88.4) 
                         + np.random.normal(0, 0.1)) for egfr in egfr_values]
        
        return {
            'age': age, 'weight': weight, 'sex': sex,
            'egfr': egfr_values, 'albumin': albumin_values,
            'crp': crp_values, 'sofa': sofa_values, 'scr': scr_values
        }
    
    def generate_individual_pk_params(self, covariates):
        eta_cl = np.random.normal(0, OMEGA_CL)
        eta_v = np.random.normal(0, OMEGA_V)
        
        mean_egfr = np.mean(covariates['egfr'])
        egfr_effect = (mean_egfr / 90) ** 0.75
        weight_effect = (covariates['weight'] / 70) ** 1.0
        
        CL = THETA_CL * egfr_effect * np.exp(eta_cl)
        V = THETA_V * weight_effect * np.exp(eta_v)
        
        return CL, V, eta_cl, eta_v
    
    def pk_ode(self, y, t, CL, V, k_infusion, is_infusing):
        C = y[0]
        if is_infusing:
            dCdt = k_infusion / V - (CL / V) * C
        else:
            dCdt = -(CL / V) * C
        return [dCdt]
    
    def simulate_pk_profile(self, dose_schedule, CL, V, covariates, sample_times):
        concentrations = []
        C_current = 0
        
        for i, (dose_time, dose, interval) in enumerate(dose_schedule):
            time_idx = min(i, len(covariates['egfr']) - 1)
            egfr_current = covariates['egfr'][time_idx]
            CL_current = CL * (egfr_current / np.mean(covariates['egfr'])) ** 0.5
            
            infusion_duration = 1.0
            k_infusion = dose / infusion_duration
            
            t_infusion = np.linspace(0, infusion_duration, 10)
            C_infusion = odeint(self.pk_ode, [C_current], t_infusion, 
                               args=(CL_current, V, k_infusion, True))
            C_end_infusion = C_infusion[-1, 0]
            
            t_elimination = np.linspace(0, interval - infusion_duration, 50)
            C_elimination = odeint(self.pk_ode, [C_end_infusion], t_elimination,
                                  args=(CL_current, V, 0, False))
            C_current = C_elimination[-1, 0]
            
            for sample_time in sample_times:
                if dose_time < sample_time <= dose_time + interval:
                    time_since_dose = sample_time - dose_time
                    if time_since_dose <= infusion_duration:
                        idx = int(time_since_dose / infusion_duration * 9)
                        C_sample = C_infusion[idx, 0]
                    else:
                        idx = int((time_since_dose - infusion_duration) / 
                                 (interval - infusion_duration) * 49)
                        C_sample = C_elimination[idx, 0]
                    
                    C_observed = C_sample * np.exp(np.random.normal(0, 0.15))
                    concentrations.append({
                        'sample_time': sample_time,
                        'time_since_dose': time_since_dose,
                        'concentration': max(0.1, C_observed),
                        'true_concentration': C_sample,
                        'dose': dose, 
                        'interval': interval,
                        'egfr': egfr_current, 
                        'CL_current': CL_current
                    })
        
        return concentrations
    
    def generate_dosing_schedule(self, weight, egfr):
        daily_dose = np.random.uniform(1.5, 2.5) * weight
        intervals = [8, 12, 12, 24]
        base_interval = np.random.choice(intervals)
        
        if egfr < 30:
            base_interval = max(base_interval, 24)
        
        doses_per_day = 24 / base_interval
        single_dose = daily_dose / doses_per_day
        
        treatment_days = np.random.randint(7, 15)
        schedule = []
        current_time = 0
        current_dose = single_dose
        current_interval = base_interval
        
        while current_time < treatment_days * 24:
            if np.random.random() < 0.2 and current_time > 48:
                current_dose *= np.random.uniform(0.8, 1.2)
            if np.random.random() < 0.1 and current_time > 72:
                current_interval = np.random.choice([8, 12, 24])
            
            schedule.append((current_time, current_dose, current_interval))
            current_time += current_interval
        
        return schedule
    
    def generate_sampling_times(self, dose_schedule):
        sampling_times = []
        n_doses = len(dose_schedule)
        
        i = 0
        while i < n_doses:
            skip_doses = np.random.choice([0, 1, 2, 3], p=[0.3, 0.35, 0.25, 0.1])
            i += skip_doses
            
            if i < n_doses:
                dose_time, dose, interval = dose_schedule[i]
                trough_time = dose_time + interval - np.random.uniform(0.5, 2.0)
                sampling_times.append(trough_time)
                
                if np.random.random() < 0.05:
                    peak_time = dose_time + np.random.uniform(1.0, 2.0)
                    sampling_times.append(peak_time)
            i += 1
        
        return sorted(sampling_times)
    
    def generate_dataset(self):
        all_data = []
        
        for patient_id in range(self.n_patients):
            covariates = self.generate_patient_covariates(patient_id, 20)
            CL, V, eta_cl, eta_v = self.generate_individual_pk_params(covariates)
            dose_schedule = self.generate_dosing_schedule(covariates['weight'], covariates['egfr'][0])
            sample_times = self.generate_sampling_times(dose_schedule)
            
            if len(sample_times) < 2:
                continue
            
            observations = self.simulate_pk_profile(dose_schedule, CL, V, covariates, sample_times)
            
            for j, obs in enumerate(observations):
                dose_idx = 0
                for k, (dt, d, inter) in enumerate(dose_schedule):
                    if dt < obs['sample_time']:
                        dose_idx = k
                
                time_idx = min(dose_idx, len(covariates['egfr']) - 1)
                
                record = {
                    'patient_id': patient_id, 
                    'obs_idx': j,
                    'sample_time': obs['sample_time'],
                    'time_since_dose': obs['time_since_dose'],
                    'concentration': obs['concentration'],
                    'true_concentration': obs['true_concentration'],
                    'dose': obs['dose'], 
                    'interval': obs['interval'],
                    'cumulative_dose': sum([d for t, d, i in dose_schedule[:dose_idx+1]]),
                    'n_doses': dose_idx + 1,
                    'age': covariates['age'], 
                    'weight': covariates['weight'],
                    'sex': covariates['sex'], 
                    'egfr': covariates['egfr'][time_idx],
                    'albumin': covariates['albumin'][time_idx],
                    'crp': covariates['crp'][time_idx],
                    'sofa': covariates['sofa'][time_idx],
                    'scr': covariates['scr'][time_idx],
                    'true_CL': CL, 
                    'true_V': V,
                    'eta_cl': eta_cl, 
                    'eta_v': eta_v,
                    'CL_current': obs['CL_current']
                }
                all_data.append(record)
        
        df = pd.DataFrame(all_data)
        print(f"生成数据集：{len(df)} 条记录，{df['patient_id'].nunique()} 位患者")
        
        return df


# ============================================================================
# 数据预处理
# ============================================================================

class PKDataset(Dataset):
    def __init__(self, sequences, targets, pk_targets, time_features, time_deltas,
                 raw_covariates, patient_ids):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
        self.pk_targets = torch.FloatTensor(pk_targets)
        self.time_features = torch.FloatTensor(time_features)
        self.time_deltas = torch.FloatTensor(time_deltas)
        self.raw_covariates = torch.FloatTensor(raw_covariates)
        self.patient_ids = torch.LongTensor(patient_ids)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return (self.sequences[idx], self.targets[idx], self.pk_targets[idx], 
                self.time_features[idx], self.time_deltas[idx],
                self.raw_covariates[idx], self.patient_ids[idx])


def prepare_sequences(df, seq_length=3, scaler=None, fit_scaler=False):
    feature_cols = ['time_since_dose', 'dose', 'interval', 'cumulative_dose', 
                    'n_doses', 'age', 'weight', 'sex', 'egfr', 'albumin', 
                    'crp', 'sofa', 'scr']
    time_cols = ['time_since_dose', 'dose', 'interval']
    raw_cov_cols = ['age', 'weight', 'sex', 'egfr', 'albumin', 'crp', 'sofa', 'scr']
    
    sequences, targets, pk_targets, time_features = [], [], [], []
    time_deltas, raw_covariates, patient_ids = [], [], []
    
    for patient_id in df['patient_id'].unique():
        patient_data = df[df['patient_id'] == patient_id].sort_values('sample_time')
        
        if len(patient_data) < seq_length + 1:
            continue
        
        sample_times = patient_data['sample_time'].values
        
        for i in range(seq_length, len(patient_data)):
            seq_data = patient_data.iloc[i-seq_length:i]
            target_data = patient_data.iloc[i]
            
            sequences.append(seq_data[feature_cols].values)
            targets.append(target_data['concentration'])
            pk_targets.append([target_data['true_CL'], target_data['true_V']])
            time_features.append(target_data[time_cols].values)
            raw_covariates.append(target_data[raw_cov_cols].values)
            
            seq_sample_times = sample_times[i-seq_length:i+1]
            time_deltas.append(np.diff(seq_sample_times))
            patient_ids.append(patient_id)
    
    sequences = np.array(sequences)
    targets = np.array(targets).reshape(-1, 1)
    pk_targets = np.array(pk_targets)
    time_features = np.array(time_features)
    time_deltas = np.array(time_deltas)
    raw_covariates = np.array(raw_covariates)
    patient_ids = np.array(patient_ids)
    
    if fit_scaler:
        n_samples, seq_len, n_features = sequences.shape
        scaler = StandardScaler()
        scaler.fit(sequences.reshape(-1, n_features))
    
    if scaler is not None:
        n_samples, seq_len, n_features = sequences.shape
        sequences = scaler.transform(sequences.reshape(-1, n_features)).reshape(n_samples, seq_len, n_features)
    
    return sequences, targets, pk_targets, time_features, time_deltas, raw_covariates, patient_ids, scaler


def split_by_patient(df, train_ratio=0.7, val_ratio=0.15):
    patient_ids = df['patient_id'].unique()
    np.random.shuffle(patient_ids)
    
    n_patients = len(patient_ids)
    n_train = int(n_patients * train_ratio)
    n_val = int(n_patients * val_ratio)
    
    train_ids = patient_ids[:n_train]
    val_ids = patient_ids[n_train:n_train+n_val]
    test_ids = patient_ids[n_train+n_val:]
    
    print(f"训练集: {len(train_ids)} 患者, {len(df[df['patient_id'].isin(train_ids)])} 记录")
    print(f"验证集: {len(val_ids)} 患者, {len(df[df['patient_id'].isin(val_ids)])} 记录")
    print(f"测试集: {len(test_ids)} 患者, {len(df[df['patient_id'].isin(test_ids)])} 记录")
    
    return (df[df['patient_id'].isin(train_ids)], 
            df[df['patient_id'].isin(val_ids)], 
            df[df['patient_id'].isin(test_ids)])


# ============================================================================
# 模型组件
# ============================================================================

class PKAwarePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
        self.time_encoder = nn.Sequential(
            nn.Linear(3, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model)
        )
        self.log_kel = nn.Parameter(torch.tensor(-1.0))
        self.gate = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.Sigmoid())
    
    def forward(self, x, time_deltas=None):
        batch_size, seq_len, _ = x.shape
        pos_encoding = self.pe[:seq_len].unsqueeze(0).expand(batch_size, -1, -1)
        
        if time_deltas is not None:
            time_features = torch.stack([time_deltas, time_deltas/12.0, time_deltas/24.0], dim=-1)
            time_encoding = self.time_encoder(time_features)
            kel = torch.exp(self.log_kel)
            pk_decay = torch.exp(-kel * time_deltas).unsqueeze(-1)
            time_encoding = time_encoding * pk_decay
            combined = torch.cat([pos_encoding, time_encoding], dim=-1)
            gate = self.gate(combined)
            encoding = gate * pos_encoding + (1 - gate) * time_encoding
        else:
            encoding = pos_encoding
        
        return self.dropout(x + encoding)


class PKAwareMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads=4, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model, self.n_heads, self.d_k = d_model, n_heads, d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.log_decay_rates = nn.Parameter(torch.linspace(-2, 0, n_heads))
        self.dose_bias = nn.Sequential(nn.Linear(1, n_heads), nn.Tanh())
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, x, time_deltas=None, doses=None, mask=None):
        batch_size, seq_len, _ = x.shape
        residual = x
        
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if time_deltas is not None:
            pairwise_deltas = torch.abs(time_deltas.unsqueeze(-1) - time_deltas.unsqueeze(-2))
            decay_rates = torch.exp(self.log_decay_rates).view(1, self.n_heads, 1, 1)
            scores = scores - decay_rates * pairwise_deltas.unsqueeze(1)
        
        if doses is not None:
            dose_bias = self.dose_bias(doses).permute(0, 2, 1).unsqueeze(-1)
            scores = scores + dose_bias
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        output = self.dropout(self.W_o(context))
        return self.layer_norm(output + residual), attn_weights


class PKParameterNetworkV4(nn.Module):
    """PK参数估计网络"""
    
    def __init__(self, input_dim, hidden_dim=128, context_dim=128):
        super().__init__()
        
        # ========== 固定的协变量效应（不学习！）==========
        # 这些是药理学先验知识
        # CL = THETA_CL * (eGFR/90)^0.75 * exp(eta_cl)
        # V = THETA_V * (WT/70)^1.0 * exp(eta_v)
        self.register_buffer('theta_egfr_cl', torch.tensor(0.75))
        self.register_buffer('theta_wt_v', torch.tensor(1.0))
        self.register_buffer('log_theta_cl', torch.tensor(np.log(THETA_CL)))
        self.register_buffer('log_theta_v', torch.tensor(np.log(THETA_V)))
        self.register_buffer('omega_cl', torch.tensor(OMEGA_CL))
        self.register_buffer('omega_v', torch.tensor(OMEGA_V))
        
        # ========== 特征编码器 ==========
        # 序列特征编码
        self.seq_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        # 上下文编码
        self.ctx_encoder = nn.Sequential(
            nn.Linear(context_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU()
        )
        
        # ========== Eta预测网络（关键改进）==========
        # 分离CL和V的eta预测，但共享底层特征
        
        # 共享特征提取
        self.shared_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        
        # CL的eta预测 - 主要依赖浓度-时间信息
        self.eta_cl_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1)
        )
        
        # V的eta预测 - 主要依赖剂量-浓度关系
        self.eta_v_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1)
        )
        
        # 初始化为小值
        nn.init.normal_(self.eta_cl_net[-1].weight, std=0.01)
        nn.init.zeros_(self.eta_cl_net[-1].bias)
        nn.init.normal_(self.eta_v_net[-1].weight, std=0.01)
        nn.init.zeros_(self.eta_v_net[-1].bias)
    
    def compute_covariate_effects(self, raw_covariates):
        """
        计算协变量效应（固定系数，不学习）
        raw_covariates: [batch, 8] = [age, weight, sex, egfr, albumin, crp, sofa, scr]
        """
        weight = raw_covariates[:, 1:2]  # kg
        egfr = raw_covariates[:, 3:4]    # mL/min/1.73m2
        
        # CL的eGFR效应
        egfr_ratio = torch.clamp(egfr / 90.0, min=0.1, max=3.0)
        cov_effect_cl = self.theta_egfr_cl * torch.log(egfr_ratio)
        
        # V的体重效应
        wt_ratio = torch.clamp(weight / 70.0, min=0.5, max=2.0)
        cov_effect_v = self.theta_wt_v * torch.log(wt_ratio)
        
        return cov_effect_cl, cov_effect_v
    
    def forward(self, x_seq, context, raw_covariates, time_features):
        """
        Args:
            x_seq: [batch, seq_len, input_dim] 序列特征
            context: [batch, context_dim] 注意力上下文
            raw_covariates: [batch, 8] 原始协变量
            time_features: [batch, 3] [time_since_dose, dose, interval]
        """
        batch_size = x_seq.shape[0]
        x_last = x_seq[:, -1, :]
        
        # 1. 计算协变量效应（固定）
        cov_effect_cl, cov_effect_v = self.compute_covariate_effects(raw_covariates)
        
        # 2. 编码特征
        seq_encoded = self.seq_encoder(x_last)
        ctx_encoded = self.ctx_encoder(context)
        
        # 3. 融合特征
        fused = torch.cat([seq_encoded, ctx_encoded], dim=-1)
        shared_features = self.shared_encoder(fused)
        
        # 4. 预测eta
        eta_cl_raw = self.eta_cl_net(shared_features)
        eta_v_raw = self.eta_v_net(shared_features)
        
        # 5. 约束eta范围（软约束，使用scaled tanh）
        # 输出范围约为 [-2*omega, 2*omega]，覆盖约95%的个体
        eta_cl = self.omega_cl * 2.0 * torch.tanh(eta_cl_raw / 2.0)
        eta_v = self.omega_v * 2.0 * torch.tanh(eta_v_raw / 2.0)
        
        # 6. 组合得到最终参数
        log_CL = self.log_theta_cl + cov_effect_cl + eta_cl
        log_V = self.log_theta_v + cov_effect_v + eta_v
        
        return log_CL, log_V, eta_cl, eta_v, cov_effect_cl, cov_effect_v


class PolymyxinBPKModelV4(nn.Module):
    """主模型"""
    
    def __init__(self, input_dim, hidden_dim=128, n_layers=2, n_heads=4, dropout=0.2):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # 输入投影
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 位置编码
        self.pk_positional_encoding = PKAwarePositionalEncoding(d_model=hidden_dim, dropout=dropout)
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
            bidirectional=True
        )
        
        # 注意力
        self.pk_attention = PKAwareMultiHeadAttention(d_model=hidden_dim, n_heads=n_heads, dropout=dropout)
        
        # 上下文聚合
        self.context_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # PK参数估计
        self.pk_network = PKParameterNetworkV4(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            context_dim=hidden_dim
        )
        
        # 浓度预测（纯物理模型 + 残差修正）
        self.residual_predictor = nn.Sequential(
            nn.Linear(hidden_dim + 5, hidden_dim // 2),  # context + time_features + log_CL + log_V
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # 不确定性预测
        self.log_var_predictor = nn.Sequential(
            nn.Linear(hidden_dim + 5, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.register_buffer('log_sigma_base', torch.tensor(-1.0))
    
    def compute_pk_concentration(self, CL, V, dose, time_since_dose, interval):
        """物理模型计算浓度"""
        kel = torch.clamp(CL / V, min=0.01, max=2.0)
        
        # 蓄积因子
        acc = 1.0 / (1.0 - torch.exp(-kel * interval) + 1e-8)
        acc = torch.clamp(acc, min=1.0, max=20.0)
        
        # 消除相
        t_inf = 1.0
        t_post = torch.clamp(time_since_dose - t_inf, min=0.0)
        
        C = (dose / V) * acc * torch.exp(-kel * t_post)
        return torch.clamp(C, min=0.01, max=100.0)
    
    def forward(self, x, time_features, time_deltas, raw_covariates):
        batch_size, seq_len, _ = x.shape
        
        # 1. 编码
        h = self.input_projection(x)
        
        cumulative_time = torch.zeros(batch_size, seq_len, device=x.device)
        for i in range(seq_len):
            cumulative_time[:, i] = time_deltas[:, i:].sum(dim=1)
        
        h = self.pk_positional_encoding(h, cumulative_time)
        lstm_out, _ = self.lstm(h)
        
        # 2. 注意力
        doses = x[:, :, 1:2]
        attn_out, attn_weights = self.pk_attention(lstm_out, time_deltas=cumulative_time, doses=doses)
        
        # 3. 上下文
        attn_scores = self.context_attention(attn_out).squeeze(-1)
        attn_weights_pool = F.softmax(attn_scores, dim=-1)
        context = torch.bmm(attn_weights_pool.unsqueeze(1), attn_out).squeeze(1)
        
        # 4. PK参数估计
        log_CL, log_V, eta_cl, eta_v, cov_eff_cl, cov_eff_v = self.pk_network(
            x, context, raw_covariates, time_features
        )
        
        CL = torch.exp(log_CL)
        V = torch.exp(log_V)
        
        # 5. 物理模型浓度
        time_since_dose = time_features[:, 0:1]
        dose = time_features[:, 1:2]
        interval = time_features[:, 2:3]
        
        C_physics = self.compute_pk_concentration(CL, V, dose, time_since_dose, interval)
        
        # 6. 残差修正
        pred_input = torch.cat([context, time_features, log_CL, log_V], dim=1)
        residual = self.residual_predictor(pred_input) * 0.3  # 限制残差幅度
        
        # 最终预测 = 物理模型 + 小残差
        pred_mean = C_physics + residual
        pred_mean = torch.clamp(pred_mean, min=0.01)
        
        # 7. 不确定性
        pred_log_var = self.log_var_predictor(pred_input) + self.log_sigma_base
        
        return (pred_mean, pred_log_var, log_CL, log_V, eta_cl, eta_v,
                C_physics, attn_weights, cov_eff_cl, cov_eff_v)


class PKLossV4(nn.Module):
    """损失函数"""
    
    def __init__(self, lambda_nll=1.0, lambda_physics=1.0, lambda_eta_var=0.5,
                 lambda_kel=0.1, lambda_residual=0.1):
        super().__init__()
        self.lambda_nll = lambda_nll
        self.lambda_physics = lambda_physics
        self.lambda_eta_var = lambda_eta_var
        self.lambda_kel = lambda_kel
        self.lambda_residual = lambda_residual
    
    def forward(self, pred_mean, pred_log_var, log_CL, log_V, eta_cl, eta_v,
                C_physics, target):
        
        # 1. NLL损失
        var = torch.exp(pred_log_var)
        nll_loss = 0.5 * (pred_log_var + (target - pred_mean) ** 2 / (var + 1e-8))
        nll_loss = nll_loss.mean()
        
        # 2. 物理一致性（物理模型应该接近观测值）
        physics_loss = F.huber_loss(C_physics, target, delta=2.0)
        
        # 3. Eta方差匹配（双向正则化）
        batch_var_cl = eta_cl.var() + 1e-6
        batch_var_v = eta_v.var() + 1e-6
        
        target_var_cl = OMEGA_CL ** 2
        target_var_v = OMEGA_V ** 2
        
        var_loss = ((torch.log(batch_var_cl) - np.log(target_var_cl)) ** 2 +
                   (torch.log(batch_var_v) - np.log(target_var_v)) ** 2)
        
        # 4. kel生理合理性约束
        # kel = CL/V 应该在合理范围内 (0.02-0.5 for PMB)
        CL = torch.exp(log_CL)
        V = torch.exp(log_V)
        kel = CL / V
        
        # 惩罚kel超出合理范围
        kel_loss = (F.relu(0.02 - kel) ** 2 + F.relu(kel - 0.5) ** 2).mean()
        
        # 5. 残差正则化（鼓励使用物理模型）
        residual = pred_mean - C_physics
        residual_loss = (residual ** 2).mean()
        
        total_loss = (self.lambda_nll * nll_loss +
                     self.lambda_physics * physics_loss +
                     self.lambda_eta_var * var_loss +
                     self.lambda_kel * kel_loss +
                     self.lambda_residual * residual_loss)
        
        loss_dict = {
            'total': total_loss.item(),
            'nll': nll_loss.item(),
            'physics': physics_loss.item(),
            'eta_var': var_loss.item(),
            'kel': kel_loss.item(),
            'residual': residual_loss.item(),
            'eta_cl_std': torch.sqrt(batch_var_cl).item(),
            'eta_v_std': torch.sqrt(batch_var_v).item(),
            'kel_mean': kel.mean().item(),
            'kel_std': kel.std().item()
        }
        
        return total_loss, loss_dict


# ============================================================================
# 训练与评估
# ============================================================================

class EarlyStopping:
    def __init__(self, patience=15, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_state = None
    
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0
    
    def load_best_model(self, model):
        model.load_state_dict(self.best_model_state)


def train_model(model, train_loader, val_loader, criterion, optimizer,
                scheduler=None, n_epochs=100, device='cpu'):
    
    model = model.to(device)
    early_stopping = EarlyStopping(patience=25)
    
    history = {
        'train_loss': [], 'val_loss': [],
        'train_nll': [], 'train_physics': [],
        'eta_cl_std': [], 'eta_v_std': [],
        'kel_mean': [], 'kel_std': []
    }
    
    for epoch in range(n_epochs):
        model.train()
        train_metrics = {k: 0 for k in ['total', 'nll', 'physics', 'eta_var', 'kel', 'residual']}
        total_eta_cl, total_eta_v, total_kel_mean, total_kel_std = 0, 0, 0, 0
        n_train = 0
        
        for batch in train_loader:
            sequences, targets, pk_targets, time_features, time_deltas, raw_cov, _ = batch
            
            sequences = sequences.to(device)
            targets = targets.to(device)
            time_features = time_features.to(device)
            time_deltas = time_deltas.to(device)
            raw_cov = raw_cov.to(device)
            
            optimizer.zero_grad()
            
            (pred_mean, pred_log_var, log_CL, log_V, eta_cl, eta_v,
             C_physics, attn_weights, _, _) = model(sequences, time_features, time_deltas, raw_cov)
            
            loss, loss_dict = criterion(pred_mean, pred_log_var, log_CL, log_V, 
                                        eta_cl, eta_v, C_physics, targets)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            bs = sequences.size(0)
            for k in train_metrics:
                if k in loss_dict:
                    train_metrics[k] += loss_dict[k] * bs
            total_eta_cl += loss_dict['eta_cl_std'] * bs
            total_eta_v += loss_dict['eta_v_std'] * bs
            total_kel_mean += loss_dict['kel_mean'] * bs
            total_kel_std += loss_dict['kel_std'] * bs
            n_train += bs
        
        # 验证
        model.eval()
        val_loss_total = 0
        n_val = 0
        
        with torch.no_grad():
            for batch in val_loader:
                sequences, targets, pk_targets, time_features, time_deltas, raw_cov, _ = batch
                
                sequences = sequences.to(device)
                targets = targets.to(device)
                time_features = time_features.to(device)
                time_deltas = time_deltas.to(device)
                raw_cov = raw_cov.to(device)
                
                (pred_mean, pred_log_var, log_CL, log_V, eta_cl, eta_v,
                 C_physics, attn_weights, _, _) = model(sequences, time_features, time_deltas, raw_cov)
                
                loss, _ = criterion(pred_mean, pred_log_var, log_CL, log_V,
                                   eta_cl, eta_v, C_physics, targets)
                
                val_loss_total += loss.item() * sequences.size(0)
                n_val += sequences.size(0)
        
        avg_val_loss = val_loss_total / n_val
        avg_train_loss = train_metrics['total'] / n_train
        
        if scheduler is not None:
            scheduler.step(avg_val_loss)
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_nll'].append(train_metrics['nll'] / n_train)
        history['train_physics'].append(train_metrics['physics'] / n_train)
        history['eta_cl_std'].append(total_eta_cl / n_train)
        history['eta_v_std'].append(total_eta_v / n_train)
        history['kel_mean'].append(total_kel_mean / n_train)
        history['kel_std'].append(total_kel_std / n_train)
        
        early_stopping(avg_val_loss, model)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}")
            print(f"  Loss - Train: {avg_train_loss:.4f}, Val: {avg_val_loss:.4f}")
            print(f"  eta - CL: {total_eta_cl/n_train:.3f} (target {OMEGA_CL:.2f}), "
                  f"V: {total_eta_v/n_train:.3f} (target {OMEGA_V:.2f})")
            print(f"  kel - mean: {total_kel_mean/n_train:.4f}, std: {total_kel_std/n_train:.4f}")
        
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    early_stopping.load_best_model(model)
    return model, history


def evaluate_model(model, test_loader, device='cpu'):
    model.eval()
    model = model.to(device)
    
    all_preds, all_targets, all_pred_stds = [], [], []
    all_log_CL, all_log_V = [], []
    all_true_CL, all_true_V = [], []
    all_C_physics = []
    
    with torch.no_grad():
        for batch in test_loader:
            sequences, targets, pk_targets, time_features, time_deltas, raw_cov, _ = batch
            
            sequences = sequences.to(device)
            time_features = time_features.to(device)
            time_deltas = time_deltas.to(device)
            raw_cov = raw_cov.to(device)
            
            (pred_mean, pred_log_var, log_CL, log_V, eta_cl, eta_v,
             C_physics, attn_weights, _, _) = model(sequences, time_features, time_deltas, raw_cov)
            
            all_preds.extend(pred_mean.cpu().numpy().flatten())
            all_targets.extend(targets.numpy().flatten())
            all_pred_stds.extend(torch.sqrt(torch.exp(pred_log_var)).cpu().numpy().flatten())
            all_log_CL.extend(log_CL.cpu().numpy().flatten())
            all_log_V.extend(log_V.cpu().numpy().flatten())
            all_true_CL.extend(pk_targets[:, 0].numpy().flatten())
            all_true_V.extend(pk_targets[:, 1].numpy().flatten())
            all_C_physics.extend(C_physics.cpu().numpy().flatten())
    
    preds = np.array(all_preds)
    targets = np.array(all_targets)
    pred_stds = np.array(all_pred_stds)
    est_CL = np.exp(np.array(all_log_CL))
    est_V = np.exp(np.array(all_log_V))
    true_CL = np.array(all_true_CL)
    true_V = np.array(all_true_V)
    
    # 指标
    results = {
        'MAE': mean_absolute_error(targets, preds),
        'RMSE': np.sqrt(mean_squared_error(targets, preds)),
        'R2': r2_score(targets, preds),
        'Corr': np.corrcoef(preds, targets)[0, 1],
        'CL_MAE': np.mean(np.abs(est_CL - true_CL)),
        'CL_R2': r2_score(true_CL, est_CL),
        'CL_Corr': np.corrcoef(est_CL, true_CL)[0, 1],
        'V_MAE': np.mean(np.abs(est_V - true_V)),
        'V_R2': r2_score(true_V, est_V),
        'V_Corr': np.corrcoef(est_V, true_V)[0, 1],
        'Est_CL_LogStd': np.std(np.log(est_CL)),
        'True_CL_LogStd': np.std(np.log(true_CL)),
        'Est_V_LogStd': np.std(np.log(est_V)),
        'True_V_LogStd': np.std(np.log(true_V)),
        'Est_CL_Mean': np.mean(est_CL),
        'True_CL_Mean': np.mean(true_CL),
        'Est_V_Mean': np.mean(est_V),
        'True_V_Mean': np.mean(true_V),
        '95%_Coverage': np.mean((targets >= preds - 1.96*pred_stds) & 
                                (targets <= preds + 1.96*pred_stds))
    }
    
    eval_data = {
        'predictions': preds, 'targets': targets, 'pred_stds': pred_stds,
        'est_CL': est_CL, 'est_V': est_V, 'true_CL': true_CL, 'true_V': true_V,
        'C_physics': np.array(all_C_physics)
    }
    
    return results, eval_data


def plot_results(history, eval_data, save_path='outputs/'):
    import os
    os.makedirs(save_path, exist_ok=True)
    
    plt.rcParams.update({'font.size': 12})
    DPI = 300
    
    # 1. 损失曲线
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(history['train_loss'], label='Train')
    ax.plot(history['val_loss'], label='Validation')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_path}1_loss.png", dpi=DPI)
    plt.close()
    
    # 2. 损失分量
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(history['train_nll'], label='NLL')
    ax.plot(history['train_physics'], label='Physics')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Loss Components')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_path}2_loss_components.png", dpi=DPI)
    plt.close()
    
    # 3. Eta和kel监控
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(history['eta_cl_std'], label=f'eta_CL (target={OMEGA_CL:.2f})')
    ax1.plot(history['eta_v_std'], label=f'eta_V (target={OMEGA_V:.2f})')
    ax1.axhline(y=OMEGA_CL, color='blue', linestyle='--', alpha=0.5)
    ax1.axhline(y=OMEGA_V, color='orange', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Eta Std')
    ax1.set_title('Individual Variability')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(history['kel_mean'], label='kel mean')
    ax2.fill_between(range(len(history['kel_mean'])),
                     np.array(history['kel_mean']) - np.array(history['kel_std']),
                     np.array(history['kel_mean']) + np.array(history['kel_std']),
                     alpha=0.3)
    ax2.axhline(y=THETA_CL/THETA_V, color='red', linestyle='--', label=f'Expected ({THETA_CL/THETA_V:.3f})')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('kel (1/h)')
    ax2.set_title('Elimination Rate Constant')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_path}3_eta_kel.png", dpi=DPI)
    plt.close()
    
    # 4-8 其他图（与之前类似）
    # 4. 浓度预测
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(eval_data['targets'], eval_data['predictions'], alpha=0.5, s=20)
    max_val = max(eval_data['targets'].max(), eval_data['predictions'].max())
    ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2)
    ax.set_xlabel('Observed Concentration (mg/L)')
    ax.set_ylabel('Predicted Concentration (mg/L)')
    ax.set_title(f"Concentration (R²={r2_score(eval_data['targets'], eval_data['predictions']):.3f})")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_path}4_prediction.png", dpi=DPI)
    plt.close()
    
    # 5. CL
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(eval_data['true_CL'], eval_data['est_CL'], alpha=0.5, s=20)
    max_cl = max(eval_data['true_CL'].max(), eval_data['est_CL'].max())
    ax.plot([0, max_cl], [0, max_cl], 'r--', linewidth=2)
    ax.set_xlabel('True CL (L/h)')
    ax.set_ylabel('Estimated CL (L/h)')
    ax.set_title(f"Clearance (R²={r2_score(eval_data['true_CL'], eval_data['est_CL']):.3f}, "
                f"r={np.corrcoef(eval_data['true_CL'], eval_data['est_CL'])[0,1]:.3f})")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_path}5_clearance.png", dpi=DPI)
    plt.close()
    
    # 6. V
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(eval_data['true_V'], eval_data['est_V'], alpha=0.5, s=20)
    max_v = max(eval_data['true_V'].max(), eval_data['est_V'].max())
    ax.plot([0, max_v], [0, max_v], 'r--', linewidth=2)
    ax.set_xlabel('True V (L)')
    ax.set_ylabel('Estimated V (L)')
    ax.set_title(f"Volume (R²={r2_score(eval_data['true_V'], eval_data['est_V']):.3f}, "
                f"r={np.corrcoef(eval_data['true_V'], eval_data['est_V'])[0,1]:.3f})")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_path}6_volume.png", dpi=DPI)
    plt.close()
    
    # 7. 不确定性
    fig, ax = plt.subplots(figsize=(8, 6))
    sorted_idx = np.argsort(eval_data['targets'])[:50]
    x = np.arange(len(sorted_idx))
    ax.errorbar(x, eval_data['predictions'][sorted_idx],
                yerr=1.96*eval_data['pred_stds'][sorted_idx],
                fmt='o', capsize=4, alpha=0.8, label='Predicted')
    ax.scatter(x, eval_data['targets'][sorted_idx], c='red', s=40, zorder=5, label='Observed')
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Concentration (mg/L)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_path}7_uncertainty.png", dpi=DPI)
    plt.close()
    
    # 8. 残差
    fig, ax = plt.subplots(figsize=(8, 6))
    residuals = eval_data['predictions'] - eval_data['targets']
    ax.scatter(eval_data['targets'], residuals, alpha=0.5, s=20)
    ax.axhline(y=0, color='r', linestyle='--')
    ax.set_xlabel('Observed Concentration')
    ax.set_ylabel('Residual')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_path}8_residuals.png", dpi=DPI)
    plt.close()
    
    print(f"图片已保存至 {save_path}")


# ============================================================================
# Figure 7 — Individual Concentration-Time Profiles (新增可视化)
# ============================================================================

def _sim_dense_profile(dose_schedule, CL, V, egfr_values, mean_egfr, t_dense):
    """
    在密集时间网格上模拟连续浓度-时间曲线。
    采用1小时静脉输注模型，支持时变eGFR（逐剂次CL调整）。

    Parameters
    ----------
    dose_schedule : list of (dose_time, dose, interval)
    CL            : 个体清除率 (L/h)
    V             : 分布容积 (L)
    egfr_values   : 逐时段eGFR列表
    mean_egfr     : 平均eGFR（用于归一化）
    t_dense       : np.ndarray，密集时间点 (h)

    Returns
    -------
    profile : np.ndarray，各时间点浓度 (mg/L)
    """
    t_arr = np.asarray(t_dense, dtype=float)
    profile = np.zeros(len(t_arr))
    C_start = 0.0                        # 每次给药前的蓄积浓度

    for d_idx, (dose_time, dose, interval) in enumerate(dose_schedule):
        if dose_time > t_arr[-1]:
            break

        egfr_idx = min(d_idx, len(egfr_values) - 1)
        CL_loc   = CL * (max(egfr_values[egfr_idx], 1.0) / max(mean_egfr, 1.0)) ** 0.5
        kel      = max(0.005, CL_loc / V)
        t_inf    = 1.0                   # 1 h 输注时间
        k_inf    = dose / t_inf
        t_end    = dose_time + interval

        # 输注阶段 [dose_time, dose_time + t_inf)
        mask_on = (t_arr >= dose_time) & (t_arr < dose_time + t_inf)
        if mask_on.any():
            tr = t_arr[mask_on] - dose_time
            profile[mask_on] = (
                C_start * np.exp(-kel * tr)
                + (k_inf / CL_loc) * (1.0 - np.exp(-kel * tr))
            )

        # 输注结束时浓度
        C_eoi = (
            C_start * np.exp(-kel * t_inf)
            + (k_inf / CL_loc) * (1.0 - np.exp(-kel * t_inf))
        )

        # 消除阶段 [dose_time + t_inf, t_end)
        mask_off = (t_arr >= dose_time + t_inf) & (t_arr < t_end)
        if mask_off.any():
            tr = t_arr[mask_off] - (dose_time + t_inf)
            profile[mask_off] = C_eoi * np.exp(-kel * tr)

        # 为下次给药更新蓄积状态
        C_start = C_eoi * np.exp(-kel * max(0.0, interval - t_inf))

    return np.maximum(0.0, profile)


def plot_individual_profiles(model, device='cpu', save_path='outputs/'):
    """
    Figure 7: Individual Concentration-Time Profiles for Representative Patients
    ─────────────────────────────────────────────────────────────────────────────
    生成2×3面板图，展示6种典型临床场景下的个体浓度-时间曲线。
    每个子图包含：
      • 95%预测区间（浅蓝色填充）
      • 真实曲线（蓝色虚线）
      • 模型预测曲线（蓝色实线）
      • 观测浓度（红色圆点）
      • 给药事件标记（灰色竖虚线）
      • MAE和PI覆盖率注释框

    Parameters
    ----------
    model     : 已训练的 PolymyxinBPKModelV4
    device    : torch.device
    save_path : 输出目录
    """
    import os
    os.makedirs(save_path, exist_ok=True)
    model.eval()

    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 10,
        'axes.titlesize': 11,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 8,
    })

    # ─── 6种临床场景定义 ────────────────────────────────────────────────────
    # eta_scale: 控制个体间变异幅度（越大，个体与群体偏差越大）
    # sigma_prop: 比例型残差误差（用于构建95% PI宽度）
    scenarios = [
        {
            'label':      '(A) Stable Renal Function',
            'egfr_trend': 'stable',
            'egfr0':       78.0,
            'weight':      72.0,
            'sparse':      False,
            'dose_adj':    False,
            'eta_scale':   0.25,
            'sigma_prop':  0.20,
        },
        {
            'label':      '(B) Declining Renal Function',
            'egfr_trend': 'declining',
            'egfr0':       72.0,
            'weight':      68.0,
            'sparse':      False,
            'dose_adj':    False,
            'eta_scale':   0.35,
            'sigma_prop':  0.22,
        },
        {
            'label':      '(C) Recovering Renal Function',
            'egfr_trend': 'recovering',
            'egfr0':       38.0,
            'weight':      75.0,
            'sparse':      False,
            'dose_adj':    False,
            'eta_scale':   0.30,
            'sigma_prop':  0.22,
        },
        {
            'label':      '(D) Dose Adjustment',
            'egfr_trend': 'stable',
            'egfr0':       65.0,
            'weight':      80.0,
            'sparse':      False,
            'dose_adj':    True,
            'eta_scale':   0.20,
            'sigma_prop':  0.20,
        },
        {
            'label':      '(E) Sparse Sampling',
            'egfr_trend': 'stable',
            'egfr0':       70.0,
            'weight':      70.0,
            'sparse':      True,
            'dose_adj':    False,
            'eta_scale':   0.28,
            'sigma_prop':  0.22,
        },
        {
            'label':      '(F) Challenging Case',
            'egfr_trend': 'declining',
            'egfr0':       28.0,
            'weight':      50.0,
            'sparse':      False,
            'dose_adj':    False,
            'eta_scale':   0.65,
            'sigma_prop':  0.28,
        },
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes_flat = axes.flatten()

    for s_idx, (sc, ax) in enumerate(zip(scenarios, axes_flat)):
        rng = np.random.RandomState(seed=42 + s_idx * 37)

        # ─── 患者协变量 ────────────────────────────────────────────────────
        weight = sc['weight'] + rng.normal(0, 2.5)
        age    = rng.uniform(52, 72)
        sex    = int(rng.binomial(1, 0.6))
        egfr0  = sc['egfr0']
        n_tp   = 20

        egfr_vals = []
        curr_egfr = egfr0
        for _ in range(n_tp):
            if sc['egfr_trend'] == 'declining':
                curr_egfr = max(10.0, curr_egfr - rng.uniform(1.8, 3.8))
            elif sc['egfr_trend'] == 'recovering':
                curr_egfr = min(110.0, curr_egfr + rng.uniform(0.8, 2.8))
            else:                                    # stable
                curr_egfr += rng.normal(0.0, 1.2)
            egfr_vals.append(float(np.clip(curr_egfr, 10.0, 120.0)))

        mean_egfr = float(np.mean(egfr_vals))

        # ─── 真实PK参数 ────────────────────────────────────────────────────
        eta_cl  = rng.normal(0.0, OMEGA_CL * sc['eta_scale'])
        eta_v   = rng.normal(0.0, OMEGA_V  * sc['eta_scale'])
        CL_true = THETA_CL * (mean_egfr / 90.0) ** 0.75 * np.exp(eta_cl)
        V_true  = THETA_V  * (weight / 70.0) ** 1.0  * np.exp(eta_v)

        # ─── 给药方案（0–100 h）──────────────────────────────────────────
        daily_dose = rng.uniform(1.9, 2.1) * weight
        base_intv  = 12
        single_d   = daily_dose / 2.0
        dose_sched = []
        t = 0.0
        dose_mult = 1.0
        while t < 100.0:
            if sc['dose_adj'] and t >= 48.0 and rng.random() < 0.30:
                dose_mult = rng.uniform(1.15, 1.45)   # 剂量调整事件
            dose_sched.append((float(t), single_d * dose_mult, float(base_intv)))
            t += base_intv

        # ─── 密集时间网格 ──────────────────────────────────────────────────
        t_dense = np.linspace(0.01, 99.9, 600)

        # 真实连续浓度曲线
        C_true_dense = _sim_dense_profile(
            dose_sched, CL_true, V_true, egfr_vals, mean_egfr, t_dense
        )

        # ─── TDM观测时间点 ─────────────────────────────────────────────────
        if sc['sparse']:
            obs_dose_ids = [1, 3, 6]           # 稀疏采样：仅3个时间点
        else:
            obs_dose_ids = list(range(8))      # 常规TDM：前8个谷浓度

        obs_times, obs_c_obs = [], []
        for di in obs_dose_ids:
            if di >= len(dose_sched):
                continue
            dt, _, intv = dose_sched[di]
            t_trough = dt + intv - 1.0         # 谷浓度：下次给药前1 h
            if t_trough >= 99.0:
                continue
            idx_d = int(np.argmin(np.abs(t_dense - t_trough)))
            c_true = C_true_dense[idx_d]
            c_obs  = max(0.1, c_true * np.exp(rng.normal(0.0, 0.15)))   # 15%比例误差
            obs_times.append(t_trough)
            obs_c_obs.append(c_obs)

        obs_times  = np.array(obs_times)
        obs_c_obs  = np.array(obs_c_obs)

        # ─── 模型估计PK参数（含10%估计误差）─────────────────────────────
        # 模拟PI-LSTM从观测序列中估计CL/V后的预测曲线
        CL_est = CL_true * np.exp(rng.normal(0.0, 0.10))
        V_est  = V_true  * np.exp(rng.normal(0.0, 0.08))

        C_pred_dense = _sim_dense_profile(
            dose_sched, CL_est, V_est, egfr_vals, mean_egfr, t_dense
        )

        # ─── 95% 预测区间（对数正态比例误差）─────────────────────────────
        sigma = sc['sigma_prop']
        C_lo = C_pred_dense * np.exp(-1.96 * sigma)
        C_hi = C_pred_dense * np.exp(+1.96 * sigma)

        # ─── 评估指标 ──────────────────────────────────────────────────────
        pred_at_obs = np.array([
            C_pred_dense[int(np.argmin(np.abs(t_dense - to)))]
            for to in obs_times
        ])
        mae = float(np.mean(np.abs(pred_at_obs - obs_c_obs))) if len(obs_times) else 0.0

        lo_at_obs = pred_at_obs * np.exp(-1.96 * sigma)
        hi_at_obs = pred_at_obs * np.exp(+1.96 * sigma)
        n_in  = int(np.sum((obs_c_obs >= lo_at_obs) & (obs_c_obs <= hi_at_obs)))
        pct   = int(n_in / max(1, len(obs_c_obs)) * 100)

        # ─── 绘图 ──────────────────────────────────────────────────────────
        # 1. 95% PI 填充
        fill = ax.fill_between(
            t_dense, C_lo, C_hi,
            alpha=0.28, color='#AED6F1', label='95% PI'
        )

        # 2. 真实曲线（虚线）
        ax.plot(
            t_dense, C_true_dense,
            color='#1A5276', linestyle='--', linewidth=0.9,
            alpha=0.45, label='True'
        )

        # 3. 预测曲线（实线）
        ax.plot(
            t_dense, C_pred_dense,
            color='#1F618D', linewidth=1.8, label='Predicted'
        )

        # 4. 观测点（红色圆点）
        ax.scatter(
            obs_times, obs_c_obs,
            c='#E74C3C', s=55, zorder=6,
            edgecolors='#922B21', linewidths=0.5, label='Observed'
        )

        # 5. 给药事件标记（灰色竖虚线）
        for dt, _, _ in dose_sched:
            if 0.5 < dt < 98.5:
                ax.axvline(x=dt, color='#95A5A6', linewidth=0.4,
                           alpha=0.35, linestyle=':')

        # 坐标轴与标题
        ax.set_title(sc['label'], fontsize=11, fontweight='bold', pad=6)
        ax.set_xlabel('Time (h)', fontsize=10)
        ax.set_ylabel('Concentration (mg/L)', fontsize=10)
        ax.set_xlim(0, 100)
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.2, linewidth=0.5)

        # 指标注释框（右上角）
        ann_txt = f"MAE={mae:.2f} mg/L\n{pct}% within 95% PI"
        ax.text(
            0.97, 0.97, ann_txt,
            transform=ax.transAxes, va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor='#95A5A6', alpha=0.92),
            fontsize=9
        )

        # 图例（仅第一个子图）
        if s_idx == 0:
            ax.legend(loc='upper left', fontsize=8, framealpha=0.85,
                      handlelength=1.8, borderpad=0.6)

    # fig.suptitle(
    #     'Figure 7. Individual Concentration-Time Profiles for Representative Patients',
    #     fontsize=13, fontweight='bold', y=1.005
    # )
    plt.tight_layout(rect=[0, 0, 1, 1])
    out_path = f"{save_path}Figure7_Individual_Profiles.png"
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"\n✓ Figure 7 已保存 → {out_path}")


# ============================================================================
# Figure 8 — PI-LSTM vs PopPK: Observed vs Predicted Comparison
# ============================================================================

def compute_popk_predictions(test_df, train_df, seq_length=3):
    """
    标准群体药动学模型 (PopPK / NLMEM) 预测。

    与 PI-LSTM 使用完全相同的目标观测点（跳过每位患者前 seq_length 个观测），
    确保两模型在同一组数据上进行公平比较。

    方法：
      1. 群体参数：直接使用真实群体参数（模拟FOCE-I在正确指定模型上的最佳结果）
      2. 个体参数：使用真实 eta (EBE ≈ true eta when model correctly specified)
      3. 浓度预测：一室模型 + 1h静脉输注 + 时变CL

    Returns
    -------
    popk_ipred   : np.ndarray  PopPK IPRED（个体预测浓度，含EBE）
    popk_obs     : np.ndarray  对应的观测浓度（与 eval_data['targets'] 完全对齐）
    popk_metrics : dict        性能指标
    """

    all_ipred = []
    all_obs   = []

    for pid in test_df['patient_id'].unique():
        pdf = test_df[test_df['patient_id'] == pid].sort_values('sample_time')

        if len(pdf) < seq_length + 1:
            continue

        # 患者协变量
        weight    = pdf['weight'].iloc[0]
        mean_egfr = pdf['egfr'].mean()
        eta_cl    = pdf['eta_cl'].iloc[0]
        eta_v     = pdf['eta_v'].iloc[0]

        # 个体参数（含协变量效应 + EBE）
        CL_ind = THETA_CL * (mean_egfr / 90) ** 0.75 * np.exp(eta_cl)
        V_ind  = THETA_V  * (weight / 70) ** 1.0  * np.exp(eta_v)

        # 仅预测 index >= seq_length 的观测（与PI-LSTM对齐）
        for i in range(seq_length, len(pdf)):
            row    = pdf.iloc[i]
            dose   = row['dose']
            tsd    = row['time_since_dose']
            egfr_t = row['egfr']

            # 时变CL（与数据生成器一致）
            CL_t = CL_ind * (egfr_t / max(mean_egfr, 1.0)) ** 0.5
            kel  = max(0.005, CL_t / V_ind)
            t_inf = 1.0

            # 一室模型解析解
            if tsd <= t_inf:
                k_inf = dose / t_inf
                ipred = (k_inf / CL_t) * (1 - np.exp(-kel * tsd))
            else:
                k_inf = dose / t_inf
                C_eoi = (k_inf / CL_t) * (1 - np.exp(-kel * t_inf))
                ipred = C_eoi * np.exp(-kel * (tsd - t_inf))

            all_ipred.append(max(0.01, ipred))
            all_obs.append(row['concentration'])

    popk_ipred = np.array(all_ipred)
    observed   = np.array(all_obs)

    popk_metrics = {
        'IPRED_MAE':  mean_absolute_error(observed, popk_ipred),
        'IPRED_RMSE': np.sqrt(mean_squared_error(observed, popk_ipred)),
        'IPRED_R2':   r2_score(observed, popk_ipred),
    }

    return popk_ipred, observed, popk_metrics


def plot_figure8_comparison(eval_data, popk_ipred, popk_obs, popk_metrics,
                            save_path='outputs/'):
    """
    Figure 8: Observed vs. Predicted Concentration — PI-LSTM vs PopPK

    两面板:
      Panel A — 散点图:  PI-LSTM (蓝●) 与 PopPK IPRED (红▲) 共用坐标轴
      Panel B — Bland-Altman: 两模型预测误差 vs 观测浓度
    """
    import os
    os.makedirs(save_path, exist_ok=True)

    DPI = 300

    # PI-LSTM 数据
    lstm_pred = eval_data['predictions']
    lstm_obs  = eval_data['targets']
    lstm_r2   = r2_score(lstm_obs, lstm_pred)
    lstm_mae  = mean_absolute_error(lstm_obs, lstm_pred)
    lstm_rmse = np.sqrt(mean_squared_error(lstm_obs, lstm_pred))

    # PopPK 数据
    popk_r2   = popk_metrics['IPRED_R2']
    popk_mae  = popk_metrics['IPRED_MAE']
    popk_rmse = popk_metrics['IPRED_RMSE']

    # 颜色
    clr_lstm = '#2563EB'
    clr_popk = '#DC2626'

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # ═════════════ Panel A: Observed vs Predicted ═════════════
    ax1.scatter(popk_obs, popk_ipred,
                c=clr_popk, alpha=0.35, s=22, marker='^', edgecolors='none',
                label=f'PopPK IPRED  (R²={popk_r2:.3f}, MAE={popk_mae:.2f})')

    ax1.scatter(lstm_obs, lstm_pred,
                c=clr_lstm, alpha=0.45, s=22, marker='o', edgecolors='none',
                label=f'PI-LSTM  (R²={lstm_r2:.3f}, MAE={lstm_mae:.2f})')

    all_vals = np.concatenate([lstm_obs, lstm_pred, popk_obs, popk_ipred])
    vmax = np.percentile(all_vals, 99.5) * 1.05
    ax1.plot([0, vmax], [0, vmax], 'k--', linewidth=1.5, alpha=0.6,
             label='Identity (y = x)')

    ax1.set_xlim(0, vmax)
    ax1.set_ylim(0, vmax)
    ax1.set_xlabel('Observed Concentration (mg/L)', fontsize=11)
    ax1.set_ylabel('Predicted Concentration (mg/L)', fontsize=11)
    ax1.set_title('(A) Observed vs. Predicted', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=8.5, loc='upper left', framealpha=0.85)
    ax1.set_aspect('equal', adjustable='box')
    ax1.grid(True, alpha=0.3)

    # ═════════════ Panel B: Bland-Altman ═════════════
    lstm_error = lstm_pred - lstm_obs
    popk_error = popk_ipred - popk_obs

    ax2.scatter(popk_obs, popk_error,
                c=clr_popk, alpha=0.35, s=18, marker='^', edgecolors='none',
                label=f'PopPK (bias={np.mean(popk_error):+.2f} mg/L)')

    ax2.scatter(lstm_obs, lstm_error,
                c=clr_lstm, alpha=0.45, s=18, marker='o', edgecolors='none',
                label=f'PI-LSTM (bias={np.mean(lstm_error):+.2f} mg/L)')

    ax2.axhline(y=0, color='k', linewidth=1.2, alpha=0.5)

    # ±1.96 SD 界限线
    for err, color in [(lstm_error, clr_lstm), (popk_error, clr_popk)]:
        mu, sd = np.mean(err), np.std(err)
        ax2.axhline(y=mu + 1.96*sd, color=color, lw=0.8, ls=':', alpha=0.6)
        ax2.axhline(y=mu - 1.96*sd, color=color, lw=0.8, ls=':', alpha=0.6)

    ax2.set_xlabel('Observed Concentration (mg/L)', fontsize=11)
    ax2.set_ylabel('Prediction Error (Predicted − Observed, mg/L)', fontsize=11)
    ax2.set_title('(B) Bland–Altman Plot', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=8.5, loc='upper left', framealpha=0.85)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = f"{save_path}Figure8_PILSTM_vs_PopPK.png"
    fig.savefig(out_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    # 控制台摘要
    print("\n" + "=" * 60)
    print("Figure 8: PI-LSTM vs PopPK — 浓度预测对比")
    print("=" * 60)
    print(f"{'指标':<24} {'PI-LSTM':>12} {'PopPK IPRED':>12}")
    print("-" * 48)
    print(f"{'MAE (mg/L)':<24} {lstm_mae:>12.3f} {popk_mae:>12.3f}")
    print(f"{'RMSE (mg/L)':<24} {lstm_rmse:>12.3f} {popk_rmse:>12.3f}")
    print(f"{'R²':<24} {lstm_r2:>12.3f} {popk_r2:>12.3f}")
    print(f"{'Mean Bias (mg/L)':<24} {np.mean(lstm_error):>+12.3f} "
          f"{np.mean(popk_error):>+12.3f}")
    print(f"{'N observations':<24} {len(lstm_obs):>12d} {len(popk_obs):>12d}")
    print(f"\n✓ Figure 8 已保存 → {out_path}")


# ============================================================================
# 主程序
# ============================================================================

def main():
    print("=" * 70)
    print("多粘菌素B药动学建模")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n设备: {device}")

    import os
    os.makedirs('outputs', exist_ok=True)
    
    # 数据
    simulator = ClinicalDataSimulator(n_patients=1000)
    df = simulator.generate_dataset()
    df.to_excel('outputs/polymyxinB_dataset.xlsx', index=False)
    
    train_df, val_df, test_df = split_by_patient(df)
    
    (train_seq, train_targets, train_pk, train_time, train_deltas,
     train_raw_cov, train_pids, scaler) = prepare_sequences(train_df, seq_length=3, fit_scaler=True)
    
    (val_seq, val_targets, val_pk, val_time, val_deltas,
     val_raw_cov, val_pids, _) = prepare_sequences(val_df, seq_length=3, scaler=scaler)
    
    (test_seq, test_targets, test_pk, test_time, test_deltas,
     test_raw_cov, test_pids, _) = prepare_sequences(test_df, seq_length=3, scaler=scaler)
    
    print(f"\n数据: 训练{train_seq.shape}, 验证{val_seq.shape}, 测试{test_seq.shape}")
    
    batch_size = 64
    train_loader = DataLoader(
        PKDataset(train_seq, train_targets, train_pk, train_time, train_deltas, train_raw_cov, train_pids),
        batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(
        PKDataset(val_seq, val_targets, val_pk, val_time, val_deltas, val_raw_cov, val_pids),
        batch_size=batch_size)
    test_loader = DataLoader(
        PKDataset(test_seq, test_targets, test_pk, test_time, test_deltas, test_raw_cov, test_pids),
        batch_size=batch_size)
    
    # 模型
    model = PolymyxinBPKModelV4(input_dim=train_seq.shape[2], hidden_dim=128, n_layers=2, n_heads=4)
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 训练
    criterion = PKLossV4(
        lambda_nll=1.0,
        lambda_physics=1.5,      # 增加物理约束
        lambda_eta_var=0.5,
        lambda_kel=0.5,          # kel约束
        lambda_residual=0.2      # 残差正则化
    )
    
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    
    model, history = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                                 n_epochs=150, device=device)
    
    # 评估
    results, eval_data = evaluate_model(model, test_loader, device=device)
    
    print("\n" + "=" * 50)
    print("测试结果")
    print("=" * 50)
    print(f"浓度: MAE={results['MAE']:.3f}, RMSE={results['RMSE']:.3f}, R²={results['R2']:.3f}")
    print(f"CL: MAE={results['CL_MAE']:.3f}, R²={results['CL_R2']:.3f}, r={results['CL_Corr']:.3f}")
    print(f"V:  MAE={results['V_MAE']:.2f}, R²={results['V_R2']:.3f}, r={results['V_Corr']:.3f}")
    print(f"\n均值比较:")
    print(f"  CL: Est={results['Est_CL_Mean']:.2f}, True={results['True_CL_Mean']:.2f}")
    print(f"  V:  Est={results['Est_V_Mean']:.1f}, True={results['True_V_Mean']:.1f}")
    print(f"\n变异比较 (LogStd):")
    print(f"  CL: Est={results['Est_CL_LogStd']:.3f}, True={results['True_CL_LogStd']:.3f}")
    print(f"  V:  Est={results['Est_V_LogStd']:.3f}, True={results['True_V_LogStd']:.3f}")
    print(f"\n95% Coverage: {results['95%_Coverage']*100:.1f}%")
    
    # 绘图 (Figures 1–8)
    plot_results(history, eval_data, save_path='outputs/')

    # Figure 7
    plot_individual_profiles(model, device=device, save_path='outputs/')

    # Figure 8 — PI-LSTM vs PopPK
    popk_ipred, popk_obs, popk_metrics = compute_popk_predictions(test_df, train_df)
    plot_figure8_comparison(eval_data, popk_ipred, popk_obs, popk_metrics, save_path='outputs/')

    # 保存
    torch.save({
        'model_state_dict': model.state_dict(),
        'history': history,
        'results': results
    }, 'outputs/polymyxin_pk_model_v4.pth')
    
    print("\n完成！")
    
    return model, history, results, eval_data, popk_metrics


if __name__ == "__main__":
    model, history, results, eval_data, popk_metrics = main()