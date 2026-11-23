"""Общие функции для ноутбуков Левина: данные, фичи, схемы валидации, метрики."""
import numpy as np, pandas as pd
from ucimlrepo import fetch_ucirepo
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, GroupKFold, TimeSeriesSplit

SEED = 42
MONTHS = {m: i + 1 for i, m in enumerate('jan feb mar apr may jun jul aug sep oct nov dec'.split())}
CATS = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
# параметры финального LightGBM из model_levin.ipynb (после тюнинга)
BEST = dict(subsample=0.6, reg_lambda=1, num_leaves=4, n_estimators=400, min_child_samples=100,
            max_cat_to_onehot=8, learning_rate=0.02, colsample_bytree=0.5, cat_smooth=100)


def load():
    """Сырая таблица, таргет, год (из порядка строк) и период 'год-месяц'."""
    bm = fetch_ucirepo(id=222)
    raw = pd.concat([bm.data.features, bm.data.targets], axis=1)
    y = (raw.y == 'yes').astype(int).values
    mnum = raw.month.map(MONTHS)
    year = 2008 + (mnum.diff().fillna(0) < 0).cumsum().values
    return raw, y, year, (year * 100 + mnum.values)


def build(raw, year=None, fe=True, duration=False, drop=()):
    """Признаки, считаются построчно (без таргета). Пропуски -> осмысленные категории."""
    d = raw.drop(columns='y').rename(columns={'day_of_week': 'day'}).copy()
    for c in CATS: d[c] = d[c].fillna('no_prev' if c == 'poutcome' else 'unknown')
    if fe:
        d['month_num'] = d.month.map(MONTHS)
        d['prev_contacted'] = (d.pdays != -1).astype(int)
        d['pdays_clean'] = d.pdays.where(d.pdays > 0)
        d['prev_success'] = (d.poutcome == 'success').astype(int)
        d['contact_unknown'] = (d.contact == 'unknown').astype(int)
        d['age_bin'] = pd.cut(d.age, [0, 25, 30, 40, 50, 60, 120], labels=['<=25', '26-30', '31-40', '41-50', '51-60', '60+']).astype(str)
        d['balance_log'] = np.sign(d.balance) * np.log1p(d.balance.abs())
        d['balance_neg'] = (d.balance < 0).astype(int)
        d['n_loans'] = (d.housing == 'yes').astype(int) + (d.loan == 'yes') + (d.default == 'yes')
        d['campaign_clip'] = d.campaign.clip(upper=10)
        d['previous_clip'] = d.previous.clip(upper=10)
        d['job_sr'] = d.job.isin(['student', 'retired']).astype(int)
        d = d.drop(columns='pdays')
    if year is not None: d['year'] = year
    if not duration: d = d.drop(columns='duration')
    d = d.drop(columns=list(drop))
    for c in [*CATS, 'age_bin']:
        if c in d: d[c] = d[c].astype('category')
    return d.reset_index(drop=True)


NP_DROP = ['month', 'month_num', 'day', 'contact_unknown', 'contact']   # прокси периода (см. model_levin.ipynb)


def make_lgb(X=None, **kw):
    return LGBMClassifier(**{**BEST, **kw}, n_jobs=8, random_state=SEED, verbose=-1)


# ---------- схемы валидации: каждая возвращает список (train_idx, test_idx) ----------
def strat_splits(y, groups=None, k=5):
    """Stratified K-fold; с groups — StratifiedGroupKFold (один клиент целиком в одном фолде)."""
    if groups is None: return list(StratifiedKFold(k, shuffle=True, random_state=SEED).split(y, y))
    return list(StratifiedGroupKFold(k, shuffle=True, random_state=SEED).split(y, y, groups))


def block_splits(period, y, groups=None, k=5):
    """GroupKFold по периодам 'год-месяц'; с groups — убираем из train клиентов, которые есть в test (purge)."""
    sp = list(GroupKFold(k).split(y, y, period))
    if groups is None: return sp
    return [(tr[~np.isin(groups[tr], np.unique(groups[te]))], te) for tr, te in sp]


def time_splits(n, k=5):
    """Forward-chaining: учимся на прошлом, тестируем на будущем (строки идут хронологически)."""
    return list(TimeSeriesSplit(k).split(np.arange(n)))


# ---------- метрики ----------
def lift_at(y_true, p, frac=0.10):
    k = int(len(y_true) * frac)
    return y_true[np.argsort(-p)[:k]].mean() / y_true.mean()


def run_cv(make, X, y, splits):
    """Метрики по фолдам + out-of-fold предсказания и номера фолдов."""
    oof, fold, rows = np.full(len(y), np.nan), np.full(len(y), -1), []
    for i, (tr, te) in enumerate(splits):
        p = make(X).fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[te])[:, 1]
        oof[te], fold[te] = p, i
        rows.append((roc_auc_score(y[te], p), average_precision_score(y[te], p), lift_at(y[te], p)))
    return pd.DataFrame(rows, columns=['auc', 'pr', 'lift10']), oof, fold


# ---------- восстановление клиентов и история на момент решения ----------
def with_dates(raw, year):
    """Копия raw с датой звонка (год восстановлен по порядку строк) и без NaN в job/education."""
    d = raw.copy()
    d['date'] = pd.to_datetime(dict(year=year, month=raw.month.map(MONTHS), day=raw.day_of_week))
    for c in ['job', 'education']: d[c] = d[c].fillna('unknown')
    return d


def link_clients(d):
    """Для строки с pdays>0 ищем прошлую запись того же клиента: дата = date - pdays, тот же профиль,
    проверка campaign_old + previous_old == previous_new. Возвращает индекс прошлой записи (-1 если нет)."""
    from collections import defaultdict
    keys = defaultdict(list)
    for i, k in enumerate(zip(d.date, d.job, d.marital, d.education, d.default)): keys[k].append(i)
    prev = np.full(len(d), -1)
    for i in np.where(d.pdays.values > 0)[0]:
        gap = int(d.pdays.iat[i]); dt = d.date.iat[i] - pd.Timedelta(days=gap)
        cand = [j for j in keys.get((dt, d.job.iat[i], d.marital.iat[i], d.education.iat[i], d.default.iat[i]), [])
                if j < i and 0 <= d.age.iat[i] - d.age.iat[j] <= gap // 365 + 1]
        if not cand: continue
        ok = [j for j in cand if d.campaign.iat[j] + d.previous.iat[j] == d.previous.iat[i]]
        if len(cand) == 1: prev[i] = cand[0]
        elif len(ok) == 1: prev[i] = ok[0]
        elif len(ok) > 1: prev[i] = min(ok, key=lambda j: abs(d.balance.iat[j] - d.balance.iat[i]))
    return prev


def history_features(d, y, prev):
    """Что известно о клиенте НА МОМЕНТ звонка: идём по цепочке прошлых записей (все строго раньше по дате)."""
    n = len(d); dt = d.date.values; rows = []
    for i in range(n):
        j = prev[i]; depth = nb = 0; last_buy = np.nan
        while j >= 0:
            depth += 1; nb += y[j]
            if y[j] == 1 and np.isnan(last_buy): last_buy = (dt[i] - dt[j]) / np.timedelta64(1, 'D')
            j = prev[j]
        rows.append((depth, nb, int(nb > 0), last_buy))
    return pd.DataFrame(rows, columns=['h_depth', 'h_n_buys', 'h_ever_bought', 'h_days_since_buy'])


def final_features(raw, year, y):
    """Финальный набор: без duration, без прокси периода, без campaign + история клиента."""
    X = build(raw, drop=NP_DROP + ['campaign', 'campaign_clip'])
    d = with_dates(raw, year)
    return pd.concat([X, history_features(d, y, link_clients(d))], axis=1)
