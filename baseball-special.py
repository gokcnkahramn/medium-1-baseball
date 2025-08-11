import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from pybaseball import statcast
import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu
from statsmodels.stats.proportion import proportions_ztest
from scipy.stats import fisher_exact
import statsmodels.formula.api as smf
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

positive_events = {
        'walk', 'single', 'double', 'triple', 'home_run',
        'hit_by_pitch', 'catcher_interf'
}
negative_events = {
        'field_out', 'strikeout', 'strikeout_double_play',
        'grounded_into_double_play', 'double_play', 'triple_play',
        'fielders_choice_out', 'force_out', 'sac_bunt',
        'sac_fly', 'sac_fly_double_play', 'field_error'
}

def classify_event(event):

    if event in positive_events:
        return "positive"
    elif event in negative_events:
        return "negative"
    else:
        return "neutral"

def classify_count_state(row):
    if row['balls'] > row['strikes']:
        return 'hitter_adv'
    elif row['balls'] == row['strikes']:
        return 'even'
    else:
        return 'pitcher_adv'

def analyze_and_compare(year, min_pitches=30):
    print(f"\n=== Analyzing season {year} ===")
    
    start_date = f"{year}-03-01"
    end_date = f"{year}-10-01"
    
    data = statcast(start_date, end_date)
    cols = ['game_date', 'batter', 'pitcher', 'balls', 'strikes', 'pitch_type', 'events']
    df = data[cols].dropna(subset=['events']).copy()

    # Classify events and count states
    df['event_type'] = df['events'].apply(classify_event)
    df = df[df['event_type'] != 'neutral'].copy()
    df['count_state'] = df.apply(classify_count_state, axis=1)
    df['is_positive'] = (df['event_type'] == 'positive').astype(int)

    # Aggregate positive rate per batter on all plays
    agg = df.groupby('batter')['is_positive'].agg(['sum', 'count']).reset_index()
    agg['positive_rate'] = agg['sum'] / agg['count']
    agg_filtered = agg[agg['count'] >= min_pitches].copy()

    if agg_filtered.empty:
        print(f"No batters with minimum {min_pitches} plays.")
        return None, None, None, None

    threshold = agg_filtered['positive_rate'].quantile(0.75)
    special = agg_filtered[agg_filtered['positive_rate'] >= threshold]
    others = agg_filtered[agg_filtered['positive_rate'] < threshold]

    if special.empty or others.empty:
        print("Special or others group is empty, cannot perform tests.")
        return None, None, None, None

    # Add is_special to df
    df['is_special'] = df['batter'].isin(special['batter']).astype(int)

    # Weighted positive ratio by count_state and group
    weighted_ratios = df.groupby(['count_state', 'is_special']).apply(
        lambda x: x['is_positive'].sum() / x.shape[0]
    ).reset_index(name='positive_ratio')

    weighted_ratios['group'] = weighted_ratios['is_special'].map({0: 'others', 1: 'special'})
    weighted_ratios = weighted_ratios.drop(columns='is_special')

    # Distribution of events by special vs others and count_state
    pivot_table = pd.crosstab(
        [df['count_state'], df['events']],
        df['is_special'],
        normalize='index'
    ).rename(columns={0: 'others', 1: 'special'})

    # Mann-Whitney U test on positive_rate per batter
    stat_mwu, p_mwu = mannwhitneyu(special['positive_rate'], others['positive_rate'], alternative='greater')

    # Proportions z-test & Fisher exact on all plays
    special_rows = df[df['is_special'] == 1]
    other_rows = df[df['is_special'] == 0]

    special_pos = special_rows['is_positive'].sum()
    special_n = special_rows.shape[0]
    other_pos = other_rows['is_positive'].sum()
    other_n = other_rows.shape[0]

    count = [special_pos, other_pos]
    nobs = [special_n, other_n]

    stat_z, p_z = proportions_ztest(count, nobs, alternative='larger')

    contingency_table = [
        [int(special_pos), int(special_n - special_pos)],
        [int(other_pos), int(other_n - other_pos)]
    ]
    oddsratio, p_fisher = fisher_exact(contingency_table, alternative='greater')

    # Logistic regression with cluster-robust SE by batter
    try:
        logit = smf.logit('is_positive ~ is_special', data=df).fit(disp=False, cov_type='cluster', cov_kwds={'groups': df['batter']})
        logit_summary = logit.summary2().tables[1]
    except Exception as e:
        logit_summary = None
        print(f"Logistic regression failed: {e}")

    # GEE model clustered by batter
    try:
        gee = smf.gee('is_positive ~ is_special', groups='batter', data=df, family=sm.families.Binomial()).fit()
        gee_summary = gee.summary2().tables[1]
    except Exception as e:
        gee_summary = None
        print(f"GEE model failed: {e}")

    # Print key results
    print(f"Special group threshold positive_rate >= {threshold:.4f}")
    print(f"Mann-Whitney U test: stat={stat_mwu:.4f}, p={p_mwu:.4g}")
    print(f"Proportions Z-test: stat={stat_z:.4f}, p={p_z:.4g}")
    print(f"Fisher's Exact test: oddsratio={oddsratio:.4f}, p={p_fisher:.4g}")

    # Count state bazlı positive event yüzdeleri
    count_state_ratios = (
        df.groupby('count_state')['is_positive']
          .mean()
          .reset_index(name='positive_event_percentage')
    )
    print("\nPositive Event Percentages by Count State:")
    for _, row in count_state_ratios.iterrows():
        print(f"  {row['count_state']}: {row['positive_event_percentage']*100:.2f}%")


    return df, pivot_table, weighted_ratios, {
        'mwu_stat': stat_mwu,
        'mwu_p': p_mwu,
        'z_stat': stat_z,
        'z_p': p_z,
        'fisher_oddsratio': oddsratio,
        'fisher_p': p_fisher,
        'logit_summary': logit_summary,
        'gee_summary': gee_summary,
        'special_threshold': threshold,
        'special_count': len(special),
        'others_count': len(others),
    }

def plot_positive_event_distribution(df, year):
    # Positive event'leri filtrele
    df_positive = df[df['events'].isin(positive_events)].copy()

    # Grup isimleri
    df_positive['group'] = df_positive['is_special'].map({0: 'others', 1: 'special'})

    # Pivot tablo (normalize edilmiş)
    pivot = df_positive.groupby(['group', 'events']).size().reset_index(name='count')
    pivot['ratio'] = pivot.groupby('group')['count'].transform(lambda x: x / x.sum())

    print(f"\n{year} Positive Event Distribution Table:")
    print(pivot.pivot(index='events', columns='group', values='ratio'))

    # Grafik
    plt.figure(figsize=(8,5))
    sns.barplot(data=pivot, x='events', y='ratio', hue='group')
    plt.title(f'{year} Season: Positive Event Distribution (Sum=1 per Group)')
    plt.ylabel('Ratio')
    plt.ylim(0, 1)
    plt.legend(title='Group')
    plt.tight_layout()
    plt.show()

# --- Çalıştırma kısmı: yıllara göre sonuçları topla ve göster ---
years = list(range(2020, 2025))
all_results = {}

for year in years:
    df, pivot, weighted_ratios, tests = analyze_and_compare(year)
    if df is not None:
        all_results[year] = {'df': df, 'pivot': pivot, 'weighted_ratios': weighted_ratios, 'tests': tests}
        plot_positive_event_distribution(df, year)

# Özet istatistik sonuçları:
for y, res in all_results.items():
    print(f"\n---- {y} Sezonu ----")
    print(f"Special threshold (positive_rate): {res['tests']['special_threshold']:.4f}")
    print(f"Special oyuncu sayısı: {res['tests']['special_count']}")
    print(f"Others oyuncu sayısı: {res['tests']['others_count']}")
    print(f"Mann-Whitney U testi p değeri: {res['tests']['mwu_p']:.4g}")
    print(f"Proportions Z-testi p değeri: {res['tests']['z_p']:.4g}")
    print(f"Fisher Exact testi p değeri: {res['tests']['fisher_p']:.4g}")
    print("Logistic Regression (is_special) coef ve p-değeri:")
    if res['tests']['logit_summary'] is not None:
        coef = res['tests']['logit_summary'].loc['is_special']
        print(f"  coef={coef['Coef.']:.4f}, p={coef['P>|z|']:.4g}")
    else:
        print("  Model başarısız.")
    print("GEE Model (is_special) coef ve p-değeri:")
    if res['tests']['gee_summary'] is not None:
        coef = res['tests']['gee_summary'].loc['is_special']
        print(f"  coef={coef['Coef.']:.4f}, p={coef['P>|z|']:.4g}")
    else:
        print("  Model başarısız.")
