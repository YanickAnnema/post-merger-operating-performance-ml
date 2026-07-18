"""
Data Visualisation — M&A Synergy Estimation Pipeline
=====================================================

Produces thesis-oriented visualisations of the DAQ pipeline outputs.

Figures generated:
  1. Pipeline attrition by stage (raw → feature-engineered → labeled → ML-ready)
  2. Label yield by effective year
  3. Feature coverage heatmap by synergy channel
  4. Train/val/test split distribution over time
  5. Target variable distribution (raw vs. winsorised)
  6. Altman Z-score distributions (acquiror vs. target)
  7. Selected feature distributions and correlations

Output:
  All figures saved to Desktop as high-res PNG files, ready for thesis inclusion.

Optimised for Spyder IDE (F5 execution).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DAQ_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "DAQ pipeline"
ML_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "ML pipeline"

CONFIG = {
    'desktop':      PROJECT_ROOT,
    'output_dir':   ML_OUTPUT_DIR / "thesis_vis",
    'stage0_csv':   DAQ_OUTPUT_DIR / "full_deal_level.csv",
    'stage1_csv':   DAQ_OUTPUT_DIR / "full_deal_level_features.csv",
    'stage2_csv':   DAQ_OUTPUT_DIR / "ml_ready.csv",
    'dpi':          300,    # publication quality
    'figsize':      (12, 7),
    'style':        'seaborn-v0_8-darkgrid',
}

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _ensure_output_dir():
    """Create output directory if it doesn't exist."""
    CONFIG['output_dir'].mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {CONFIG['output_dir']}")


def _load_stages():
    """Load the three key pipeline outputs."""
    logger.info("Loading pipeline outputs...")

    df0 = pd.read_csv(CONFIG['stage0_csv'], low_memory=False)
    df0['DateEffective'] = pd.to_datetime(df0['DateEffective'], errors='coerce')
    df0['_year'] = df0['DateEffective'].dt.year
    logger.info(f"  Stage 0 (raw DAQ): {len(df0)} rows")

    df1 = pd.read_csv(CONFIG['stage1_csv'], low_memory=False)
    df1['DateEffective'] = pd.to_datetime(df1['DateEffective'], errors='coerce')
    df1['_year'] = df1['DateEffective'].dt.year
    logger.info(f"  Stage 1 (features): {len(df1)} rows")

    df2 = pd.read_csv(CONFIG['stage2_csv'], low_memory=False)
    df2['DateEffective'] = pd.to_datetime(df2['DateEffective'], errors='coerce')
    df2['_year'] = df2['DateEffective'].dt.year
    logger.info(f"  Stage 2 (ML-ready): {len(df2)} rows")

    return df0, df1, df2


def _savefig(fig, name):
    """Save figure to output directory."""
    path = CONFIG['output_dir'] / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=CONFIG['dpi'], bbox_inches='tight')
    logger.info(f"  Saved: {path.name}")
    plt.close(fig)


# =============================================================================
# FIGURE 1: PIPELINE ATTRITION
# =============================================================================

def fig_pipeline_attrition(df0, df1, df2):
    """
    Waterfall plot showing sample survival through pipeline stages.
    """
    logger.info("\n[FIG 1] Pipeline attrition")

    stages = ['Raw DAQ\n(Stage 0)', 'Features\n(Stage 1)',
              'With Label\n(Stage 1)', 'ML-Ready\n(Stage 2)']
    counts = [
        len(df0),
        len(df1),
        (df1['synergy_healy1992_w'].notna()).sum(),
        len(df2),
    ]

    fig, ax = plt.subplots(figsize=CONFIG['figsize'])
    colors = ['#1f77b4', '#1f77b4', '#ff7f0e', '#2ca02c']
    bars = ax.bar(stages, counts, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    # Add count labels on bars
    for i, (bar, count) in enumerate(zip(bars, counts)):
        height = bar.get_height()
        pct = 100 * count / counts[0]
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(count):,}\n({pct:.1f}%)',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_ylabel('Number of Deals', fontsize=12, fontweight='bold')
    ax.set_title('Pipeline Attrition: Sample Survival by Stage',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim(0, max(counts) * 1.15)
    ax.grid(axis='y', alpha=0.3)

    _savefig(fig, '01_pipeline_attrition')


# =============================================================================
# FIGURE 2: LABEL YIELD BY YEAR
# =============================================================================

def fig_label_yield_by_year(df1):
    """
    Line plot: labeled deals per year and label yield rate.
    """
    logger.info("\n[FIG 2] Label yield by year")

    df1['_has_label'] = df1['synergy_healy1992_w'].notna()
    yearly = df1.groupby('_year').agg({
        '_has_label': ['sum', 'count']
    }).reset_index()
    yearly.columns = ['year', 'labeled', 'total']
    yearly['yield_pct'] = 100 * yearly['labeled'] / yearly['total']
    yearly = yearly[yearly['year'] >= 1985]

    fig, ax1 = plt.subplots(figsize=CONFIG['figsize'])

    # Left axis: labeled deal count
    ax1.bar(yearly['year'], yearly['labeled'], alpha=0.6, color='#2ca02c',
            label='Labeled Deals', edgecolor='black', linewidth=0.8)
    ax1.set_ylabel('Number of Labeled Deals', fontsize=11, fontweight='bold', color='#2ca02c')
    ax1.tick_params(axis='y', labelcolor='#2ca02c')

    # Right axis: yield percentage
    ax2 = ax1.twinx()
    ax2.plot(yearly['year'], yearly['yield_pct'], color='#d62728', linewidth=2.5,
             marker='o', markersize=5, label='Label Yield %')
    ax2.set_ylabel('Label Yield (%)', fontsize=11, fontweight='bold', color='#d62728')
    ax2.tick_params(axis='y', labelcolor='#d62728')
    ax2.set_ylim(0, max(yearly['yield_pct']) * 1.15)

    ax1.set_xlabel('Effective Year', fontsize=12, fontweight='bold')
    ax1.set_title('Label Yield by Effective Year (Feature-Engineered Sample)',
                  fontsize=14, fontweight='bold', pad=20)
    ax1.grid(axis='y', alpha=0.3)

    # Add sample cutoff line (1995)
    ax1.axvline(x=1995, color='gray', linestyle='--', linewidth=2, alpha=0.7,
                label='Sample window start (1995)')
    ax1.legend(loc='upper left', fontsize=10)

    _savefig(fig, '02_label_yield_by_year')


# =============================================================================
# FIGURE 3: FEATURE COVERAGE HEATMAP BY CHANNEL
# =============================================================================

def fig_feature_coverage_by_channel(df2):
    """
    Heatmap: coverage (% non-NaN) of features in ML-ready sample, grouped by synergy channel.
    """
    logger.info("\n[FIG 3] Feature coverage by channel")

    # Define channels
    channels = {
        'COST': [
            'cost_relative_asset_size', 'cost_ppe_intensity_diff',
            'cost_inventory_turnover_gap', 'cost_target_asset_utilization',
            'log_deal_value', 'deal_tender_offer', 'deal_friendly',
        ],
        'REVENUE': [
            'revenue_rd_intensity_diff', 'revenue_capex_intensity_diff',
            'revenue_intangible_intensity_diff', 'revenue_relative_size_sales',
            'deal_cross_border',
        ],
        'OPERATIONAL': [
            'operational_asset_turnover_gap', 'operational_roa_gap',
            'operational_acquiror_op_margin', 'operational_target_cf_margin',
            'deal_industry_4dig', 'deal_industry_2dig',
        ],
        'FINANCIAL': [
            'financial_leverage_gap', 'financial_cash_ratio_diff',
            'financial_acquiror_cash_to_sales', 'financial_quick_ratio_acquiror',
            'financial_quick_ratio_target', 'deal_stock_payment', 'deal_all_cash',
            'financial_altman_z_acquiror', 'financial_altman_z_target',
        ],
        'MACRO': [
            'sp500_trailing_12m', 'credit_spread_bbb_aaa',
        ],
    }

    # Compute coverage for each feature
    coverage_data = []
    for ch_name, features in channels.items():
        for feat in features:
            if feat in df2.columns:
                cov = 100 * df2[feat].notna().sum() / len(df2)
            else:
                cov = 0.0
            coverage_data.append({'channel': ch_name, 'feature': feat, 'coverage': cov})

    df_cov = pd.DataFrame(coverage_data)

    # Create heatmap
    fig, ax = plt.subplots(figsize=(14, 8))

    # Sort by channel and coverage
    df_cov = df_cov.sort_values(['channel', 'coverage'], ascending=[True, False])

    # Reshape for heatmap
    pivot = df_cov.pivot_table(index='feature', columns='channel', values='coverage')
    pivot = pivot[['COST', 'REVENUE', 'OPERATIONAL', 'FINANCIAL', 'MACRO']]

    sns.heatmap(pivot, annot=True, fmt='.0f', cmap='RdYlGn', cbar_kws={'label': 'Coverage (%)'},
                vmin=0, vmax=100, ax=ax, cbar=True, linewidths=0.5, linecolor='gray')

    ax.set_title('Feature Coverage by Synergy Channel (ML-Ready Sample, n=4,229)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Synergy Channel', fontsize=12, fontweight='bold')
    ax.set_ylabel('Feature', fontsize=12, fontweight='bold')

    _savefig(fig, '03_feature_coverage_heatmap')


# =============================================================================
# FIGURE 4: SPLIT DISTRIBUTION OVER TIME
# =============================================================================

def fig_split_distribution_over_time(df2):
    """
    Stacked bar plot: train/val/test split composition by year.
    """
    logger.info("\n[FIG 4] Split distribution over time")

    yearly_split = df2.groupby(['_year', 'split']).size().reset_index(name='count')
    pivot = yearly_split.pivot(index='_year', columns='split', values='count').fillna(0)

    # Ensure all three splits are present
    for s in ['train', 'val', 'test']:
        if s not in pivot.columns:
            pivot[s] = 0

    pivot = pivot[['train', 'val', 'test']]

    fig, ax = plt.subplots(figsize=CONFIG['figsize'])

    pivot.plot(kind='bar', stacked=True, ax=ax, color=['#1f77b4', '#ff7f0e', '#2ca02c'],
               alpha=0.8, edgecolor='black', linewidth=0.8)

    ax.set_xlabel('Effective Year', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Deals', fontsize=12, fontweight='bold')
    ax.set_title('Train/Validation/Test Split Composition by Year',
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(title='Split', loc='upper left', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    # Add split boundary lines
    ax.axvline(x=20, color='red', linestyle='--', linewidth=2, alpha=0.7)  # 2015
    ax.axvline(x=23, color='orange', linestyle='--', linewidth=2, alpha=0.7)  # 2018

    _savefig(fig, '04_split_distribution_over_time')


# =============================================================================
# FIGURE 5: TARGET VARIABLE DISTRIBUTION
# =============================================================================

def fig_target_distribution(df1, df2):
    """
    Side-by-side: raw and winsorised target distributions.
    """
    logger.info("\n[FIG 5] Target variable distribution")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: raw synergy_healy1992
    target_raw = df1['synergy_healy1992'].dropna()
    axes[0].hist(target_raw, bins=50, alpha=0.7, color='#ff7f0e', edgecolor='black')
    axes[0].axvline(target_raw.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {target_raw.mean():.4f}')
    axes[0].axvline(target_raw.median(), color='green', linestyle='--', linewidth=2, label=f'Median: {target_raw.median():.4f}')
    axes[0].set_xlabel('Synergy (CFROA difference)', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Frequency', fontsize=11, fontweight='bold')
    axes[0].set_title(f'Raw Target: synergy_healy1992 (n={len(target_raw)})',
                      fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)

    # Right: winsorised synergy_healy1992_w
    target_win = df2['synergy_healy1992_w'].dropna()
    axes[1].hist(target_win, bins=50, alpha=0.7, color='#2ca02c', edgecolor='black')
    axes[1].axvline(target_win.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {target_win.mean():.4f}')
    axes[1].axvline(target_win.median(), color='green', linestyle='--', linewidth=2, label=f'Median: {target_win.median():.4f}')
    axes[1].set_xlabel('Synergy (CFROA difference)', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Frequency', fontsize=11, fontweight='bold')
    axes[1].set_title(f'Winsorised Target: synergy_healy1992_w (n={len(target_win)})',
                      fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)

    fig.suptitle('Target Variable Distribution: Raw vs. Winsorised',
                 fontsize=14, fontweight='bold', y=1.00)

    _savefig(fig, '05_target_distribution')


# =============================================================================
# FIGURE 6: ALTMAN Z-SCORE DISTRIBUTIONS
# =============================================================================

def fig_altman_distributions(df2):
    """
    Side-by-side distributions: acquiror vs. target Altman Z-scores.
    """
    logger.info("\n[FIG 6] Altman Z-score distributions")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Acquiror
    z_acq = df2['financial_altman_z_acquiror'].dropna()
    axes[0].hist(z_acq, bins=40, alpha=0.7, color='#1f77b4', edgecolor='black')
    axes[0].axvline(z_acq.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {z_acq.mean():.2f}')
    axes[0].axvline(z_acq.median(), color='green', linestyle='--', linewidth=2, label=f'Median: {z_acq.median():.2f}')
    axes[0].set_xlabel('Altman Z-Score', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Frequency', fontsize=11, fontweight='bold')
    axes[0].set_title(f'Acquiror Z-Score (n={len(z_acq)})', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)

    # Target
    z_tgt = df2['financial_altman_z_target'].dropna()
    axes[1].hist(z_tgt, bins=40, alpha=0.7, color='#d62728', edgecolor='black')
    axes[1].axvline(z_tgt.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {z_tgt.mean():.2f}')
    axes[1].axvline(z_tgt.median(), color='green', linestyle='--', linewidth=2, label=f'Median: {z_tgt.median():.2f}')
    axes[1].set_xlabel('Altman Z-Score', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Frequency', fontsize=11, fontweight='bold')
    axes[1].set_title(f'Target Z-Score (n={len(z_tgt)})', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)

    fig.suptitle('Modified Altman Z-Score Distributions (Financial Distress Channel)',
                 fontsize=14, fontweight='bold', y=1.00)

    _savefig(fig, '06_altman_distributions')


# =============================================================================
# FIGURE 7: FEATURE CORRELATION MATRIX
# =============================================================================

def fig_feature_correlations(df2):
    """
    Correlation heatmap: selected features (one per channel + Altman scores).
    """
    logger.info("\n[FIG 7] Feature correlations")

    # Select representative features + Altman + target
    selected_features = [
        # Cost
        'cost_relative_asset_size',
        # Revenue
        'revenue_relative_size_sales',
        # Operational
        'operational_roa_gap',
        # Financial
        'financial_leverage_gap',
        'financial_altman_z_acquiror',
        'financial_altman_z_target',
        # Macro
        'sp500_trailing_12m',
        # Target
        'synergy_healy1992_w',
    ]

    df_sel = df2[selected_features].dropna()
    corr = df_sel.corr()

    fig, ax = plt.subplots(figsize=(10, 9))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                vmin=-1, vmax=1, ax=ax, cbar_kws={'label': 'Pearson r'})

    ax.set_title('Feature Correlation Matrix (Selected Features, ML-Ready Sample)',
                 fontsize=14, fontweight='bold', pad=20)

    _savefig(fig, '07_feature_correlations')


# =============================================================================
# SUMMARY STATISTICS
# =============================================================================

def print_summary_statistics(df0, df1, df2):
    """Print key statistics to console."""
    logger.info("\n" + "=" * 70)
    logger.info("VISUALISATION SUMMARY STATISTICS")
    logger.info("=" * 70)

    logger.info(f"\nPipeline Attrition:")
    logger.info(f"  Stage 0 (raw DAQ)           : {len(df0):>6,} deals")
    logger.info(f"  Stage 1 (feature-engineered): {len(df1):>6,} deals  ({100*len(df1)/len(df0):.1f}%)")
    logger.info(f"  Stage 1 (labeled)            : {df1['synergy_healy1992_w'].notna().sum():>6,} deals  ({100*df1['synergy_healy1992_w'].notna().sum()/len(df0):.1f}%)")
    logger.info(f"  Stage 2 (ML-ready)          : {len(df2):>6,} deals  ({100*len(df2)/len(df0):.1f}%)")

    logger.info(f"\nSplit Distribution:")
    split_counts = df2['split'].value_counts()
    for s in ['train', 'val', 'test']:
        n = int(split_counts.get(s, 0))
        logger.info(f"  {s:<10}: {n:>6,} deals  ({100*n/len(df2):>5.1f}%)")

    logger.info(f"\nTarget Variable (ML-Ready):")
    target = df2['synergy_healy1992_w']
    logger.info(f"  Mean    : {target.mean():>+.4f}")
    logger.info(f"  Median  : {target.median():>+.4f}")
    logger.info(f"  Std dev : {target.std():>+.4f}")
    logger.info(f"  Range   : [{target.min():>+.4f}, {target.max():>+.4f}]")

    logger.info(f"\nAltman Z-Scores (ML-Ready):")
    z_acq = df2['financial_altman_z_acquiror'].dropna()
    z_tgt = df2['financial_altman_z_target'].dropna()
    logger.info(f"  Acquiror Z: mean={z_acq.mean():>+.2f}, median={z_acq.median():>+.2f}, coverage={100*len(z_acq)/len(df2):.1f}%")
    logger.info(f"  Target Z  : mean={z_tgt.mean():>+.2f}, median={z_tgt.median():>+.2f}, coverage={100*len(z_tgt)/len(df2):.1f}%")

    logger.info("\n" + "=" * 70)


# =============================================================================
# MAIN
# =============================================================================

def run_visualisation():
    """Run all visualisations."""
    logger.info("=" * 70)
    logger.info("DATA VISUALISATION PIPELINE")
    logger.info("=" * 70)

    _ensure_output_dir()

    df0, df1, df2 = _load_stages()

    logger.info("\nGenerating figures...")
    fig_pipeline_attrition(df0, df1, df2)
    fig_label_yield_by_year(df1)
    fig_feature_coverage_by_channel(df2)
    fig_split_distribution_over_time(df2)
    fig_target_distribution(df1, df2)
    fig_altman_distributions(df2)
    fig_feature_correlations(df2)

    print_summary_statistics(df0, df1, df2)

    logger.info(f"\nAll figures saved to: {CONFIG['output_dir']}")
    logger.info("✓ VISUALISATION COMPLETE")
    logger.info("=" * 70)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    run_visualisation()
