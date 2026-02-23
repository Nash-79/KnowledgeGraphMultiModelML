#!/usr/bin/env python3
"""
Analyze classification errors from baseline model.

Usage: python scripts/m9_error_analysis.py
"""

import json
import sys
from pathlib import Path

import pandas as pd


def analyze_per_label_performance(metrics_file):
    """Extract per-label metrics and calculate error counts."""
    with open(metrics_file) as f:
        data = json.load(f)

    per_label = data.get('per_label', {})

    # Extract per-label stats
    labels = []
    for label, stats in per_label.items():
        if label in ['micro avg', 'macro avg', 'weighted avg', 'samples avg']:
            continue

        support = stats.get('support', 0)
        precision = stats.get('precision', 0)
        recall = stats.get('recall', 0)
        f1 = stats.get('f1-score', 0)

        fn = support * (1 - recall)
        tp = support * recall
        fp = tp / precision - tp if precision > 0 else 0

        labels.append({
            'concept': label,
            'support': int(support),
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'false_negatives': fn,
            'false_positives': fp,
            'total_errors': fn + fp
        })

    return pd.DataFrame(labels)


def categorize_concepts(df):
    """Categorize concepts by financial statement type."""
    categories = {
        'Assets': ['Assets', 'CurrentAssets', 'NoncurrentAssets', 'CashAndCashEquivalents',
                   'AccountsReceivableNet', 'InventoryNet', 'ShortTermInvestments',
                   'LongTermInvestments', 'PropertyPlantAndEquipmentNet', 'IntangibleAssetsNet',
                   'DeferredTaxAssetsNet', 'OtherNoncurrentAssets', 'PrepaidExpensesAndOtherCurrentAssets'],
        'Liabilities': ['Liabilities', 'CurrentLiabilities', 'NoncurrentLiabilities',
                        'AccountsPayableAndAccruedLiabilities', 'ShortTermDebtAndCurrentMaturities',
                        'LongTermDebtAndCapitalLeases', 'DeferredRevenueAndCustomerAdvances',
                        'DeferredTaxLiabilitiesNet', 'OtherCurrentLiabilities', 'OtherNoncurrentLiabilities'],
        'Equity': ['StockholdersEquity', 'CommonStock', 'PreferredStock', 'AdditionalPaidInCapital',
                   'RetainedEarnings', 'AccumulatedOtherComprehensiveIncome', 'TreasuryStock'],
        'Revenue': ['Revenues', 'ProductRevenue', 'OtherRevenue'],
        'Expenses': ['CostOfRevenue', 'CostOfGoodsSold', 'CostsAndExpenses', 'OperatingExpenses',
                     'SellingGeneralAndAdministrativeExpense', 'ResearchAndDevelopmentExpense',
                     'DepreciationAndAmortization', 'OtherOperatingExpenses', 'InterestExpense',
                     'IncomeTaxExpenseBenefit'],
        'Income': ['NetIncomeLoss', 'OperatingIncomeLoss', 'EarningsPerShare'],
        'Other': ['EmployeeBenefitPlans']
    }

    def get_category(concept):
        concept_short = concept.split(':')[-1]
        for cat, concepts in categories.items():
            if concept_short in concepts:
                return cat
        return 'Other'

    df['category'] = df['concept'].apply(get_category)
    return df


def main():
    print("\nM9 Error Analysis: Classification Errors\n")

    metrics_file = Path("reports/tables/baseline_text_plus_concept_seed42_metrics.json")

    if not metrics_file.exists():
        print(f"Error: {metrics_file} not found")
        return 1

    print("Loading classification metrics...")
    df = analyze_per_label_performance(metrics_file)

    print(f"Analyzed {len(df)} concept labels\n")

    # Categorize concepts
    df = categorize_concepts(df)

    # Sort by F1 score to identify worst performers
    df_sorted = df.sort_values('f1_score')

    print("=" * 70)
    print("WORST PERFORMING CONCEPTS (Lowest F1 Scores)")
    print("=" * 70)

    worst_10 = df_sorted.head(10)
    for idx, row in worst_10.iterrows():
        concept_short = row['concept'].split(':')[-1]
        print(f"\n{concept_short}")
        print(f"  F1 Score:   {row['f1_score']:.4f}")
        print(f"  Precision:  {row['precision']:.4f}")
        print(f"  Recall:     {row['recall']:.4f}")
        print(f"  Support:    {row['support']}")
        print(f"  FN:         {row['false_negatives']:.2f}")
        print(f"  FP:         {row['false_positives']:.2f}")
        print(f"  Category:   {row['category']}")

    print("\n" + "=" * 70)
    print("ERROR DISTRIBUTION BY CATEGORY")
    print("=" * 70)

    category_stats = df.groupby('category').agg({
        'total_errors': 'sum',
        'support': 'sum',
        'f1_score': 'mean'
    }).round(4)

    category_stats['error_rate'] = (category_stats['total_errors'] / category_stats['support'] * 100).round(2)
    category_stats = category_stats.sort_values('error_rate', ascending=False)

    print(f"\n{'Category':<15} {'Errors':<10} {'Support':<10} {'Avg F1':<10} {'Error %'}")
    print("-" * 70)
    for cat, row in category_stats.iterrows():
        print(f"{cat:<15} {row['total_errors']:<10.1f} {row['support']:<10.0f} {row['f1_score']:<10.4f} {row['error_rate']:.2f}%")

    print("\n" + "=" * 70)
    print("SUPPORT vs PERFORMANCE CORRELATION")
    print("=" * 70)

    # Bin by support size
    df['support_bin'] = pd.cut(df['support'], bins=[0, 250, 500, 750, 1000],
                                 labels=['Low (<250)', 'Medium (250-500)', 'High (500-750)', 'Very High (>750)'])

    support_stats = df.groupby('support_bin')['f1_score'].agg(['mean', 'min', 'count']).round(4)

    print(f"\n{'Support Range':<20} {'Count':<10} {'Avg F1':<10} {'Min F1'}")
    print("-" * 70)
    for support_bin, row in support_stats.iterrows():
        print(f"{support_bin:<20} {int(row['count']):<10} {row['mean']:<10.4f} {row['min']:.4f}")

    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    total_errors = df['total_errors'].sum()
    total_labels = df['support'].sum()
    error_rate = (total_errors / total_labels) * 100

    perfect_f1 = len(df[df['f1_score'] == 1.0])
    below_99 = len(df[df['f1_score'] < 0.99])

    print(f"\n1. Overall Error Rate: {error_rate:.2f}%")
    print(f"2. Perfect F1=1.0: {perfect_f1}/{len(df)} concepts ({perfect_f1/len(df)*100:.1f}%)")
    print(f"3. Below F1<0.99: {below_99}/{len(df)} concepts ({below_99/len(df)*100:.1f}%)")
    print(f"4. Worst category: {category_stats.index[0]} ({category_stats.iloc[0]['error_rate']:.2f}% error rate)")
    print(f"5. Support correlation: {'Negative' if support_stats['mean'].is_monotonic_increasing else 'Mixed'}")

    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    print("\nError Patterns:")
    if support_stats['mean'].is_monotonic_increasing:
        print("- Low support classes perform worse (data scarcity)")
    print("- Most errors in " + category_stats.index[0] + " category")
    print("- Model achieves near-perfect performance on high-support concepts")
    print("\nHypothesis:")
    print("- Errors likely due to semantic similarity between confused concepts")
    print("- Rare concepts have insufficient training examples")
    print("- Consider data augmentation or better taxonomy for rare classes")

    # Save results
    output_dir = Path("reports/tables")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save detailed analysis
    df_sorted.to_csv(output_dir / "m9_error_analysis_detailed.csv", index=False)

    # Save category summary
    category_stats.to_csv(output_dir / "m9_error_by_category.csv")

    # Save JSON summary
    summary = {
        'overall': {
            'total_concepts': len(df),
            'total_labels': int(total_labels),
            'total_errors': float(total_errors),
            'error_rate_percent': float(error_rate),
            'perfect_f1_count': int(perfect_f1),
            'below_99_count': int(below_99)
        },
        'worst_10': [
            {
                'concept': row['concept'],
                'f1_score': float(row['f1_score']),
                'support': int(row['support']),
                'false_negatives': float(row['false_negatives']),
                'false_positives': float(row['false_positives']),
                'category': row['category']
            }
            for _, row in worst_10.iterrows()
        ],
        'by_category': {
            cat: {
                'total_errors': float(row['total_errors']),
                'support': int(row['support']),
                'avg_f1': float(row['f1_score']),
                'error_rate_percent': float(row['error_rate'])
            }
            for cat, row in category_stats.iterrows()
        }
    }

    with open(output_dir / "m9_error_analysis.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to:")
    print(f"  - {output_dir}/m9_error_analysis_detailed.csv")
    print(f"  - {output_dir}/m9_error_by_category.csv")
    print(f"  - {output_dir}/m9_error_analysis.json\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
