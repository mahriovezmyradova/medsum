"""
Statistical analysis for ASR evaluation results.
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
import logging
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import warnings
warnings.filterwarnings("ignore")

from ..utils.logger import get_logger

logger = get_logger(__name__)

class StatisticalAnalyzer:
    """Statistical analysis for ASR model comparisons."""
    
    def __init__(self, alpha: float = 0.05):
        """
        Initialize statistical analyzer.
        
        Args:
            alpha: Significance level
        """
        self.alpha = alpha
        logger.info(f"Initialized StatisticalAnalyzer with alpha={alpha}")
    
    def paired_t_test(self, model1_scores: List[float], model2_scores: List[float], 
                     metric: str = "wer") -> Dict[str, Any]:
        """
        Perform paired t-test between two models.
        
        Args:
            model1_scores: Scores from model 1
            model2_scores: Scores from model 2
            metric: Metric being tested
            
        Returns:
            Dictionary with test results
        """
        # Ensure same length
        min_len = min(len(model1_scores), len(model2_scores))
        model1_scores = model1_scores[:min_len]
        model2_scores = model2_scores[:min_len]
        
        # Calculate differences
        differences = [m1 - m2 for m1, m2 in zip(model1_scores, model2_scores)]
        
        # Perform t-test
        t_stat, p_value = stats.ttest_rel(model1_scores, model2_scores)
        
        # Calculate effect size (Cohen's d)
        mean_diff = np.mean(differences)
        std_diff = np.std(differences, ddof=1)
        cohens_d = mean_diff / std_diff if std_diff > 0 else 0
        
        # Determine significance
        significant = p_value < self.alpha
        
        results = {
            "test": "paired_t_test",
            "metric": metric,
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "significant": significant,
            "mean_difference": float(mean_diff),
            "std_difference": float(std_diff),
            "cohens_d": float(cohens_d),
            "effect_size": self._interpret_effect_size(cohens_d),
            "sample_size": min_len,
            "alpha": self.alpha
        }
        
        self._log_test_results(results)
        
        return results
    
    def wilcoxon_signed_rank_test(self, model1_scores: List[float], 
                                 model2_scores: List[float], metric: str = "wer") -> Dict[str, Any]:
        """
        Perform Wilcoxon signed-rank test (non-parametric alternative to paired t-test).
        
        Args:
            model1_scores: Scores from model 1
            model2_scores: Scores from model 2
            metric: Metric being tested
            
        Returns:
            Dictionary with test results
        """
        # Ensure same length
        min_len = min(len(model1_scores), len(model2_scores))
        model1_scores = model1_scores[:min_len]
        model2_scores = model2_scores[:min_len]
        
        # Perform Wilcoxon test
        stat, p_value = stats.wilcoxon(model1_scores, model2_scores)
        
        # Calculate effect size (r = Z / sqrt(N))
        z_stat = stats.norm.ppf(p_value / 2) if p_value > 0 else 0
        effect_size_r = abs(z_stat) / np.sqrt(min_len)
        
        # Determine significance
        significant = p_value < self.alpha
        
        results = {
            "test": "wilcoxon_signed_rank_test",
            "metric": metric,
            "statistic": float(stat),
            "p_value": float(p_value),
            "significant": significant,
            "effect_size_r": float(effect_size_r),
            "effect_size": self._interpret_effect_size(effect_size_r),
            "sample_size": min_len,
            "alpha": self.alpha
        }
        
        self._log_test_results(results)
        
        return results
    
    def anova_multiple_models(self, model_scores: Dict[str, List[float]], 
                            metric: str = "wer") -> Dict[str, Any]:
        """
        Perform ANOVA for comparing multiple models.
        
        Args:
            model_scores: Dictionary of model_name: scores
            metric: Metric being tested
            
        Returns:
            Dictionary with ANOVA results
        """
        # Prepare data for ANOVA
        all_scores = []
        groups = []
        
        for model_name, scores in model_scores.items():
            all_scores.extend(scores)
            groups.extend([model_name] * len(scores))
        
        # Perform one-way ANOVA
        f_stat, p_value = stats.f_oneway(*model_scores.values())
        
        # Calculate eta squared (effect size)
        df_between = len(model_scores) - 1
        df_within = len(all_scores) - len(model_scores)
        
        ss_between = 0
        grand_mean = np.mean(all_scores)
        
        for scores in model_scores.values():
            group_mean = np.mean(scores)
            ss_between += len(scores) * (group_mean - grand_mean) ** 2
        
        ss_total = np.sum((np.array(all_scores) - grand_mean) ** 2)
        eta_squared = ss_between / ss_total if ss_total > 0 else 0
        
        # Determine significance
        significant = p_value < self.alpha
        
        results = {
            "test": "one_way_anova",
            "metric": metric,
            "f_statistic": float(f_stat),
            "p_value": float(p_value),
            "significant": significant,
            "eta_squared": float(eta_squared),
            "effect_size": self._interpret_effect_size(np.sqrt(eta_squared)),
            "df_between": int(df_between),
            "df_within": int(df_within),
            "num_models": len(model_scores),
            "alpha": self.alpha
        }
        
        self._log_test_results(results)
        
        # If significant, perform post-hoc tests
        if significant and len(model_scores) > 2:
            results["post_hoc"] = self._perform_post_hoc_analysis(all_scores, groups)
        
        return results
    
    def _perform_post_hoc_analysis(self, scores: List[float], groups: List[str]) -> Dict[str, Any]:
        """Perform Tukey's HSD post-hoc analysis."""
        try:
            # Perform Tukey's HSD
            tukey = pairwise_tukeyhsd(scores, groups, alpha=self.alpha)
            
            # Convert to DataFrame for better readability
            tukey_df = pd.DataFrame(
                data=tukey.summary().data[1:],  # Skip header
                columns=tukey.summary().data[0]
            )
            
            # Convert to dictionary
            post_hoc_results = {
                "method": "tukey_hsd",
                "comparisons": tukey_df.to_dict(orient='records'),
                "reject": tukey.reject.tolist(),
                "meandiffs": tukey.meandiffs.tolist(),
                "pvalues": tukey.pvalues.tolist(),
                "conf_low": tukey.confint[:, 0].tolist(),
                "conf_high": tukey.confint[:, 1].tolist()
            }
            
            return post_hoc_results
            
        except Exception as e:
            logger.warning(f"Failed to perform post-hoc analysis: {e}")
            return {"error": str(e)}
    
    def _interpret_effect_size(self, effect_size: float) -> str:
        """Interpret effect size magnitude."""
        if effect_size < 0.2:
            return "negligible"
        elif effect_size < 0.5:
            return "small"
        elif effect_size < 0.8:
            return "medium"
        else:
            return "large"
    
    def _log_test_results(self, results: Dict[str, Any]):
        """Log statistical test results."""
        logger.info(f"Statistical Test: {results['test']}")
        logger.info(f"  Metric: {results['metric']}")
        logger.info(f"  p-value: {results['p_value']:.6f}")
        logger.info(f"  Significant (α={self.alpha}): {results['significant']}")
        
        if 'effect_size' in results:
            logger.info(f"  Effect size: {results['effect_size']}")
        
        if results['significant']:
            logger.info("  → Statistically significant difference found")
        else:
            logger.info("  → No statistically significant difference")
        logger.info("")
    
    def analyze_model_comparison(self, results_df: pd.DataFrame, 
                               metric: str = "wer") -> Dict[str, Any]:
        """
        Comprehensive statistical analysis of model comparison.
        
        Args:
            results_df: DataFrame with model comparison results
            metric: Metric to analyze
            
        Returns:
            Dictionary with comprehensive analysis
        """
        # Extract model scores (assuming each row is a sample)
        # This would require the per-sample scores, not just aggregated metrics
        # For now, we'll analyze the aggregated metrics
        
        analysis = {
            "descriptive_statistics": self._compute_descriptive_stats(results_df, metric),
            "ranking_analysis": self._compute_model_ranking(results_df, metric),
            "correlation_analysis": self._analyze_correlations(results_df),
            "best_model": results_df.loc[results_df[metric].idxmin(), "model"] if metric in results_df.columns else None,
            "worst_model": results_df.loc[results_df[metric].idxmax(), "model"] if metric in results_df.columns else None
        }
        
        return analysis
    
    def _compute_descriptive_stats(self, df: pd.DataFrame, metric: str) -> Dict[str, Any]:
        """Compute descriptive statistics for a metric."""
        if metric not in df.columns:
            return {}
        
        values = df[metric].dropna()
        
        return {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "std": float(np.std(values, ddof=1)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "q1": float(np.percentile(values, 25)),
            "q3": float(np.percentile(values, 75)),
            "iqr": float(np.percentile(values, 75) - np.percentile(values, 25)),
            "cv": float(np.std(values, ddof=1) / np.mean(values) if np.mean(values) > 0 else 0)
        }
    
    def _compute_model_ranking(self, df: pd.DataFrame, metric: str) -> Dict[str, Any]:
        """Compute model ranking based on a metric."""
        if metric not in df.columns:
            return {}
        
        # Sort by metric (lower is better for error rates)
        sorted_df = df.sort_values(metric)
        
        rankings = {}
        for rank, (_, row) in enumerate(sorted_df.iterrows(), 1):
            rankings[row["model"]] = {
                "rank": rank,
                metric: row[metric],
                "percentile": (rank - 1) / len(sorted_df) * 100
            }
        
        return rankings
    
    def _analyze_correlations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze correlations between different metrics."""
        # Select numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return {}
        
        # Compute correlation matrix
        corr_matrix = df[numeric_cols].corr()
        
        # Find strong correlations (|r| > 0.7)
        strong_correlations = []
        for i in range(len(numeric_cols)):
            for j in range(i + 1, len(numeric_cols)):
                corr = corr_matrix.iloc[i, j]
                if abs(corr) > 0.7:
                    strong_correlations.append({
                        "metric1": numeric_cols[i],
                        "metric2": numeric_cols[j],
                        "correlation": float(corr),
                        "interpretation": "strong positive" if corr > 0 else "strong negative"
                    })
        
        return {
            "correlation_matrix": corr_matrix.to_dict(),
            "strong_correlations": strong_correlations
        }
    
    def generate_statistical_report(self, analysis_results: Dict[str, Any]) -> str:
        """Generate a textual statistical report."""
        report = []
        report.append("=" * 70)
        report.append("STATISTICAL ANALYSIS REPORT")
        report.append("=" * 70)
        report.append("")
        
        # Descriptive statistics
        if "descriptive_statistics" in analysis_results:
            stats = analysis_results["descriptive_statistics"]
            if stats:
                report.append("DESCRIPTIVE STATISTICS")
                report.append("-" * 40)
                for key, value in stats.items():
                    report.append(f"{key:15}: {value:.4f}")
                report.append("")
        
        # Model ranking
        if "ranking_analysis" in analysis_results:
            rankings = analysis_results["ranking_analysis"]
            if rankings:
                report.append("MODEL RANKING")
                report.append("-" * 40)
                for model, data in rankings.items():
                    report.append(f"{model:20}: Rank {data['rank']} (percentile: {data['percentile']:.1f}%)")
                report.append("")
        
        # Best/Worst models
        if "best_model" in analysis_results and analysis_results["best_model"]:
            report.append("PERFORMANCE SUMMARY")
            report.append("-" * 40)
            report.append(f"Best model:  {analysis_results['best_model']}")
            if "worst_model" in analysis_results:
                report.append(f"Worst model: {analysis_results['worst_model']}")
            report.append("")
        
        # Strong correlations
        if "correlation_analysis" in analysis_results:
            corr_analysis = analysis_results["correlation_analysis"]
            if "strong_correlations" in corr_analysis and corr_analysis["strong_correlations"]:
                report.append("STRONG CORRELATIONS BETWEEN METRICS")
                report.append("-" * 40)
                for corr in corr_analysis["strong_correlations"]:
                    report.append(f"{corr['metric1']} ↔ {corr['metric2']}: "
                                f"r = {corr['correlation']:.3f} ({corr['interpretation']})")
                report.append("")
        
        report.append("=" * 70)
        
        return "\n".join(report)