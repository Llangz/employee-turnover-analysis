import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pandas import DataFrame
import logging
from pathlib import Path

# Set up colorblind-friendly palette
COLORBLIND_PALETTE = sns.color_palette("colorblind")
OUTPUT_DIR = Path('outputs/figures')

def plot_correlation_heatmap(df_numeric: DataFrame, 
                           fig_size: tuple[int, int] = (10, 8)) -> None:
    """
    Plot correlation heatmap for HR variables with improved visibility.
    
    Args:
        df_numeric: DataFrame containing numeric HR variables
        fig_size: Tuple specifying figure dimensions
        
    Returns:
        None - Saves plot to outputs/figures directory
    """
    plt.figure(figsize=fig_size)
    mask = np.triu(np.ones_like(df_numeric.corr()))
    sns.heatmap(df_numeric.corr(), 
                annot=True, 
                mask=mask,
                cmap='RdBu_r',  # Colorblind-friendly diverging palette
                center=0,
                fmt='.2f',
                square=True)
    plt.title('Correlation Analysis of HR Metrics')
    plt.tight_layout()
    
    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_DIR / 'correlation_heatmap.png', dpi=300)
    plt.close()

def plot_turnover_analysis(df: DataFrame) -> None:
    """
    Create comprehensive turnover analysis visualizations.
    
    Args:
        df: DataFrame containing HR data with 'Department' and 'left' columns
        
    Returns:
        None - Saves plots to outputs/figures directory
    """
    # Department turnover with error bars
    plt.figure(figsize=(12, 6))
    dept_turnover = df.groupby('Department')['left'].agg(['mean', 'std']).sort_values('mean', ascending=False)
    
    sns.barplot(x=dept_turnover.index, 
                y=dept_turnover['mean'],
                yerr=dept_turnover['std'],
                palette=COLORBLIND_PALETTE)
    
    plt.title('Employee Turnover Rate by Department')
    plt.xlabel('Department')
    plt.ylabel('Turnover Rate')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_DIR / 'department_turnover.png', dpi=300)
    plt.close()

def generate_insights(df: DataFrame) -> dict:
    """
    Generate statistical insights from HR turnover data.
    
    Args:
        df: DataFrame containing HR data
        
    Returns:
        dict: Dictionary containing key turnover insights
    """
    insights = {}
    
    # Department analysis with confidence intervals
    dept_stats = df.groupby('Department')['left'].agg(['mean', 'std', 'count'])
    dept_stats['ci'] = 1.96 * dept_stats['std'] / np.sqrt(dept_stats['count'])
    
    highest_dept = dept_stats.sort_values('mean', ascending=False).index[0]
    lowest_dept = dept_stats.sort_values('mean').index[0]
    
    insights['department'] = {
        'highest_turnover': {
            'name': highest_dept,
            'rate': dept_stats.loc[highest_dept, 'mean'],
            'ci': dept_stats.loc[highest_dept, 'ci']
        },
        'lowest_turnover': {
            'name': lowest_dept,
            'rate': dept_stats.loc[lowest_dept, 'mean'],
            'ci': dept_stats.loc[lowest_dept, 'ci']
        }
    }
    
    # Log insights
    logging.info("\n=== KEY INSIGHTS FROM TURNOVER ANALYSIS ===")
    logging.info(f"Highest turnover department: {highest_dept} "
                f"({dept_stats.loc[highest_dept, 'mean']:.2%} ± {dept_stats.loc[highest_dept, 'ci']:.2%})")
    logging.info(f"Lowest turnover department: {lowest_dept} "
                f"({dept_stats.loc[lowest_dept, 'mean']:.2%} ± {dept_stats.loc[lowest_dept, 'ci']:.2%})")
    
    return insights