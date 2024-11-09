import matplotlib.pyplot as plt
import seaborn as sns

def plot_correlation_heatmap(df_numeric):
    """Plot correlation heatmap"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(df_numeric.corr(), annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Heatmap of HR Variables')
    plt.tight_layout()
    plt.savefig('outputs/figures/correlation_heatmap.png')
    plt.close()

def plot_turnover_analysis(df):
    """Create various turnover analysis plots"""
    # Department turnover
    plt.figure(figsize=(10, 6))
    dept_turnover = df.groupby('Department')['left'].mean().sort_values(ascending=False)
    sns.barplot(x=dept_turnover.index, y=dept_turnover.values)
    plt.title('Turnover Rate by Department')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('outputs/figures/department_turnover.png')
    plt.close()
    
    # Add other plots here...

def generate_insights(df):
    """Generate statistical insights"""
    print("\n=== KEY INSIGHTS FROM TURNOVER ANALYSIS ===")
    
    # Department analysis
    dept_turnover = df.groupby('Department')['left'].mean().sort_values(ascending=False)
    print(f"\n1. Department Insights:")
    print(f"- Highest turnover department: {dept_turnover.index[0]} ({dept_turnover.values[0]:.2%})")
    print(f"- Lowest turnover department: {dept_turnover.index[-1]} ({dept_turnover.values[-1]:.2%})")
    
    # Add other insights here...