from src.data_preprocessing import load_data, prepare_data_for_analysis, prepare_data_for_modeling
from src.analysis import plot_correlation_heatmap, plot_turnover_analysis, generate_insights
from src.modeling import train_model, evaluate_model

def main():
    # Load and preprocess data
    df = load_data()
    
    # Analysis
    df_numeric = prepare_data_for_analysis(df)
    plot_correlation_heatmap(df_numeric)
    plot_turnover_analysis(df)
    generate_insights(df)

    
    # Modeling
    X_scaled, y, feature_names, scaler = prepare_data_for_modeling(df)
    model, X_train, X_test, y_train, y_test = train_model(X_scaled, y)
    evaluate_model(model, X_test, y_test, feature_names)

if __name__ == "__main__":
    main()

    