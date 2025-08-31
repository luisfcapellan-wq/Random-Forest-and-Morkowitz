import streamlit as st
import pandas as pd
import numpy as np
import warnings
from datetime import datetime, timedelta

# Importaciones con verificación
try:
    import yfinance as yf
except ImportError:
    st.error("yfinance no está instalado")
    st.stop()

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error
except ImportError:
    st.error("scikit-learn no está instalado") 
    st.stop()

try:
    from scipy.optimize import minimize
except ImportError:
    st.error("scipy no está instalado")
    st.stop()

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Random Forest + Markowitz Portfolio",
    page_icon="📊",
    layout="wide"
)

# CSS
st.markdown("""
<style>
.main-header {
    background: linear-gradient(90deg, #1f4e79, #2e8b57);
    padding: 20px;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# FUNCIONES

def create_sample_data(tickers, start_date, end_date):
    """Crea datos simulados realistas"""
    np.random.seed(42)  # Para reproducibilidad
    
    n_days = (end_date - start_date).days
    dates = pd.date_range(start=start_date, periods=min(n_days, 500), freq='D')
    
    annual_returns = [0.08, 0.12, 0.10, 0.09, 0.06]
    annual_vols = [0.20, 0.25, 0.18, 0.20, 0.30]
    
    data = {}
    
    for i, ticker in enumerate(tickers[:5]):
        initial_price = 100
        prices = [initial_price]
        
        daily_return = annual_returns[i] / 252
        daily_vol = annual_vols[i] / np.sqrt(252)
        
        for day in range(1, len(dates)):
            price_change = daily_return + daily_vol * np.random.normal()
            new_price = prices[-1] * (1 + price_change)
            prices.append(max(new_price, 10))
        
        data[ticker] = prices[:len(dates)]
    
    return pd.DataFrame(data, index=dates)

@st.cache_data(ttl=3600)
def get_market_data(tickers, start_date, end_date):
    """Obtiene datos con fallback a simulación"""
    
    # Intentar datos reales primero
    try:
        data = {}
        for ticker in tickers[:5]:  # Limitar a 5 para evitar rate limits
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(start=start_date, end=end_date)
                if len(hist) > 100:
                    data[ticker] = hist['Close']
            except:
                continue
        
        if len(data) >= 3:
            df = pd.DataFrame(data).dropna()
            if len(df) > 100:
                return df, "real"
    except:
        pass
    
    # Fallback a datos simulados
    return create_sample_data(tickers, start_date, end_date), "simulado"

def calculate_features(prices):
    """Calcula características técnicas básicas"""
    features = pd.DataFrame(index=prices.index)
    
    for ticker in prices.columns:
        price_series = prices[ticker]
        
        # Momentum
        features[f'{ticker}_mom'] = price_series.pct_change(20)
        
        # Volatilidad
        features[f'{ticker}_vol'] = price_series.pct_change().rolling(20).std()
        
        # Moving average ratio
        features[f'{ticker}_ma'] = price_series / price_series.rolling(50).mean()
        
        # RSI aproximado
        returns = price_series.pct_change()
        gain = returns.where(returns > 0, 0).rolling(14).mean()
        loss = -returns.where(returns < 0, 0).rolling(14).mean()
        rs = gain / loss
        features[f'{ticker}_rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        rolling_mean = price_series.rolling(20).mean()
        rolling_std = price_series.rolling(20).std()
        features[f'{ticker}_bb_upper'] = (price_series - (rolling_mean + 2 * rolling_std)) / rolling_std
        features[f'{ticker}_bb_lower'] = ((rolling_mean - 2 * rolling_std) - price_series) / rolling_std
    
    return features.dropna()

def train_models(X, y):
    """Entrena modelos Random Forest"""
    models = {}
    
    split_idx = int(len(X) * 0.8)
    X_train = X.iloc[:split_idx]
    y_train = y.iloc[:split_idx]
    
    for asset in y.columns:
        try:
            # Filtrar datos válidos
            valid_mask = ~(y_train[asset].isna() | X_train.isna().any(axis=1))
            
            if valid_mask.sum() < 50:
                continue
            
            X_clean = X_train[valid_mask]
            y_clean = y_train[asset][valid_mask]
            
            rf = RandomForestRegressor(
                n_estimators=50,  # Aumentado para mejor precisión
                max_depth=8,      # Aumentado para capturar más complejidad
                min_samples_split=5,
                random_state=42
            )
            
            rf.fit(X_clean, y_clean)
            models[asset] = rf
            
        except Exception as e:
            continue
    
    return models

def predict_returns(models, current_features):
    """Predice rendimientos usando los modelos Random Forest"""
    predicted_returns = {}
    
    for asset, model in models.items():
        try:
            # Asegurar que las características estén en el orden correcto
            prediction = model.predict(current_features.values.reshape(1, -1))[0]
            predicted_returns[asset] = prediction
        except Exception as e:
            predicted_returns[asset] = 0  # Fallback a 0 si hay error
    
    return predicted_returns

def optimize_portfolio(expected_returns, cov_matrix, risk_aversion=2):
    """Optimización de Markowitz usando rendimientos esperados"""
    n_assets = len(expected_returns)
    
    def objective(weights):
        ret = np.dot(weights, expected_returns)
        risk = np.dot(weights, np.dot(cov_matrix, weights))
        return -(ret - risk_aversion * risk)
    
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    bounds = [(0.05, 0.4) for _ in range(n_assets)]
    
    try:
        result = minimize(
            objective,
            x0=np.ones(n_assets) / n_assets,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if result.success:
            return result.x
        else:
            return np.ones(n_assets) / n_assets
    except:
        return np.ones(n_assets) / n_assets

def calculate_metrics(returns_list):
    """Calcula métricas de performance"""
    if len(returns_list) == 0:
        return {}
    
    returns_array = np.array(returns_list)
    
    total_ret = np.prod(1 + returns_array) - 1
    mean_ret = np.mean(returns_array)
    vol = np.std(returns_array)
    
    annual_ret = (1 + mean_ret)**252 - 1
    annual_vol = vol * np.sqrt(252)
    sharpe = annual_ret / annual_vol if annual_vol > 0 else 0
    
    return {
        'Total Return': total_ret,
        'Annual Return': annual_ret,
        'Annual Volatility': annual_vol,
        'Sharpe Ratio': sharpe,
        'Win Rate': np.mean(returns_array > 0)
    }

# APLICACIÓN PRINCIPAL
def main():
    st.markdown("""
    <div class="main-header">
        <h1>📊 Random Forest + Markowitz Portfolio</h1>
        <p>Práctica Académica - Construcción de Portafolios Inteligentes</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("Configuración")
    
    asset_universe = st.sidebar.selectbox(
        "Universo de inversión:",
        ["ETFs Sectoriales", "Tech Stocks"]
    )
    
    if asset_universe == "ETFs Sectoriales":
        tickers = ['XLF', 'XLK', 'XLV', 'XLI', 'XLE']
        ticker_names = {
            'XLF': 'Financials',
            'XLK': 'Technology', 
            'XLV': 'Healthcare',
            'XLI': 'Industrials',
            'XLE': 'Energy'
        }
    else:
        tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META']
        ticker_names = {t: t for t in tickers}
    
    end_date = st.sidebar.date_input("Fecha final:", datetime.now().date())
    start_date = st.sidebar.date_input("Fecha inicial:", end_date - timedelta(days=730))  # 2 años para más datos
    
    prediction_horizon = st.sidebar.slider("Horizonte predicción (días):", 5, 42, 21)
    risk_aversion = st.sidebar.slider("Aversión al riesgo:", 0.5, 5.0, 2.0)
    
    run_analysis = st.sidebar.button("🚀 Ejecutar Análisis", type="primary")
    
    if not run_analysis:
        st.markdown("""
        ## Bienvenido a la Práctica Académica
        
        **Esta aplicación implementa:**
        
        - **Random Forest**: Predicción de rendimientos futuros usando características técnicas
        - **Optimización de Markowitz**: Construcción de portafolios eficientes
        - **Backtesting**: Evaluación histórica de la estrategia
        
        Configure los parámetros en la barra lateral y presione "Ejecutar Análisis"
        """)
        return
    
    # ANÁLISIS
    
    st.subheader("📊 Obtención de Datos")
    
    with st.spinner("Descargando datos del mercado..."):
        asset_prices, data_type = get_market_data(tickers, start_date, end_date)
    
    if asset_prices.empty:
        st.error("No se pudieron obtener datos")
        return
    
    if data_type == "simulado":
        st.info("Usando datos simulados para demostración")
    else:
        st.success("Datos reales descargados exitosamente")
    
    st.write(f"**Dataset**: {len(asset_prices)} días, {asset_prices.shape[1]} activos")
    
    # Mostrar datos básicos
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Precios Actuales:**")
        final_prices_df = pd.DataFrame({
            'Precio': asset_prices.iloc[-1].round(2)
        })
        st.dataframe(final_prices_df)
    
    with col2:
        returns = asset_prices.pct_change().dropna()
        annual_returns = returns.mean() * 252
        st.write("**Rendimientos Anualizados:**")
        annual_df = pd.DataFrame({
            'Rendimiento Anual': (annual_returns * 100).round(2)
        })
        annual_df['Rendimiento Anual'] = annual_df['Rendimiento Anual'].astype(str) + '%'
        st.dataframe(annual_df)
    
    # Feature Engineering
    st.subheader("🔧 Creación de Características")
    
    with st.spinner("Calculando indicadores técnicos..."):
        features = calculate_features(asset_prices)
        
        # Crear targets
        targets = pd.DataFrame(index=returns.index, columns=returns.columns)
        for col in returns.columns:
            targets[col] = returns[col].shift(-prediction_horizon)
        
        # Alinear fechas
        common_dates = features.index.intersection(targets.index)
        X = features.loc[common_dates].fillna(method='ffill').fillna(0)
        y = targets.loc[common_dates]
    
    st.success(f"Características creadas: {X.shape[1]} features, {len(X)} observaciones")
    
    # Entrenamiento
    st.subheader("🤖 Entrenamiento Random Forest")
    
    with st.spinner("Entrenando modelos..."):
        models = train_models(X, y)
    
    if len(models) == 0:
        st.error("No se pudieron entrenar modelos")
        return
    
    st.success(f"Modelos entrenados: {len(models)}/{len(tickers)}")
    
    # Mostrar importancia de características para un modelo de ejemplo
    if len(models) > 0:
        example_asset = list(models.keys())[0]
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': models[example_asset].feature_importances_
        }).sort_values('Importance', ascending=False).head(10)
        
        st.write("**Importancia de Características (Ejemplo):**")
        st.dataframe(feature_importance)
    
    # Backtesting
    st.subheader("🔄 Backtesting con Random Forest")
    
    with st.spinner("Ejecutando backtesting con predicciones RF..."):
        
        # Configuración
        split_idx = int(len(returns) * 0.7)  # 70% para entrenamiento
        test_period = returns.iloc[split_idx:]
        
        portfolio_rets = []
        benchmark_rets = []
        all_weights = []
        rf_predictions_history = []
        historical_predictions_history = []
        
        n_periods = min(6, len(test_period) // 30)  # Máximo 6 períodos
        if n_periods == 0:
            n_periods = 1
        period_length = len(test_period) // n_periods
        
        for period in range(n_periods):
            start_p = period * period_length
            end_p = min((period + 1) * period_length, len(test_period))
            
            if end_p <= start_p + prediction_horizon:
                break
            
            # Obtener características más recientes para predicción
            current_date = test_period.index[start_p]
            current_features_row = features.loc[current_date]
            
            # PREDICCIÓN CON RANDOM FOREST
            rf_predicted_returns = predict_returns(models, current_features_row)
            rf_expected_returns = np.array([rf_predicted_returns.get(ticker, 0) for ticker in tickers])
            
            # Predicción histórica (para comparación)
            hist_data = returns.iloc[:split_idx + start_p]
            historical_expected_returns = hist_data.tail(63).mean().values * 252
            
            # Matriz de covarianzas (usando datos históricos)
            cov_matrix = hist_data.tail(126).cov().values * 252
            
            # Optimizar usando predicciones de Random Forest
            weights = optimize_portfolio(rf_expected_returns, cov_matrix, risk_aversion)
            all_weights.append(weights)
            
            # Guardar predicciones para análisis
            rf_predictions_history.append(rf_expected_returns)
            historical_predictions_history.append(historical_expected_returns)
            
            # Calcular rendimientos del período
            period_data = test_period.iloc[start_p:end_p]
            
            for _, day_returns in period_data.iterrows():
                port_ret = np.sum(weights * day_returns.values)
                bench_ret = np.mean(day_returns.values)
                
                portfolio_rets.append(port_ret)
                benchmark_rets.append(bench_ret)
        
        final_weights = np.mean(all_weights, axis=0) if all_weights else np.ones(len(tickers)) / len(tickers)
    
    # Verificar datos
    if len(portfolio_rets) == 0:
        st.error("No se generaron datos de backtesting")
        return
    
    st.success(f"Backtesting completado: {len(portfolio_rets)} días analizados, {n_periods} rebalanceos")
    
    # Mostrar comparación de predicciones
    if rf_predictions_history:
        st.subheader("📈 Comparación de Predicciones")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Últimas Predicciones RF (% anualizado):**")
            last_rf_pred = rf_predictions_history[-1] * 252 * 100
            rf_pred_df = pd.DataFrame({
                'Activo': tickers,
                'Predicción RF': [f"{p:.2f}%" for p in last_rf_pred]
            })
            st.dataframe(rf_pred_df, hide_index=True)
        
        with col2:
            st.write("**Últimas Predicciones Históricas (% anualizado):**")
            last_hist_pred = historical_predictions_history[-1] * 100
            hist_pred_df = pd.DataFrame({
                'Activo': tickers,
                'Predicción Histórica': [f"{p:.2f}%" for p in last_hist_pred]
            })
            st.dataframe(hist_pred_df, hide_index=True)
    
    # RESULTADOS
    st.subheader("📊 Resultados del Análisis")
    
    # Calcular métricas
    port_metrics = calculate_metrics(portfolio_rets)
    bench_metrics = calculate_metrics(benchmark_rets)
    
    # Mostrar métricas
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🤖 Random Forest + Markowitz")
        for key, value in port_metrics.items():
            if 'Return' in key or 'Volatility' in key or 'Rate' in key:
                st.metric(key, f"{value:.2%}")
            else:
                st.metric(key, f"{value:.3f}")
    
    with col2:
        st.markdown("### 📈 Benchmark (Equal Weight)")
        for key, value in bench_metrics.items():
            if 'Return' in key or 'Volatility' in key or 'Rate' in key:
                st.metric(key, f"{value:.2%}")
            else:
                st.metric(key, f"{value:.3f}")
    
    # COMPOSICIÓN DEL PORTAFOLIO
    st.subheader("💼 Composición Recomendada del Portafolio")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Distribución de Pesos:**")
        
        weights_df = pd.DataFrame({
            'Activo': tickers,
            'Descripción': [ticker_names.get(t, t) for t in tickers],
            'Peso (%)': [f"{w*100:.1f}%" for w in final_weights],
            'Inversión $1,000': [f"${w*1000:.0f}" for w in final_weights],
            'Inversión $10,000': [f"${w*10000:.0f}" for w in final_weights]
        })
        
        st.dataframe(weights_df, hide_index=True)
    
    with col2:
        st.write("**Gráfico de Composición:**")
        
        # Crear gráfico de pie simple
        weights_chart = pd.DataFrame({
            'Activo': tickers,
            'Peso': final_weights
        })
        
        # Usar solo activos con peso significativo
        significant_weights = weights_chart[weights_chart['Peso'] > 0.05]
        
        if len(significant_weights) > 0:
            try:
                st.bar_chart(significant_weights.set_index('Activo')['Peso'])
            except:
                st.dataframe(significant_weights)
        else:
            st.write("Todos los pesos son muy pequeños")
    
    # PERFORMANCE VISUAL
    st.subheader("📈 Performance Comparativa")
    
    try:
        # Calcular performance acumulada
        port_cumulative = np.cumprod(1 + np.array(portfolio_rets))
        bench_cumulative = np.cumprod(1 + np.array(benchmark_rets))
        
        # Crear DataFrame para gráfico
        performance_df = pd.DataFrame({
            'RF_Markowitz': port_cumulative,
            'Benchmark': bench_cumulative
        })
        
        st.line_chart(performance_df)
        
    except Exception as e:
        st.error(f"Error creando gráfico de performance: {str(e)}")
        
        # Mostrar métricas numéricas
        final_port_value = np.prod(1 + np.array(portfolio_rets))
        final_bench_value = np.prod(1 + np.array(benchmark_rets))
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Valor Final Portafolio", f"${final_port_value:.3f}")
        with col2:
            st.metric("Valor Final Benchmark", f"${final_bench_value:.3f}")
    
    # ANÁLISIS DE VALOR AÑADIDO
    st.subheader("⭐ Valor Añadido de la Estrategia")
    
    excess_returns = np.array(portfolio_rets) - np.array(benchmark_rets)
    alpha_daily = np.mean(excess_returns)
    alpha_annual = alpha_daily * 252
    
    outperform_days = np.sum(np.array(portfolio_rets) > np.array(benchmark_rets))
    total_days = len(portfolio_rets)
    outperform_rate = outperform_days / total_days
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Alfa Anualizada", f"{alpha_annual:.2%}")
    
    with col2:
        st.metric("Días que Superó Benchmark", f"{outperform_days}/{total_days}")
    
    with col3:
        st.metric("Tasa de Éxito", f"{outperform_rate:.1%}")
    
    # CONCLUSIONES
    st.subheader("🎯 Conclusiones y Recomendaciones")
    
    excess_total = port_metrics.get('Total Return', 0) - bench_metrics.get('Total Return', 0)
    sharpe_diff = port_metrics.get('Sharpe Ratio', 0) - bench_metrics.get('Sharpe Ratio', 0)
    
    if excess_total > 0 and sharpe_diff > 0:
        st.success("🏆 **ESTRATEGIA EXITOSA**: La combinación Random Forest + Markowitz superó al benchmark")
        
        st.markdown("""
        **Por qué funcionó:**
        - Predicciones más precisas usando Machine Learning
        - Optimización matemática de riesgo-retorno basada en señales predictivas
        - Diversificación inteligente basada en correlaciones y predicciones
        """)
        
    elif excess_total > 0:
        st.warning("⚠️ **ÉXITO PARCIAL**: Mayor retorno pero sin mejora significativa en ratio de Sharpe")
        st.markdown("""
        **Posibles razones:**
        - Las predicciones de RF capturaron retornos pero con mayor volatilidad
        - Puede necesitar ajuste de parámetros de riesgo
        """)
        
    else:
        st.info("📊 **ANÁLISIS COMPLETO**: En este período específico, el benchmark tuvo mejor performance")
        
        st.markdown("""
        **Posibles razones:**
        - Período de prueba limitado
        - Mercados eficientes donde es difícil predecir
        - Parámetros del modelo requieren ajuste
        - Las predicciones de RF no fueron efectivas en este período
        """)
    
    # INSTRUCCIONES PRÁCTICAS
    st.subheader("💡 Cómo Usar Estos Resultados")
    
    st.markdown(f"""
    **Para implementar esta estrategia:**
    
    1. **Distribución recomendada** (basada en el análisis):
    """)
    
    for i, (ticker, weight) in enumerate(zip(tickers, final_weights)):
        if weight > 0.05:  # Solo mostrar pesos significativos
            st.markdown(f"   - **{ticker}** ({ticker_names[ticker]}): {weight*100:.1f}% de tu capital")
    
    st.markdown("""
    2. **Rebalanceo**: Revisar y ajustar mensualmente usando las últimas predicciones de RF
    3. **Monitoreo**: Seguir las métricas de Sharpe Ratio y drawdown
    4. **Ajustes**: Modificar aversión al riesgo según tu perfil
    """)
    
    # Exportar resultados
    if st.button("💾 Generar Reporte de Inversión"):
        
        report = f"""
REPORTE DE PORTAFOLIO - {datetime.now().strftime('%Y-%m-%d')}
{'='*60}

CONFIGURACIÓN:
- Universo: {asset_universe}
- Período análisis: {start_date} a {end_date}
- Horizonte predicción: {prediction_horizon} días
- Aversión al riesgo: {risk_aversion}
- Tipo de datos: {data_type}
- Modelo: Random Forest + Markowitz

COMPOSICIÓN RECOMENDADA:
"""
        
        for ticker, weight in zip(tickers, final_weights):
            if weight > 0.01:
                report += f"- {ticker} ({ticker_names[ticker]}): {weight*100:.1f}%\n"
        
        report += f"""

PERFORMANCE:
- Retorno total estrategia: {port_metrics.get('Total Return', 0):.2%}
- Retorno total benchmark: {bench_metrics.get('Total Return', 0):.2%}
- Sharpe Ratio estrategia: {port_metrics.get('Sharpe Ratio', 0):.3f}
- Sharpe Ratio benchmark: {bench_metrics.get('Sharpe Ratio', 0):.3f}
- Alfa anualizada: {alpha_annual:.2%}
- Tasa de éxito: {outperform_rate:.1%}

PREDICCIONES RF ACTUALES (ANUALIZADAS):
"""
        
        if rf_predictions_history:
            last_pred = rf_predictions_history[-1] * 252 * 100
            for ticker, pred in zip(tickers, last_pred):
                report += f"- {ticker}: {pred:.2f}%\n"
        
        report += f"""

INSTRUCCIONES DE INVERSIÓN:
Para $1,000:
"""
        
        for ticker, weight in zip(tickers, final_weights):
            if weight > 0.05:
                amount = weight * 1000
                report += f"- Comprar ${amount:.0f} en {ticker}\n"
        
        st.download_button(
            label="📋 Descargar Reporte Completo",
            data=report,
            file_name=f"portfolio_report_{datetime.now().strftime('%Y%m%d')}.txt",
            mime="text/plain"
        )

if __name__ == "__main__":
    main()
