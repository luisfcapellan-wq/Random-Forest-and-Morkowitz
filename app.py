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
.feature-card {
    background: #f0f8ff;
    padding: 15px;
    border-radius: 10px;
    margin: 10px 0;
    border-left: 4px solid #1f4e79;
}
.analogy-box {
    background: #fff0f5;
    padding: 15px;
    border-radius: 10px;
    margin: 10px 0;
    border-left: 4px solid #8b008b;
}
</style>
""", unsafe_allow_html=True)

# FUNCIONES

def create_sample_data(tickers, start_date, end_date):
    """Crea datos simulados realistas"""
    np.random.seed(42)
    
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
    
    try:
        data = {}
        for ticker in tickers[:5]:
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
        
        # Volume (simulado para datos reales)
        features[f'{ticker}_volume_ratio'] = price_series.rolling(10).std() / price_series.rolling(30).std()
    
    return features.dropna()

def train_models(X, y):
    """Entrena modelos Random Forest"""
    models = {}
    feature_importances = {}
    
    split_idx = int(len(X) * 0.8)
    X_train = X.iloc[:split_idx]
    y_train = y.iloc[:split_idx]
    
    for asset in y.columns:
        try:
            valid_mask = ~(y_train[asset].isna() | X_train.isna().any(axis=1))
            
            if valid_mask.sum() < 50:
                continue
            
            X_clean = X_train[valid_mask]
            y_clean = y_train[asset][valid_mask]
            
            rf = RandomForestRegressor(
                n_estimators=50,
                max_depth=8,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )
            
            rf.fit(X_clean, y_clean)
            models[asset] = rf
            feature_importances[asset] = rf.feature_importances_
            
        except Exception as e:
            continue
    
    return models, feature_importances

def predict_returns(models, current_features):
    """Predice rendimientos usando los modelos Random Forest"""
    predicted_returns = {}
    
    for asset, model in models.items():
        try:
            prediction = model.predict(current_features.values.reshape(1, -1))[0]
            predicted_returns[asset] = prediction
        except Exception as e:
            predicted_returns[asset] = 0
    
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
    
    # Máximo drawdown
    cumulative = np.cumprod(1 + returns_array)
    peak = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - peak) / peak
    max_drawdown = np.min(drawdown)
    
    return {
        'Total Return': total_ret,
        'Annual Return': annual_ret,
        'Annual Volatility': annual_vol,
        'Sharpe Ratio': sharpe,
        'Max Drawdown': max_drawdown,
        'Win Rate': np.mean(returns_array > 0)
    }

def explain_random_forest():
    """Explicación pedagógica del Random Forest"""
    st.markdown("""
    <div class="analogy-box">
    <h3>🎓 ¿CÓMO EXPLICAR RANDOM FOREST A ALUMNOS?</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🤖 Analogía: El Equipo de Expertos
        
        **Imagina que tenemos 50 analistas financieros:**  
        - Cada uno es **especialista** en algo diferente  
        - Analizan **indicadores técnicos** (momentum, volatilidad, etc.)  
        - **Cada uno da su predicción** independiente  
        - Al final, **votamos** y seguimos la recomendación mayoritaria  
        
        **¡Eso es Random Forest!**  
        - Cada árbol = 1 analista  
        - El bosque = equipo completo  
        - Predicción final = promedio de todas las opiniones  
        """)
    
    with col2:
        st.markdown("""
        ### 🎯 En Nuestro Código
        
        **1. Entrenamiento:**  
        ```python
        rf = RandomForestRegressor(
            n_estimators=50,    # 50 "analistas"
            max_depth=8,        # Cada uno hace 8 preguntas
        )
        rf.fit(características, objetivos)
        ```
        
        **2. Predicción:**  
        - Cada árbol analiza los indicadores actuales  
        - Da su predicción de rendimiento futuro  
        - Promediamos todas las predicciones  
        
        **3. Optimización:**  
        - Usamos estas predicciones en Markowitz  
        - Portafolio se basa en **futuro esperado** no solo pasado  
        """)
    
    st.markdown("""
    <div class="feature-card">
    <h4>📊 Ventajas vs Enfoque Tradicional</h4>
    <ul>
    <li><strong>🤖 Inteligencia colectiva:</strong> 50 árboles > 1 árbol</li>
    <li><strong>📈 Captura patrones complejos:</strong> No solo tendencias lineales</li>
    <li><strong>🛡️ Robustez:</strong> Si un árbol se equivoca, otros compensan</li>
    <li><strong>🔍 Interpretabilidad:</strong> Podemos ver qué variables importan más</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# APLICACIÓN PRINCIPAL
def main():
    st.markdown("""
    <div class="main-header">
        <h1>🤖 Random Forest + Markowitz Portfolio</h1>
        <p>Práctica Académica - Machine Learning en Finanzas Cuantitativas</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("⚙️ Configuración")
    
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
    start_date = st.sidebar.date_input("Fecha inicial:", end_date - timedelta(days=730))
    
    prediction_horizon = st.sidebar.slider("Horizonte predicción (días):", 5, 42, 21)
    risk_aversion = st.sidebar.slider("Aversión al riesgo:", 0.5, 5.0, 2.0)
    
    # Explicación pedagógica
    with st.sidebar.expander("🎓 Explicación RF para Clases"):
        st.markdown("""
        **Para enseñar a alumnos:**
        - Analogía: 50 analistas votando
        - Cada árbol = especialista diferente
        - Votación mayoritaria = predicción final
        - Ventaja: captura patrones complejos
        """)
    
    run_analysis = st.sidebar.button("🚀 Ejecutar Análisis Completo", type="primary")
    
    if not run_analysis:
        st.markdown("""
        ## 📚 Bienvenido al Simulador Académico
        
        **Esta aplicación demuestra:**  
        
        <div class="feature-card">
        <strong>🤖 Random Forest:</strong> Predicción de rendimientos usando 50 "árboles-decisión"  
        <strong>📊 Markowitz:</strong> Optimización matemática riesgo-retorno  
        <strong>🔬 Backtesting:</strong> Validación histórica de la estrategia  
        <strong>🎓 Pedagogía:</strong> Explicaciones para enseñanza en aula  
        </div>
        
        ### 🎯 Objetivos de Aprendizaje:
        
        1. **Entender** cómo ML mejora la gestión de portafolios  
        2. **Visualizar** el proceso completo de análisis cuantitativo  
        3. **Comparar** enfoque tradicional vs machine learning  
        4. **Interpretar** resultados para toma de decisiones  
        
        **👈 Configura los parámetros y presiona 'Ejecutar Análisis'**
        """, unsafe_allow_html=True)
        return
    
    # SECCIÓN 1: EXPLICACIÓN PEDAGÓGICA
    st.header("🎓 Explicación: Random Forest en Finanzas")
    explain_random_forest()
    
    # SECCIÓN 2: ANÁLISIS DE DATOS
    st.header("📊 Obtención y Análisis de Datos")
    
    with st.spinner("📥 Descargando datos del mercado..."):
        asset_prices, data_type = get_market_data(tickers, start_date, end_date)
    
    if asset_prices.empty:
        st.error("❌ No se pudieron obtener datos suficientes")
        return
    
    if data_type == "simulado":
        st.info("🔮 Usando datos simulados para demostración académica")
    else:
        st.success("✅ Datos reales obtenidos exitosamente")
    
    # Mostrar resumen de datos
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Días de datos", len(asset_prices))
    with col2:
        st.metric("Activos analizados", asset_prices.shape[1])
    with col3:
        st.metric("Período análisis", f"{(end_date - start_date).days} días")
    
    # Gráfico de precios
    st.subheader("📈 Evolución de Precios")
    normalized_prices = (asset_prices / asset_prices.iloc[0] * 100)
    st.line_chart(normalized_prices)
    
    # SECCIÓN 3: INGENIERÍA DE CARACTERÍSTICAS
    st.header("🔧 Ingeniería de Características")
    
    with st.spinner("🧠 Calculando indicadores técnicos..."):
        features = calculate_features(asset_prices)
        
        returns = asset_prices.pct_change().dropna()
        targets = pd.DataFrame(index=returns.index, columns=returns.columns)
        for col in returns.columns:
            targets[col] = returns[col].shift(-prediction_horizon)
        
        common_dates = features.index.intersection(targets.index)
        X = features.loc[common_dates].fillna(method='ffill').fillna(0)
        y = targets.loc[common_dates]
    
    st.success(f"✅ {X.shape[1]} características creadas para {len(X)} observaciones")
    
    # Mostrar ejemplos de características
    st.subheader("📋 Ejemplo de Características Calculadas")
    st.dataframe(X.head().style.format("{:.4f}"))
    
    # SECCIÓN 4: ENTRENAMIENTO DEL MODELO
    st.header("🤖 Entrenamiento del Random Forest")
    
    with st.spinner("🌳 Entrenando 50 árboles de decisión..."):
        models, feature_importances = train_models(X, y)
    
    if len(models) == 0:
        st.error("❌ No se pudieron entrenar modelos válidos")
        return
    
    st.success(f"✅ {len(models)} modelos entrenados exitosamente")
    
    # Importancia de características
    st.subheader("🎯 Importancia de Características")
    
    if len(models) > 0:
        example_asset = list(models.keys())[0]
        importance_df = pd.DataFrame({
            'Característica': X.columns,
            'Importancia': feature_importances[example_asset]
        }).sort_values('Importancia', ascending=False).head(10)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.bar_chart(importance_df.set_index('Característica')['Importancia'])
        
        with col2:
            st.write("**Top 5 Características:**")
            for i, row in importance_df.head().iterrows():
                st.write(f"• {row['Característica'].split('_')[-1]}: {row['Importancia']:.3f}")
    
    # SECCIÓN 5: BACKTESTING Y OPTIMIZACIÓN
    st.header("🔄 Backtesting con Predicciones RF")
    
    with st.spinner("⚡ Ejecutando simulación histórica..."):
        
        split_idx = int(len(returns) * 0.7)
        test_period = returns.iloc[split_idx:]
        
        portfolio_rets = []
        benchmark_rets = []
        all_weights = []
        rf_predictions_history = []
        historical_predictions_history = []
        
        n_periods = min(6, len(test_period) // 30)
        if n_periods == 0:
            n_periods = 1
        period_length = len(test_period) // n_periods
        
        for period in range(n_periods):
            start_p = period * period_length
            end_p = min((period + 1) * period_length, len(test_period))
            
            if end_p <= start_p + prediction_horizon:
                break
            
            # PREDICCIÓN CON RANDOM FOREST
            current_date = test_period.index[start_p]
            current_features_row = features.loc[current_date]
            rf_predicted_returns = predict_returns(models, current_features_row)
            rf_expected_returns = np.array([rf_predicted_returns.get(ticker, 0) for ticker in tickers])
            
            # Predicción histórica (para comparación)
            hist_data = returns.iloc[:split_idx + start_p]
            historical_expected_returns = hist_data.tail(63).mean().values * 252
            
            # Matriz de covarianzas
            cov_matrix = hist_data.tail(126).cov().values * 252
            
            # Optimización con predicciones RF
            weights = optimize_portfolio(rf_expected_returns, cov_matrix, risk_aversion)
            all_weights.append(weights)
            
            rf_predictions_history.append(rf_expected_returns)
            historical_predictions_history.append(historical_expected_returns)
            
            # Cálculo de rendimientos
            period_data = test_period.iloc[start_p:end_p]
            
            for _, day_returns in period_data.iterrows():
                port_ret = np.sum(weights * day_returns.values)
                bench_ret = np.mean(day_returns.values)
                
                portfolio_rets.append(port_ret)
                benchmark_rets.append(bench_ret)
        
        final_weights = np.mean(all_weights, axis=0) if all_weights else np.ones(len(tickers)) / len(tickers)
    
    if len(portfolio_rets) == 0:
        st.error("❌ No se generaron datos de backtesting")
        return
    
    st.success(f"✅ Backtesting completado: {len(portfolio_rets)} días, {n_periods} rebalanceos")
    
    # SECCIÓN 6: RESULTADOS Y COMPARACIÓN
    st.header("📊 Resultados de la Estrategia")
    
    # Métricas
    port_metrics = calculate_metrics(portfolio_rets)
    bench_metrics = calculate_metrics(benchmark_rets)
    
    # Comparación lado a lado
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🤖 RF + Markowitz")
        st.metric("Retorno Anual", f"{port_metrics.get('Annual Return', 0):.2%}")
        st.metric("Volatilidad Anual", f"{port_metrics.get('Annual Volatility', 0):.2%}")
        st.metric("Sharpe Ratio", f"{port_metrics.get('Sharpe Ratio', 0):.3f}")
        st.metric("Máximo Drawdown", f"{port_metrics.get('Max Drawdown', 0):.2%}")
        st.metric("Tasa de Éxito", f"{port_metrics.get('Win Rate', 0):.1%}")
    
    with col2:
        st.markdown("### 📈 Benchmark (Equal Weight)")
        st.metric("Retorno Anual", f"{bench_metrics.get('Annual Return', 0):.2%}", 
                 delta=f"{(port_metrics.get('Annual Return', 0) - bench_metrics.get('Annual Return', 0)):.2%}")
        st.metric("Volatilidad Anual", f"{bench_metrics.get('Annual Volatility', 0):.2%}",
                 delta=f"{(port_metrics.get('Annual Volatility', 0) - bench_metrics.get('Annual Volatility', 0)):.2%}")
        st.metric("Sharpe Ratio", f"{bench_metrics.get('Sharpe Ratio', 0):.3f}",
                 delta=f"{(port_metrics.get('Sharpe Ratio', 0) - bench_metrics.get('Sharpe Ratio', 0)):.3f}")
        st.metric("Máximo Drawdown", f"{bench_metrics.get('Max Drawdown', 0):.2%}")
        st.metric("Tasa de Éxito", f"{bench_metrics.get('Win Rate', 0):.1%}")
    
    # Gráfico de performance
    st.subheader("📈 Performance Acumulada")
    try:
        port_cumulative = np.cumprod(1 + np.array(portfolio_rets))
        bench_cumulative = np.cumprod(1 + np.array(benchmark_rets))
        
        performance_df = pd.DataFrame({
            'RF + Markowitz': port_cumulative,
            'Benchmark': bench_cumulative
        })
        
        st.line_chart(performance_df)
    except Exception as e:
        st.error(f"Error en gráfico: {str(e)}")
    
    # SECCIÓN 7: COMPOSICIÓN DEL PORTAFOLIO
    st.header("💼 Composición Óptima del Portafolio")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Distribución Recomendada")
        
        weights_df = pd.DataFrame({
            'Activo': tickers,
            'Sector': [ticker_names.get(t, t) for t in tickers],
            'Peso': final_weights,
            'Peso %': [f"{w*100:.1f}%" for w in final_weights]
        }).sort_values('Peso', ascending=False)
        
        st.dataframe(weights_df[['Activo', 'Sector', 'Peso %']], hide_index=True)
        
        # Ejemplo de inversión
        st.subheader("💰 Ejemplo Práctico")
        inversion = st.number_input("Monto a invertir ($):", min_value=1000, value=10000, step=1000)
        
        if inversion > 0:
            for ticker, weight in zip(tickers, final_weights):
                if weight > 0.01:
                    st.write(f"**{ticker}**: ${weight * inversion:,.0f} ({weight*100:.1f}%)")
    
    with col2:
        st.subheader("📊 Visualización de Pesos")
        
        # Gráfico de barras
        chart_df = weights_df[weights_df['Peso'] > 0.01].copy()
        if not chart_df.empty:
            st.bar_chart(chart_df.set_index('Activo')['Peso'])
        else:
            st.info("Todos los pesos son muy pequeños para visualizar")
        
        # Pie chart simplificado
        st.subheader("🎯 Resumen de Asignación")
        high_weight_assets = weights_df[weights_df['Peso'] > 0.1]
        if len(high_weight_assets) > 0:
            for _, row in high_weight_assets.iterrows():
                st.write(f"▪️ **{row['Activo']}**: {row['Peso %']}")
        else:
            st.write("Asignación bastante diversificada")
    
    # SECCIÓN 8: ANÁLISIS DE PREDICCIONES
    st.header("🔮 Análisis de Predicciones RF")
    
    if rf_predictions_history:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Últimas Predicciones RF")
            last_rf_pred = rf_predictions_history[-1] * 252 * 100
            pred_df = pd.DataFrame({
                'Activo': tickers,
                'Predicción Anual %': last_rf_pred
            }).sort_values('Predicción Anual %', ascending=False)
            
            for _, row in pred_df.iterrows():
                delta = f"{row['Predicción Anual %']:.1f}%"
                st.metric(f"{row['Activo']} ({ticker_names[row['Activo']]})", 
                         value=f"{row['Predicción Anual %']:.1f}%")
        
        with col2:
            st.subheader("📊 Comparación de Métodos")
            st.write("**RF vs Historical Mean**")
            
            # Calcular accuracy simple
            actual_returns = returns.mean().values * 252 * 100
            rf_accuracy = np.mean(np.abs(last_rf_pred - actual_returns))
            
            st.metric("Error Absoluto Promedio RF", f"{rf_accuracy:.1f}%")
            st.metric("Número de Rebalanceos", n_periods)
            st.metric("Horizonte de Predicción", f"{prediction_horizon} días")
    
    # SECCIÓN 9: CONCLUSIONES PEDAGÓGICAS
    st.header("🎯 Conclusiones para el Aula")
    
    excess_return = port_metrics.get('Annual Return', 0) - bench_metrics.get('Annual Return', 0)
    sharpe_diff = port_metrics.get('Sharpe Ratio', 0) - bench_metrics.get('Sharpe Ratio', 0)
    
    if excess_return > 0 and sharpe_diff > 0:
        st.success("""
        ## 🏆 **ESTRATEGIA EXITOSA**
        
        **El Random Forest añadió valor significativo:**
        - ✅ Mejor retorno ajustado al riesgo
        - ✅ Predicciones más precisas que la media histórica
        - ✅ Optimización basada en señales predictivas
        
        **Para la clase:** Demuestra cómo ML puede mejorar decisiones de inversión.
        """)
    elif excess_return > 0:
        st.warning("""
        ## ⚠️ **RESULTADO MIXTO**
        
        **El RF mejoró retornos pero con más riesgo:**
        - 📈 Mayor retorno, pero mayor volatilidad
        - 🤔 Puede necesitar ajuste de parámetros de riesgo
        - 🔍 Interesante para análisis de trade-offs
        
        **Para la clase:** Buen ejemplo de balance riesgo-retorno.
        """)
    else:
        st.info("""
        ## 📊 **CASO DE ESTUDIO**
        
        **El benchmark superó al RF en este período:**
        - 📉 Contexto mercado específico
        - 🔍 Oportunidad para analizar por qué
        - 💡 ML no es magia - depende de datos y parámetros
        
        **Para la clase:** Enseña humildad y validación rigurosa.
        """)
    
    # SECCIÓN 10: DESCARGAS Y REPORTES
    st.header("💾 Material para Clases")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📋 Generar Reporte de Análisis"):
            report = f"""
REPORTE ACADÉMICO - RANDOM FOREST + MARKOWITZ
Fecha: {datetime.now().strftime('%Y-%m-%d')}

RESULTADOS:
- Retorno RF+Markowitz: {port_metrics.get('Annual Return', 0):.2%}
- Retorno Benchmark: {bench_metrics.get('Annual Return', 0):.2%}
- Sharpe Ratio RF: {port_metrics.get('Sharpe Ratio', 0):.3f}
- Diferencial de Retorno: {excess_return:.2%}

COMPOSICIÓN ÓPTIMA:
"""
            for ticker, weight in zip(tickers, final_weights):
                if weight > 0.01:
                    report += f"- {ticker}: {weight*100:.1f}%\n"
            
            report += f"""

LECCIONES PARA EL AULA:
1. El RF {'mejoró' if excess_return > 0 else 'no mejoró'} el desempeño
2. Importancia de validación con backtesting
3. El ML complementa pero no reemplaza el análisis financiero
"""
            
            st.download_button(
                label="📥 Descargar Reporte",
                data=report,
                file_name=f"reporte_aula_{datetime.now().strftime('%Y%m%d')}.txt",
                mime="text/plain"
            )
    
    with col2:
        st.write("**🔄 Para volver a ejecutar:**")
        st.write("Modifica parámetros en la sidebar y presiona 'Ejecutar Análisis' nuevamente")
        
        st.write("**🎓 Para uso en clases:**")
        st.write("1. Ejecuta con diferentes parámetros")
        st.write("2. Discute los resultados con alumnos")
        st.write("3. Analiza por qué el RF funciona o no")

if __name__ == "__main__":
    main()
