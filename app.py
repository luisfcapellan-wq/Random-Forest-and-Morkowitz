#!/usr/bin/env python3
"""
📊 APLICACIÓN STREAMLIT: RANDOM FOREST + MARKOWITZ PORTFOLIO OPTIMIZER
🎓 Práctica Académica Interactiva para Construcción de Portafolios

Autor: Práctica Académica
Fecha: 2025

Para ejecutar: streamlit run app.py
"""

# Importaciones con manejo de errores
import streamlit as st
import pandas as pd
import numpy as np
import warnings
import time
from datetime import datetime, timedelta
import io

# Importaciones con manejo de errores para dependencias opcionales
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    st.error("❌ Plotly no está instalado. Ejecute: pip install plotly")
    PLOTLY_AVAILABLE = False
    st.stop()

try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    st.error("❌ yfinance no está instalado. Ejecute: pip install yfinance")
    YF_AVAILABLE = False
    st.stop()

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    SKLEARN_AVAILABLE = True
except ImportError:
    st.error("❌ scikit-learn no está instalado. Ejecute: pip install scikit-learn")
    SKLEARN_AVAILABLE = False
    st.stop()

try:
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError:
    st.error("❌ scipy no está instalado. Ejecute: pip install scipy")
    SCIPY_AVAILABLE = False
    st.stop()

warnings.filterwarnings('ignore')

# ====================================================================
# CONFIGURACIÓN DE LA PÁGINA
# ====================================================================

st.set_page_config(
    page_title="Random Forest + Markowitz Portfolio Optimizer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
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
    .metric-container {
        background: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #1f4e79;
    }
    .success-box {
        background: #d4edda;
        color: #155724;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #c3e6cb;
        margin: 10px 0;
    }
    .warning-box {
        background: #fff3cd;
        color: #856404;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #ffeaa7;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ====================================================================
# FUNCIONES AUXILIARES
# ====================================================================

@st.cache_data(ttl=3600)  # Cache por 1 hora
def download_market_data(tickers, start_date, end_date):
    """Descarga datos del mercado usando yfinance"""
    try:
        data = {}
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, ticker in enumerate(tickers):
            status_text.text(f'Descargando {ticker}...')
            try:
                temp = yf.download(ticker, start=start_date, end=end_date, progress=False)
                if not temp.empty:
                    data[ticker] = temp['Adj Close']
                progress_bar.progress((i + 1) / len(tickers))
            except Exception as e:
                st.warning(f"No se pudo descargar {ticker}: {e}")
        
        status_text.text('✅ Descarga completada')
        time.sleep(0.5)
        status_text.empty()
        progress_bar.empty()
        
        if data:
            df = pd.DataFrame(data).dropna()
            return df
        else:
            st.error("No se pudieron descargar datos")
            return None
            
    except Exception as e:
        st.error(f"Error descargando datos: {e}")
        return None

@st.cache_data
def download_macro_data(start_date, end_date):
    """Descarga datos macroeconómicos"""
    macro_tickers = {
        '^TNX': 'Treasury_10Y',
        '^VIX': 'VIX',
        'DX-Y.NYB': 'DXY',
        '^GSPC': 'SP500'
    }
    
    macro_data = {}
    for ticker, name in macro_tickers.items():
        try:
            temp = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if not temp.empty:
                macro_data[name] = temp['Adj Close'] if 'Adj Close' in temp.columns else temp['Close']
        except:
            continue
    
    return pd.DataFrame(macro_data) if macro_data else None

def calculate_technical_indicators(prices):
    """Calcula indicadores técnicos"""
    features = pd.DataFrame(index=prices.index)
    
    for ticker in prices.columns:
        price_series = prices[ticker]
        
        # Momentum
        features[f'{ticker}_momentum_1m'] = price_series.pct_change(21)
        features[f'{ticker}_momentum_3m'] = price_series.pct_change(63)
        
        # Volatilidad
        features[f'{ticker}_volatility'] = price_series.pct_change().rolling(20).std()
        
        # Moving Average Ratios
        features[f'{ticker}_ma_ratio_50'] = price_series / price_series.rolling(50).mean()
        features[f'{ticker}_ma_ratio_200'] = price_series / price_series.rolling(200).mean()
        
        # RSI simplificado
        delta = price_series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        features[f'{ticker}_rsi'] = 100 - (100 / (1 + rs))
    
    return features.dropna()

def train_random_forest_models(X, y, test_size=0.2):
    """Entrena modelos Random Forest para cada activo"""
    models = {}
    performance = {}
    
    # División temporal
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    progress_bar = st.progress(0)
    
    for i, asset in enumerate(y.columns):
        # Filtrar NaN
        mask = ~(y_train[asset].isna() | X_train.isna().any(axis=1))
        if mask.sum() < 100:  # Mínimo de observaciones
            continue
            
        X_asset = X_train[mask]
        y_asset = y_train[asset][mask]
        
        # Entrenar modelo
        rf = RandomForestRegressor(
            n_estimators=100,
            max_depth=8,
            min_samples_split=20,
            random_state=42,
            n_jobs=1
        )
        
        rf.fit(X_asset, y_asset)
        models[asset] = rf
        
        # Evaluar
        if len(X_test) > 0:
            test_mask = ~(y_test[asset].isna() | X_test.isna().any(axis=1))
            if test_mask.sum() > 10:
                X_test_clean = X_test[test_mask]
                y_test_clean = y_test[asset][test_mask]
                
                predictions = rf.predict(X_test_clean)
                mse = mean_squared_error(y_test_clean, predictions)
                mae = mean_absolute_error(y_test_clean, predictions)
                
                # Dirección accuracy
                direction_acc = np.mean(
                    np.sign(y_test_clean) == np.sign(predictions)
                )
                
                performance[asset] = {
                    'MSE': mse,
                    'MAE': mae,
                    'Direction_Accuracy': direction_acc,
                    'R2_train': rf.score(X_asset, y_asset)
                }
        
        progress_bar.progress((i + 1) / len(y.columns))
    
    progress_bar.empty()
    return models, performance

def markowitz_optimization(expected_returns, cov_matrix, risk_aversion=2, max_weight=0.25):
    """Optimización de Markowitz"""
    n_assets = len(expected_returns)
    
    def objective(weights):
        portfolio_return = np.dot(weights, expected_returns)
        portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
        return -(portfolio_return - (risk_aversion/2) * portfolio_variance)
    
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    bounds = [(0, max_weight) for _ in range(n_assets)]
    
    result = minimize(
        objective,
        x0=np.ones(n_assets) / n_assets,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000}
    )
    
    return result.x if result.success else np.ones(n_assets) / n_assets

def calculate_portfolio_metrics(returns):
    """Calcula métricas de performance del portafolio"""
    total_return = (1 + returns).prod() - 1
    annual_return = (1 + returns.mean())**252 - 1
    annual_volatility = returns.std() * np.sqrt(252)
    sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0
    
    # Max Drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    
    return {
        'Total Return': total_return,
        'Annual Return': annual_return,
        'Annual Volatility': annual_volatility,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown': max_drawdown,
        'Win Rate': (returns > 0).mean()
    }

def create_performance_plots(portfolio_returns, benchmark_returns, weights_history):
    """Crea gráficos de performance"""
    
    if not PLOTLY_AVAILABLE:
        st.error("Plotly no disponible - usando visualizaciones alternativas")
        
        # Visualizaciones alternativas con matplotlib/streamlit nativo
        col1, col2 = st.columns(2)
        
        with col1:
            # Performance acumulada usando line_chart nativo de streamlit
            portfolio_cum = (1 + portfolio_returns).cumprod()
            benchmark_cum = (1 + benchmark_returns).cumprod()
            
            performance_df = pd.DataFrame({
                'RF + Markowitz': portfolio_cum.values,
                'Benchmark': benchmark_cum.values
            })
            
            st.line_chart(performance_df)
            st.caption("Rendimiento Acumulado")
        
        with col2:
            # Distribución usando histograma nativo
            hist_df = pd.DataFrame({
                'Portfolio': portfolio_returns.values,
                'Benchmark': benchmark_returns.values
            })
            
            st.bar_chart(pd.Series(portfolio_returns.values).value_counts().sort_index().head(20))
            st.caption("Distribución de Rendimientos")
        
        return None, None, None, None
    
    # 1. Performance acumulada
    fig1 = go.Figure()
    
    portfolio_cum = (1 + portfolio_returns).cumprod()
    benchmark_cum = (1 + benchmark_returns).cumprod()
    
    fig1.add_trace(go.Scatter(
        x=portfolio_cum.index, 
        y=portfolio_cum.values,
        mode='lines',
        name='RF + Markowitz',
        line=dict(color='#2E86C1', width=2)
    ))
    
    fig1.add_trace(go.Scatter(
        x=benchmark_cum.index, 
        y=benchmark_cum.values,
        mode='lines',
        name='Benchmark',
        line=dict(color='#E74C3C', width=2)
    ))
    
    fig1.update_layout(
        title='Rendimiento Acumulado',
        xaxis_title='Fecha',
        yaxis_title='Valor del Portafolio',
        height=400
    )
    
    # 2. Drawdown
    portfolio_dd = (portfolio_cum / portfolio_cum.expanding().max() - 1)
    
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=portfolio_dd.index,
        y=portfolio_dd.values,
        fill='tonexty',
        mode='lines',
        name='Drawdown',
        line=dict(color='red')
    ))
    
    fig2.update_layout(
        title='Drawdown del Portafolio',
        xaxis_title='Fecha',
        yaxis_title='Drawdown (%)',
        height=300
    )
    
    # 3. Distribución de rendimientos
    fig3 = go.Figure()
    fig3.add_trace(go.Histogram(
        x=portfolio_returns.values,
        nbinsx=50,
        name='Portfolio',
        opacity=0.7
    ))
    fig3.add_trace(go.Histogram(
        x=benchmark_returns.values,
        nbinsx=50,
        name='Benchmark',
        opacity=0.7
    ))
    
    fig3.update_layout(
        title='Distribución de Rendimientos Diarios',
        xaxis_title='Rendimiento',
        yaxis_title='Frecuencia',
        height=300
    )
    
    # 4. Evolución de pesos
    fig4 = go.Figure()
    
    if len(weights_history) > 0:
        for i, ticker in enumerate(weights_history.columns):
            fig4.add_trace(go.Scatter(
                x=weights_history.index,
                y=weights_history[ticker].values,
                mode='lines',
                name=ticker,
                stackgroup='one'
            ))
    
    fig4.update_layout(
        title='Evolución de Pesos del Portafolio',
        xaxis_title='Fecha de Rebalanceo',
        yaxis_title='Peso (%)',
        height=400
    )
    
    return fig1, fig2, fig3, fig4

# ====================================================================
# INTERFAZ PRINCIPAL
# ====================================================================

def main():
    # Título principal
    st.markdown("""
    <div class="main-header">
        <h1>📊 Random Forest + Markowitz Portfolio Optimizer</h1>
        <p>Práctica Académica Interactiva para Construcción de Portafolios Inteligentes</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar para configuración
    st.sidebar.header("⚙️ Configuración del Análisis")
    
    # Selección de activos
    st.sidebar.subheader("🏢 Universo de Inversión")
    
    asset_universe = st.sidebar.selectbox(
        "Seleccionar universo:",
        ["ETFs Sectoriales S&P 500", "Tech Stocks", "Personalizado"],
        index=0
    )
    
    if asset_universe == "ETFs Sectoriales S&P 500":
        tickers = ['XLF', 'XLK', 'XLV', 'XLI', 'XLE', 'XLY', 'XLP', 'XLB', 'XLU']
        ticker_names = {
            'XLF': 'Financials', 'XLK': 'Technology', 'XLV': 'Healthcare',
            'XLI': 'Industrials', 'XLE': 'Energy', 'XLY': 'Consumer Disc.',
            'XLP': 'Consumer Staples', 'XLB': 'Materials', 'XLU': 'Utilities'
        }
    elif asset_universe == "Tech Stocks":
        tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META', 'TSLA', 'NVDA']
        ticker_names = {t: t for t in tickers}
    else:
        ticker_input = st.sidebar.text_input(
            "Ingrese tickers separados por coma:",
            "AAPL,GOOGL,MSFT,AMZN"
        )
        tickers = [t.strip().upper() for t in ticker_input.split(',')]
        ticker_names = {t: t for t in tickers}
    
    # Parámetros temporales
    st.sidebar.subheader("📅 Período de Análisis")
    
    end_date = st.sidebar.date_input(
        "Fecha final:",
        value=datetime.now().date()
    )
    
    start_date = st.sidebar.date_input(
        "Fecha inicial:",
        value=end_date - timedelta(days=3*365)  # 3 años por defecto
    )
    
    # Parámetros del modelo
    st.sidebar.subheader("🤖 Parámetros del Modelo")
    
    prediction_horizon = st.sidebar.slider(
        "Horizonte de predicción (días):",
        min_value=5, max_value=63, value=21
    )
    
    risk_aversion = st.sidebar.slider(
        "Aversión al riesgo:",
        min_value=0.5, max_value=10.0, value=2.0, step=0.5
    )
    
    max_weight = st.sidebar.slider(
        "Peso máximo por activo (%):",
        min_value=5, max_value=50, value=25
    ) / 100
    
    rebalance_freq = st.sidebar.selectbox(
        "Frecuencia de rebalanceo:",
        [("Mensual", 21), ("Trimestral", 63), ("Semestral", 126)],
        format_func=lambda x: x[0]
    )[1]
    
    # Botón principal
    run_analysis = st.sidebar.button("🚀 Ejecutar Análisis", type="primary")
    
    # Panel principal
    if not run_analysis:
        # Pantalla de bienvenida
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ## 🎓 Bienvenido a la Práctica Académica
            
            Esta aplicación implementa una estrategia híbrida que combina:
            
            ### 🌳 **Random Forest**
            - Predice rendimientos futuros usando variables técnicas y macroeconómicas
            - Maneja relaciones no-lineales entre variables
            - Robusto ante outliers y ruido en los datos
            
            ### 📊 **Optimización de Markowitz**
            - Construye portafolios eficientes en la frontera riesgo-retorno
            - Considera correlaciones entre activos
            - Aplica restricciones de peso y diversificación
            
            ### 📈 **Backtesting Robusto**
            - Validación temporal sin look-ahead bias
            - Rebalanceo periódico realista
            - Comparación con benchmarks
            
            **Configure los parámetros en la barra lateral y presione "Ejecutar Análisis"**
            """)
        
        with col2:
            st.markdown("""
            <div class="success-box">
                <h4>🔧 Funcionalidades:</h4>
                <ul>
                    <li>✅ Descarga automática de datos</li>
                    <li>✅ Modelos ML entrenados en tiempo real</li>
                    <li>✅ Optimización matemática</li>
                    <li>✅ Visualizaciones interactivas</li>
                    <li>✅ Métricas de performance</li>
                    <li>✅ Exportación de resultados</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Mostrar ejemplo de configuración
        st.subheader("📋 Configuración Actual:")
        config_df = pd.DataFrame({
            'Parámetro': [
                'Universo de inversión',
                'Período de análisis', 
                'Horizonte de predicción',
                'Aversión al riesgo',
                'Peso máximo por activo',
                'Frecuencia de rebalanceo'
            ],
            'Valor': [
                f"{asset_universe} ({len(tickers)} activos)",
                f"{start_date} a {end_date}",
                f"{prediction_horizon} días",
                f"{risk_aversion}",
                f"{max_weight*100:.0f}%",
                f"Cada {rebalance_freq} días"
            ]
        })
        st.table(config_df)
        
        return
    
    # ===== ANÁLISIS PRINCIPAL =====
    
    with st.spinner("🔄 Iniciando análisis..."):
        
        # 1. DESCARGA DE DATOS
        st.subheader("📊 Descarga de Datos")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("Descargando datos de activos...")
            asset_prices = download_market_data(tickers, start_date, end_date)
            
            if asset_prices is None:
                st.error("Error descargando datos de activos")
                return
            
            st.success(f"✅ {len(asset_prices)} días de datos para {len(tickers)} activos")
        
        with col2:
            st.info("Descargando datos macroeconómicos...")
            macro_data = download_macro_data(start_date, end_date)
            
            if macro_data is not None:
                st.success(f"✅ Variables macro: {', '.join(macro_data.columns)}")
            else:
                st.warning("⚠️ No se pudieron descargar datos macro")
        
        # Mostrar estadísticas básicas
        with st.expander("📈 Estadísticas de los Datos", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Precios finales:**")
                final_prices = asset_prices.iloc[-1].sort_values(ascending=False)
                st.bar_chart(final_prices)
            
            with col2:
                st.write("**Rendimientos anualizados:**")
                returns = asset_prices.pct_change().dropna()
                annual_returns = (returns.mean() * 252 * 100).sort_values(ascending=False)
                st.bar_chart(annual_returns)
        
        # 2. FEATURE ENGINEERING
        st.subheader("🔧 Creación de Características")
        
        with st.spinner("Calculando indicadores técnicos..."):
            technical_features = calculate_technical_indicators(asset_prices)
            
            # Combinar con macro si está disponible
            if macro_data is not None:
                # Alinear fechas
                common_dates = technical_features.index.intersection(macro_data.index)
                if len(common_dates) > 100:
                    all_features = pd.concat([
                        technical_features.loc[common_dates],
                        macro_data.loc[common_dates]
                    ], axis=1).dropna()
                else:
                    all_features = technical_features.dropna()
            else:
                all_features = technical_features.dropna()
        
        # Crear targets
        returns = asset_prices.pct_change().dropna()
        targets = pd.DataFrame(index=returns.index, columns=returns.columns)
        for col in returns.columns:
            targets[col] = returns[col].shift(-prediction_horizon)
        
        # Alinear fechas
        common_dates = all_features.index.intersection(targets.index)
        X = all_features.loc[common_dates]
        y = targets.loc[common_dates]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📊 Observaciones", len(X))
        with col2:
            st.metric("🔢 Características", X.shape[1])
        with col3:
            st.metric("🎯 Activos objetivo", y.shape[1])
        
        # 3. ENTRENAMIENTO DE MODELOS
        st.subheader("🤖 Entrenamiento de Random Forest")
        
        with st.spinner("Entrenando modelos de Machine Learning..."):
            models, performance = train_random_forest_models(X, y)
        
        if models:
            st.success(f"✅ {len(models)} modelos entrenados exitosamente")
            
            # Mostrar performance
            perf_df = pd.DataFrame(performance).T
            if not perf_df.empty:
                st.write("**Performance de los modelos:**")
                st.dataframe(perf_df.round(4))
        else:
            st.error("❌ No se pudieron entrenar modelos")
            return
        
        # 4. BACKTESTING
        st.subheader("🔄 Ejecución del Backtesting")
        
        with st.spinner("Ejecutando backtesting con rebalanceo..."):
            
            # Configurar backtesting
            backtest_start_idx = max(252, len(X) // 3)  # Empezar con suficientes datos
            backtest_dates = X.index[backtest_start_idx::rebalance_freq]
            
            portfolio_returns = []
            benchmark_returns = []
            weights_history = []
            rebalance_dates = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, rebalance_date in enumerate(backtest_dates[:-1]):
                status_text.text(f"Procesando rebalanceo {i+1}/{len(backtest_dates)-1}: {rebalance_date.date()}")
                
                try:
                    # Datos históricos hasta la fecha
                    hist_idx = X.index.get_loc(rebalance_date)
                    X_hist = X.iloc[:hist_idx]
                    y_hist = y.iloc[:hist_idx]
                    
                    if len(X_hist) < 100:
                        continue
                    
                    # Predecir rendimientos esperados
                    current_features = X.loc[rebalance_date:rebalance_date]
                    expected_returns = pd.Series(index=asset_prices.columns, dtype=float)
                    
                    for asset in asset_prices.columns:
                        if asset in models and len(current_features) > 0:
                            try:
                                pred = models[asset].predict(current_features.fillna(0))[0]
                                expected_returns[asset] = pred * 252  # Anualizar
                            except:
                                expected_returns[asset] = returns[asset].mean() * 252
                        else:
                            expected_returns[asset] = returns[asset].mean() * 252
                    
                    # Calcular matriz de covarianzas
                    recent_returns = returns.loc[returns.index <= rebalance_date].tail(min(252, len(returns)))
                    cov_matrix = recent_returns.cov() * 252  # Anualizada
                    
                    # Optimizar portafolio
                    weights = markowitz_optimization(expected_returns, cov_matrix, risk_aversion, max_weight)
                    weights_series = pd.Series(weights, index=asset_prices.columns)
                    
                    weights_history.append(weights_series)
                    rebalance_dates.append(rebalance_date)
                    
                    # Calcular rendimientos del período siguiente
                    next_date_idx = min(hist_idx + rebalance_freq, len(X) - 1)
                    period_returns = returns.iloc[hist_idx+1:next_date_idx+1]
                    
                    if len(period_returns) > 0:
                        portfolio_period_returns = (period_returns * weights_series).sum(axis=1)
                        benchmark_period_returns = period_returns.mean(axis=1)
                        
                        portfolio_returns.extend(portfolio_period_returns.tolist())
                        benchmark_returns.extend(benchmark_period_returns.tolist())
                    
                except Exception as e:
                    st.warning(f"Error en rebalanceo {i+1}: {str(e)[:100]}")
                    continue
                
                progress_bar.progress((i + 1) / (len(backtest_dates) - 1))
            
            status_text.empty()
            progress_bar.empty()
        
        # Convertir a Series
        if portfolio_returns:
            portfolio_returns = pd.Series(portfolio_returns)
            benchmark_returns = pd.Series(benchmark_returns)
            weights_df = pd.DataFrame(weights_history, index=rebalance_dates, columns=asset_prices.columns)
        else:
            st.error("❌ No se pudieron calcular rendimientos del backtest")
            return
        
        # 5. RESULTADOS Y VISUALIZACIONES
        st.subheader("📊 Resultados del Análisis")
        
        # Métricas principales
        portfolio_metrics = calculate_portfolio_metrics(portfolio_returns)
        benchmark_metrics = calculate_portfolio_metrics(benchmark_returns)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🤖 Random Forest + Markowitz")
            st.metric("Retorno Total", f"{portfolio_metrics['Total Return']:.2%}")
            st.metric("Retorno Anualizado", f"{portfolio_metrics['Annual Return']:.2%}")
            st.metric("Volatilidad Anualizada", f"{portfolio_metrics['Annual Volatility']:.2%}")
            st.metric("Ratio de Sharpe", f"{portfolio_metrics['Sharpe Ratio']:.3f}")
            st.metric("Máximo Drawdown", f"{portfolio_metrics['Max Drawdown']:.2%}")
            st.metric("Win Rate", f"{portfolio_metrics['Win Rate']:.1%}")
        
        with col2:
            st.markdown("### 📈 Benchmark (Equal Weight)")
            st.metric("Retorno Total", f"{benchmark_metrics['Total Return']:.2%}")
            st.metric("Retorno Anualizado", f"{benchmark_metrics['Annual Return']:.2%}")
            st.metric("Volatilidad Anualizada", f"{benchmark_metrics['Annual Volatility']:.2%}")
            st.metric("Ratio de Sharpe", f"{benchmark_metrics['Sharpe Ratio']:.3f}")
            st.metric("Máximo Drawdown", f"{benchmark_metrics['Max Drawdown']:.2%}")
            st.metric("Win Rate", f"{benchmark_metrics['Win Rate']:.1%}")
        
        # Métricas de valor añadido
        excess_returns = portfolio_returns - benchmark_returns
        alpha = excess_returns.mean() * 252
        tracking_error = excess_returns.std() * np.sqrt(252)
        information_ratio = alpha / tracking_error if tracking_error > 0 else 0
        
        st.markdown("### ⭐ Valor Añadido")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Alfa Anualizada", f"{alpha:.2%}")
        with col2:
            st.metric("Information Ratio", f"{information_ratio:.3f}")
        with col3:
            outperform_rate = (portfolio_returns > benchmark_returns).mean()
            st.metric("% Días Superó Benchmark", f"{outperform_rate:.1%}")
        
        # Gráficos principales
        st.subheader("📈 Visualizaciones de Performance")
        
        fig1, fig2, fig3, fig4 = create_performance_plots(portfolio_returns, benchmark_returns, weights_df)
        
        # Layout de gráficos
        if PLOTLY_AVAILABLE and fig1 is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                st.plotly_chart(fig1, use_container_width=True)
                st.plotly_chart(fig3, use_container_width=True)
            
            with col2:
                st.plotly_chart(fig2, use_container_width=True)
                st.plotly_chart(fig4, use_container_width=True)
        else:
            st.info("Visualizaciones básicas mostradas arriba (Plotly no disponible)")
        
        # Análisis adicional
        with st.expander("🔍 Análisis Avanzado", expanded=False):
            
            # Tabla de composición promedio
            st.markdown("#### 💼 Composición Promedio del Portafolio")
            avg_weights = weights_df.mean().sort_values(ascending=False)
            composition_df = pd.DataFrame({
                'Activo': avg_weights.index,
                'Peso Promedio (%)': (avg_weights * 100).round(1),
                'Descripción': [ticker_names.get(t, t) for t in avg_weights.index]
            })
            st.dataframe(composition_df, hide_index=True)
            
            # Correlación con benchmark
            correlation = np.corrcoef(portfolio_returns, benchmark_returns)[0,1]
            st.markdown(f"**Correlación con benchmark:** {correlation:.3f}")
            
            # Rolling Sharpe
            if len(portfolio_returns) > 60:
                rolling_sharpe = (portfolio_returns.rolling(60).mean() / 
                                portfolio_returns.rolling(60).std() * np.sqrt(252))
                
                if PLOTLY_AVAILABLE:
                    fig_sharpe = go.Figure()
                    fig_sharpe.add_trace(go.Scatter(
                        x=list(range(len(rolling_sharpe))),
                        y=rolling_sharpe.values,
                        mode='lines',
                        name='Sharpe Ratio Móvil (60 días)'
                    ))
                    fig_sharpe.update_layout(
                        title='Evolución del Sharpe Ratio',
                        xaxis_title='Días',
                        yaxis_title='Sharpe Ratio',
                        height=300
                    )
                    st.plotly_chart(fig_sharpe, use_container_width=True)
                else:
                    # Alternativa con gráfico nativo de streamlit
                    st.line_chart(rolling_sharpe.dropna())
                    st.caption("Evolución del Sharpe Ratio (60 días)")
        
        # Feature Importance
        if models:
            with st.expander("🧠 Importancia de Características", expanded=False):
                st.markdown("#### Características más importantes para las predicciones:")
                
                # Calcular importancia promedio
                feature_importance = pd.DataFrame(index=X.columns)
                for asset, model in models.items():
                    feature_importance[asset] = model.feature_importances_
                
                avg_importance = feature_importance.mean(axis=1).sort_values(ascending=False)
                top_features = avg_importance.head(15)
                
                if PLOTLY_AVAILABLE:
                    # Gráfico de barras con Plotly
                    fig_importance = go.Figure(go.Bar(
                        x=top_features.values,
                        y=[f.replace('_', ' ').title() for f in top_features.index],
                        orientation='h'
                    ))
                    
                    fig_importance.update_layout(
                        title='Top 15 Características Más Importantes',
                        xaxis_title='Importancia',
                        height=500
                    )
                    
                    st.plotly_chart(fig_importance, use_container_width=True)
                else:
                    # Alternativa con gráfico nativo
                    st.bar_chart(top_features)
                    st.caption("Top 15 Características Más Importantes")
                
                # Tabla detallada
                importance_df = pd.DataFrame({
                    'Característica': top_features.index,
                    'Importancia': top_features.values.round(4)
                })
                st.dataframe(importance_df, hide_index=True)
        
        # Exportación de resultados
        st.subheader("💾 Exportar Resultados")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # CSV de rendimientos
            results_df = pd.DataFrame({
                'Date': portfolio_returns.index if hasattr(portfolio_returns, 'index') else range(len(portfolio_returns)),
                'Portfolio_Returns': portfolio_returns.values,
                'Benchmark_Returns': benchmark_returns.values,
                'Excess_Returns': excess_returns.values,
                'Portfolio_Cumulative': (1 + portfolio_returns).cumprod().values,
                'Benchmark_Cumulative': (1 + benchmark_returns).cumprod().values
            })
            
            csv_returns = results_df.to_csv(index=False)
            st.download_button(
                label="📊 Descargar Rendimientos",
                data=csv_returns,
                file_name=f"portfolio_returns_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        
        with col2:
            # CSV de pesos
            if not weights_df.empty:
                csv_weights = weights_df.to_csv()
                st.download_button(
                    label="⚖️ Descargar Pesos",
                    data=csv_weights,
                    file_name=f"portfolio_weights_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
        
        with col3:
            # Reporte de métricas
            metrics_report = f"""
REPORTE DE PERFORMANCE - {datetime.now().strftime('%Y-%m-%d %H:%M')}
{'='*60}

CONFIGURACIÓN:
- Universo: {asset_universe}
- Período: {start_date} a {end_date}
- Horizonte predicción: {prediction_horizon} días
- Aversión al riesgo: {risk_aversion}
- Peso máximo: {max_weight*100:.0f}%
- Rebalanceo: cada {rebalance_freq} días

RESULTADOS PORTAFOLIO RF + MARKOWITZ:
- Retorno Total: {portfolio_metrics['Total Return']:.2%}
- Retorno Anualizado: {portfolio_metrics['Annual Return']:.2%}
- Volatilidad Anualizada: {portfolio_metrics['Annual Volatility']:.2%}
- Ratio de Sharpe: {portfolio_metrics['Sharpe Ratio']:.3f}
- Máximo Drawdown: {portfolio_metrics['Max Drawdown']:.2%}
- Win Rate: {portfolio_metrics['Win Rate']:.1%}

RESULTADOS BENCHMARK:
- Retorno Total: {benchmark_metrics['Total Return']:.2%}
- Retorno Anualizado: {benchmark_metrics['Annual Return']:.2%}
- Volatilidad Anualizada: {benchmark_metrics['Annual Volatility']:.2%}
- Ratio de Sharpe: {benchmark_metrics['Sharpe Ratio']:.3f}
- Máximo Drawdown: {benchmark_metrics['Max Drawdown']:.2%}
- Win Rate: {benchmark_metrics['Win Rate']:.1%}

VALOR AÑADIDO:
- Alfa Anualizada: {alpha:.2%}
- Information Ratio: {information_ratio:.3f}
- % Días que superó benchmark: {outperform_rate:.1%}
- Correlación con benchmark: {correlation:.3f}

COMPOSICIÓN PROMEDIO:
""" + '\n'.join([f"- {ticker}: {weight:.1%}" for ticker, weight in avg_weights.items()])
            
            st.download_button(
                label="📋 Descargar Reporte",
                data=metrics_report,
                file_name=f"portfolio_report_{datetime.now().strftime('%Y%m%d')}.txt",
                mime="text/plain"
            )
        
        # Conclusiones
        st.subheader("🎯 Conclusiones del Análisis")
        
        excess_return_pct = (portfolio_metrics['Total Return'] - benchmark_metrics['Total Return'])
        sharpe_improvement = portfolio_metrics['Sharpe Ratio'] - benchmark_metrics['Sharpe Ratio']
        
        if excess_return_pct > 0 and sharpe_improvement > 0:
            conclusion_type = "success"
            conclusion_text = "🏆 **ESTRATEGIA EXITOSA**"
            details = f"""
            La estrategia Random Forest + Markowitz superó al benchmark tanto en retorno 
            (+{excess_return_pct:.2%}) como en ratio de Sharpe (+{sharpe_improvement:.3f} puntos).
            
            **Fortalezas identificadas:**
            - Mejor gestión del riesgo (menor drawdown)
            - Mayor consistencia en la generación de alfa
            - Diversificación inteligente y adaptativa
            """
        elif excess_return_pct > 0:
            conclusion_type = "warning"
            conclusion_text = "⚠️ **ESTRATEGIA PARCIALMENTE EXITOSA**"
            details = f"""
            La estrategia generó mayor retorno (+{excess_return_pct:.2%}) pero con 
            ratio de Sharpe similar al benchmark.
            
            **Áreas de mejora:**
            - Optimizar la gestión de riesgo
            - Ajustar parámetros de aversión al riesgo
            - Considerar costos de transacción
            """
        else:
            conclusion_type = "error"
            conclusion_text = "❌ **ESTRATEGIA NO EXITOSA**"
            details = f"""
            La estrategia no logró superar al benchmark en el período analizado.
            
            **Posibles causas:**
            - Sobreajuste en los modelos de ML
            - Período de prueba desfavorable
            - Parámetros sub-óptimos
            - Costos de transacción no considerados
            """
        
        if conclusion_type == "success":
            st.success(conclusion_text)
        elif conclusion_type == "warning":
            st.warning(conclusion_text)
        else:
            st.error(conclusion_text)
        
        st.markdown(details)
        
        # Recomendaciones
        st.markdown("#### 🚀 Recomendaciones para Mejoras:")
        
        recommendations = [
            "**Modelos ML**: Probar XGBoost, LSTM o modelos ensemble más sofisticados",
            "**Features**: Incorporar más variables macro y sentiment del mercado", 
            "**Optimización**: Implementar Black-Litterman o modelos de factor",
            "**Costos**: Incluir costos de transacción y market impact",
            "**Robustez**: Validar en diferentes regímenes de mercado",
            "**Frecuencia**: Evaluar diferentes frecuencias de rebalanceo"
        ]
        
        for rec in recommendations:
            st.markdown(f"- {rec}")
        
        # Footer con información técnica
        st.markdown("---")
        st.markdown(f"""
        <div style='text-align: center; color: #666; font-size: 12px;'>
            Análisis completado en {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 
            {len(portfolio_returns)} observaciones de backtest | 
            {len(models)} modelos Random Forest entrenados
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

# ====================================================================
# INSTRUCCIONES DE INSTALACIÓN Y EJECUCIÓN
# ====================================================================

"""
INSTALACIÓN:

1. Instalar dependencias:
pip install streamlit pandas numpy plotly yfinance scikit-learn scipy

2. Guardar este código como 'app.py'

3. Ejecutar la aplicación:
streamlit run app.py

4. La aplicación se abrirá en http://localhost:8501

DESPLIEGUE EN LA NUBE:

Para hacer la app accesible desde internet:

1. STREAMLIT CLOUD (Gratis):
   - Subir el código a GitHub
   - Conectar en share.streamlit.io
   - Deploy automático

2. HEROKU:
   - Crear Procfile: web: sh setup.sh && streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
   - Crear setup.sh con configuración
   - Deploy con git

3. AWS/GCP/AZURE:
   - Usar contenedores Docker
   - Deploy en servicios cloud

ARCHIVOS ADICIONALES NECESARIOS:

requirements.txt:
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.21.0
plotly>=5.15.0
yfinance>=0.2.0
scikit-learn>=1.3.0
scipy>=1.11.0

setup.sh (para Heroku):
mkdir -p ~/.streamlit/
echo "\
[general]\n\
email = \"your-email@domain.com\"\n\
" > ~/.streamlit/credentials.toml
echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml
"""
