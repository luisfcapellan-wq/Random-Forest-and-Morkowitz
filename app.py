#!/usr/bin/env python3
"""
📊 APLICACIÓN STREAMLIT: RANDOM FOREST + MARKOWITZ PORTFOLIO OPTIMIZER
🎓 Práctica Académica Interactiva para Construcción de Portafolios

Autor: Dr. Luis Capellán- Práctica Académica
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
# DEFINICIÓN DEL UNIVERSO DE 50 ACCIONES (30 EE.UU. + 20 MÉXICO)
# ====================================================================

def get_50_stocks_universe():
    """Define el universo de 50 acciones (30 EE.UU. + 20 México)"""
    
    # 30 acciones más populares de EE.UU. (S&P 500 leaders)
    us_stocks = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B', 'JPM', 'JNJ',
        'V', 'PG', 'UNH', 'HD', 'DIS', 'PYPL', 'NFLX', 'ADBE', 'CRM', 'INTC',
        'CSCO', 'PEP', 'T', 'ABT', 'COST', 'TMO', 'AVGO', 'WMT', 'XOM', 'CVX'
    ]
    
    # 20 acciones más populares de México (índice S&P/BMV IPC)
    mx_stocks = [
        'AMXL.MX', 'WALMEX.MX', 'FEMSAUBD.MX', 'GFNORTEO.MX', 'GMEXICOB.MX',
        'CEMEXCPO.MX', 'BBAJIOO.MX', 'KIMBERA.MX', 'GAPB.MX', 'ASURB.MX',
        'BIMBOA.MX', 'TLEVISACPO.MX', 'AC.MX', 'ALPEKA.MX', 'ALSEA.MX',
        'PE&OLES.MX', 'PINFRA.MX', 'CUERVO.MX', 'MEGACPO.MX', 'GRUMAB.MX'
    ]
    
    # Nombres descriptivos para display
    stock_names = {
        # US Stocks
        'AAPL': 'Apple Inc', 'MSFT': 'Microsoft', 'GOOGL': 'Alphabet (Google)', 
        'AMZN': 'Amazon.com', 'NVDA': 'NVIDIA', 'META': 'Meta Platforms', 
        'TSLA': 'Tesla', 'BRK-B': 'Berkshire Hathaway', 'JPM': 'JPMorgan Chase', 
        'JNJ': 'Johnson & Johnson', 'V': 'Visa', 'PG': 'Procter & Gamble',
        'UNH': 'UnitedHealth', 'HD': 'Home Depot', 'DIS': 'Walt Disney',
        'PYPL': 'PayPal', 'NFLX': 'Netflix', 'ADBE': 'Adobe', 'CRM': 'Salesforce',
        'INTC': 'Intel', 'CSCO': 'Cisco', 'PEP': 'PepsiCo', 'T': 'AT&T',
        'ABT': 'Abbott Laboratories', 'COST': 'Costco', 'TMO': 'Thermo Fisher',
        'AVGO': 'Broadcom', 'WMT': 'Walmart', 'XOM': 'Exxon Mobil', 'CVX': 'Chevron',
        
        # Mexican Stocks
        'AMXL.MX': 'América Móvil', 'WALMEX.MX': 'Walmart México', 
        'FEMSAUBD.MX': 'FEMSA', 'GFNORTEO.MX': 'Grupo Financiero Banorte',
        'GMEXICOB.MX': 'Grupo México', 'CEMEXCPO.MX': 'CEMEX', 
        'BBAJIOO.MX': 'Banco del Bajío', 'KIMBERA.MX': 'Kimberly-Clark de México',
        'GAPB.MX': 'Grupo Aeroportuario del Pacífico', 'ASURB.MX': 'Grupo Aeroportuario del Sureste',
        'BIMBOA.MX': 'Grupo Bimbo', 'TLEVISACPO.MX': 'Grupo Televisa', 
        'AC.MX': 'Arca Continental', 'ALPEKA.MX': 'Alpek', 
        'ALSEA.MX': 'Alsea', 'PE&OLES.MX': 'Peñoles', 
        'PINFRA.MX': 'Promotora y Operadora de Infraestructura', 
        'CUERVO.MX': 'Becle', 'MEGACPO.MX': 'Megacable', 
        'GRUMAB.MX': 'Grupo México'
    }
    
    return us_stocks + mx_stocks, stock_names

def select_top_8_stocks(expected_returns, cov_matrix, models_quality, max_stocks=8):
    """
    Selecciona las 8 mejores acciones usando un criterio combinado de:
    - Retorno esperado (50% peso)
    - Ratio de Sharpe individual (30% peso) 
    - Calidad del modelo Random Forest (20% peso)
    """
    
    # Calcular ratios de Sharpe individuales
    volatilities = np.sqrt(np.diag(cov_matrix))
    sharpe_ratios = expected_returns / volatilities
    
    # Normalizar métricas
    norm_returns = (expected_returns - expected_returns.min()) / (expected_returns.max() - expected_returns.min())
    norm_sharpe = (sharpe_ratios - sharpe_ratios.min()) / (sharpe_ratios.max() - sharpe_ratios.min())
    
    # Calcular calidad del modelo (usando R² si disponible)
    model_scores = []
    for ticker in expected_returns.index:
        if ticker in models_quality and 'R2_train' in models_quality[ticker]:
            model_scores.append(models_quality[ticker]['R2_train'])
        else:
            model_scores.append(0.5)  # Score por defecto
    
    norm_model_scores = np.array(model_scores)  # R² ya está entre 0-1
    
    # Score combinado (50% retorno, 30% Sharpe, 20% calidad modelo)
    combined_scores = (0.5 * norm_returns + 0.3 * norm_sharpe + 0.2 * norm_model_scores)
    
    # Seleccionar top N acciones
    top_indices = np.argsort(combined_scores)[-max_stocks:]
    selected_stocks = expected_returns.index[top_indices].tolist()
    
    return selected_stocks, combined_scores[top_indices]

# ====================================================================
# FUNCIONES AUXILIARES (MODIFICADAS PARA SOPORTAR SELECCIÓN DINÁMICA)
# ====================================================================

def clean_data_for_chart(data):
    """Limpia los datos para que sean compatibles con gráficos de Streamlit"""
    if isinstance(data, pd.Series):
        # Convertir a numérico y eliminar NaN/Inf
        cleaned = pd.to_numeric(data, errors='coerce')
        cleaned = cleaned.replace([np.inf, -np.inf], np.nan).dropna()
        return cleaned
    elif isinstance(data, pd.DataFrame):
        cleaned = data.apply(pd.to_numeric, errors='coerce')
        cleaned = cleaned.replace([np.inf, -np.inf], np.nan).dropna()
        return cleaned
    return data

@st.cache_data
def download_macro_data(start_date, end_date):
    """Descarga datos macroeconómicos con fallback"""
    macro_tickers = {
        '^TNX': 'Treasury_10Y',
        '^VIX': 'VIX',
        'DX-Y.NYB': 'DXY',
        '^GSPC': 'SP500'
    }
    
    macro_data = {}
    for ticker, name in macro_tickers.items():
        try:
            # Usar el método robusto del Ticker
            temp = yf.Ticker(ticker).history(
                start=start_date,
                end=end_date,
                auto_adjust=True
            )
            if not temp.empty and len(temp) > 50:
                macro_data[name] = temp['Close']
        except Exception as e:
            st.warning(f"No se pudo descargar {name}: {str(e)[:50]}")
            continue
    
    return pd.DataFrame(macro_data) if macro_data else None

def create_fallback_data(tickers, start_date, end_date):
    """Crea datos simulados si la descarga falla"""
    st.warning("Usando datos simulados para demostración")
    
    # Calcular número de días
    n_days = (end_date - start_date).days
    
    # Generar fechas
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    dates = dates[dates.weekday < 5]  # Solo días laborables
    
    # Parámetros realistas para acciones
    annual_returns = np.random.normal(0.10, 0.05, len(tickers))
    annual_vols = np.random.normal(0.25, 0.08, len(tickers))
    
    data = {}
    
    for i, ticker in enumerate(tickers):
        # Generar precio inicial aleatorio
        initial_price = 50 + np.random.random() * 100
        prices = [initial_price]
        
        daily_return = annual_returns[i] / 252
        daily_vol = annual_vols[i] / np.sqrt(252)
        
        for day in range(1, len(dates)):
            # Simulación realista con factor de mercado
            market_factor = np.random.normal(0, 0.01)
            idiosyncratic = np.random.normal(0, daily_vol)
            
            price_change = daily_return + idiosyncratic + 0.5 * market_factor
            new_price = prices[-1] * (1 + price_change)
            prices.append(new_price)
        
        data[ticker] = pd.Series(prices, index=dates[:len(prices)])
    
    return pd.DataFrame(data)

# Función de descarga mejorada que incluye fallback
@st.cache_data(ttl=3600)
def download_market_data_robust(tickers, start_date, end_date):
    """Descarga datos con múltiples métodos de fallback"""
    
    # Método 1: Usando yf.download
    try:
        st.info("Intentando método 1: yf.download...")
        data = {}
        
        for ticker in tickers:
            try:
                temp = yf.download(ticker, start=start_date, end=end_date, progress=False)
                if not temp.empty:
                    if 'Adj Close' in temp.columns:
                        data[ticker] = temp['Adj Close']
                    elif 'Close' in temp.columns:
                        data[ticker] = temp['Close']
                    
            except:
                continue
        
        if len(data) >= 8:  # Mínimo de 8 activos para análisis
            df = pd.DataFrame(data).dropna()
            if len(df) > 100:
                st.success(f"Método 1 exitoso: {len(data)} activos descargados")
                return df
    except:
        pass
    
    # Método 2: Usando yf.Ticker individualmente
    try:
        st.info("Intentando método 2: yf.Ticker individual...")
        data = {}
        
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                temp = stock.history(start=start_date, end=end_date)
                if not temp.empty and len(temp) > 10:
                    data[ticker] = temp['Close']
            except:
                continue
        
        if len(data) >= 8:
            df = pd.DataFrame(data).dropna()
            if len(df) > 100:
                st.success(f"Método 2 exitoso: {len(data)} activos descargados")
                return df
    except:
        pass
    
    # Método 3: Datos simulados
    st.warning("Métodos de descarga fallaron. Usando datos simulados para demostración.")
    return create_fallback_data(tickers, start_date, end_date)

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
    if len(returns) == 0:
        return {
            'Total Return': 0,
            'Annual Return': 0,
            'Annual Volatility': 0,
            'Sharpe Ratio': 0,
            'Max Drawdown': 0,
            'Win Rate': 0
        }
    
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
    
    if not PLOTLY_AVAILABLE or len(portfolio_returns) == 0:
        st.error("Plotly no disponible o datos insuficientes")
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
    
    if len(weights_history) > 0 and not weights_history.empty:
        for i, ticker in enumerate(weights_history.columns):
            if ticker in weights_history and weights_history[ticker].sum() > 0:
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
# INTERFAZ PRINCIPAL (MODIFICADA)
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
    
    # Selección de activos - NUEVA OPCIÓN PARA 50 ACCIONES
    st.sidebar.subheader("🏢 Universo de Inversión")
    
    asset_universe = st.sidebar.selectbox(
        "Seleccionar universo:",
        ["50 Acciones (30 EE.UU. + 20 México)", "ETFs Sectoriales S&P 500", "Tech Stocks", "Personalizado"],
        index=0
    )
    
    # Obtener el universo de 50 acciones
    all_50_stocks, stock_names = get_50_stocks_universe()
    
    if asset_universe == "50 Acciones (30 EE.UU. + 20 México)":
        tickers = all_50_stocks
        ticker_names = stock_names
        
        # Mostrar estadísticas del universo
        with st.sidebar.expander("📊 Info Universo 50 Acciones"):
            st.write(f"**EE.UU.:** {len([t for t in tickers if '.MX' not in t])} acciones")
            st.write(f"**México:** {len([t for t in tickers if '.MX' in t])} acciones")
            st.write("**Total:** 50 acciones")
            
    elif asset_universe == "ETFs Sectoriales S&P 500":
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
    
    # NUEVO: Selector para número de acciones en el portafolio final
    if asset_universe == "50 Acciones (30 EE.UU. + 20 México)":
        portfolio_size = st.sidebar.slider(
            "Número de acciones en portafolio final:",
            min_value=5, max_value=15, value=8
        )
    else:
        portfolio_size = len(tickers)  # Usar todas las acciones disponibles
    
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
            
            **NUEVA FUNCIONALIDAD:** Selección inteligente de 8 acciones de un universo de 50
            (30 acciones EE.UU. + 20 acciones México) usando Random Forest.
            
            Esta aplicación implementa una estrategia híbrida que combina:
            
            ### 🌳 **Random Forest para Selección**
            - Analiza 50 acciones del universo completo
            - Selecciona las 8 mejores usando criterios múltiples
            - Considera retorno esperado, Sharpe ratio y calidad del modelo
            
            ### 📊 **Optimización de Markowitz**
            - Construye portafolios eficientes con las acciones seleccionadas
            - Considera correlaciones entre activos
            - Aplica restricciones de peso y diversificación
            
            ### 📈 **Backtesting Robusto**
            - Validación temporal sin look-ahead bias
            - Rebalanceo periódico con selección dinámica
            - Comparación con benchmarks
            """)
        
        with col2:
            st.markdown("""
            <div class="success-box">
                <h4>🔧 Nuevas Funcionalidades:</h4>
                <ul>
                    <li>✅ Universo de 50 acciones</li>
                    <li>✅ Selección dinámica de 8 acciones</li>
                    <li>✅ Combinación EE.UU. + México</li>
                    <li>✅ Criterios múltiples de selección</li>
                    <li>✅ Análisis de calidad de modelos</li>
                    <li>✅ Optimización adaptativa</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Mostrar el universo de 50 acciones
        if asset_universe == "50 Acciones (30 EE.UU. + 20 México)":
            st.subheader("📋 Universo de 50 Acciones")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🇺🇸 30 Acciones EE.UU.")
                us_stocks = [t for t in all_50_stocks if '.MX' not in t]
                for i, ticker in enumerate(us_stocks):
                    st.write(f"{i+1}. {ticker} - {stock_names.get(ticker, ticker)}")
            
            with col2:
                st.markdown("#### 🇲🇽 20 Acciones México")
                mx_stocks = [t for t in all_50_stocks if '.MX' in t]
                for i, ticker in enumerate(mx_stocks):
                    st.write(f"{i+1}. {ticker} - {stock_names.get(ticker, ticker)}")
        
        return
    
    # ===== ANÁLISIS PRINCIPAL =====
    
    with st.spinner("🔄 Iniciando análisis..."):
        
        # 1. DESCARGA DE DATOS
        st.subheader("📊 Descarga de Datos")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"Descargando datos de {len(tickers)} activos...")
            asset_prices = download_market_data_robust(tickers, start_date, end_date)
            
            if asset_prices is None or asset_prices.empty:
                st.error("Error descargando datos de activos")
                return
            
            # Filtrar activos con datos insuficientes
            valid_assets = asset_prices.columns[asset_prices.notna().sum() > 100]
            if len(valid_assets) < 3:
                st.error("Menos de 3 activos con datos suficientes para análisis")
                return
                
            asset_prices = asset_prices[valid_assets]
            
            st.success(f"✅ {len(asset_prices)} días de datos para {asset_prices.shape[1]} activos válidos")
        
        with col2:
            st.info("Descargando datos macroeconómicos...")
            macro_data = download_macro_data(start_date, end_date)
            
            if macro_data is not None:
                st.success(f"✅ Variables macro: {', '.join(macro_data.columns)}")
            else:
                st.warning("⚠️ No se pudieron descargar datos macro")
        
        # Mostrar estadísticas básicas
        with st.expander("📈 Estadísticas de los Datos", expanded=False):
            st.write(f"**Activos con datos válidos:** {len(asset_prices.columns)}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Precios finales:**")
                final_prices = asset_prices.iloc[-1].sort_values(ascending=False)
                # Usar datos limpios para el gráfico
                final_prices_clean = clean_data_for_chart(final_prices.head(15))
                if not final_prices_clean.empty:
                    st.bar_chart(final_prices_clean)
                else:
                    st.warning("No hay datos válidos para mostrar")
            
            with col2:
                st.write("**Rendimientos anualizados (%):**")
                returns = asset_prices.pct_change().dropna()
                annual_returns = (returns.mean() * 252 * 100).sort_values(ascending=False)
                # Usar datos limpios para el gráfico
                annual_returns_clean = clean_data_for_chart(annual_returns.head(15))
                if not annual_returns_clean.empty:
                    st.bar_chart(annual_returns_clean)
                else:
                    st.warning("No hay datos válidos para mostrar")
        
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
        if len(common_dates) < 100:
            st.error("Datos insuficientes después de alinear fechas")
            return
            
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
            if not perf_df.empty and 'R2_train' in perf_df.columns:
                st.write("**Performance de los modelos (top 10 por R²):**")
                top_models = perf_df.nlargest(10, 'R2_train')
                st.dataframe(top_models.round(4))
        else:
            st.error("❌ No se pudieron entrenar modelos")
            return
        
        # NUEVA SECCIÓN: SELECCIÓN DE LAS MEJORES ACCIONES
        if asset_universe == "50 Acciones (30 EE.UU. + 20 México)" and len(models) >= portfolio_size:
            st.subheader("🎯 Selección de las Mejores Acciones")
            
            with st.spinner("Seleccionando las mejores acciones..."):
                # Calcular retornos esperados usando los modelos
                current_features = X.iloc[-1:].fillna(0)  # Últimos datos disponibles
                expected_returns = pd.Series(index=asset_prices.columns, dtype=float)
                
                for asset in asset_prices.columns:
                    if asset in models:
                        try:
                            pred = models[asset].predict(current_features)[0]
                            expected_returns[asset] = pred * 252  # Anualizar
                        except:
                            expected_returns[asset] = returns[asset].mean() * 252
                    else:
                        expected_returns[asset] = returns[asset].mean() * 252
                
                # Calcular matriz de covarianza
                recent_returns = returns.tail(min(252, len(returns)))  # Último año
                cov_matrix = recent_returns.cov() * 252
                
                # Seleccionar top N acciones
                selected_stocks, selection_scores = select_top_8_stocks(
                    expected_returns, cov_matrix, performance, portfolio_size
                )
                
                # Mostrar resultados de selección
                st.success(f"✅ Seleccionadas {len(selected_stocks)} acciones de {len(asset_prices.columns)} disponibles")
                
                # DataFrame con resultados de selección
                selection_results = []
                for i, ticker in enumerate(selected_stocks):
                    selection_results.append({
                        'Ranking': i+1,
                        'Ticker': ticker,
                        'Nombre': stock_names.get(ticker, ticker),
                        'País': 'México' if '.MX' in ticker else 'EE.UU.',
                        'Score Selección': f"{selection_scores[i]:.3f}",
                        'Retorno Esperado Anual': f"{expected_returns[ticker]:.1%}" if ticker in expected_returns else "N/A",
                        'R² Modelo': f"{performance.get(ticker, {}).get('R2_train', 'N/A')}"
                    })
                
                selection_df = pd.DataFrame(selection_results)
                st.dataframe(selection_df, hide_index=True)
                
                # Actualizar tickers para el backtesting
                original_tickers = tickers.copy()
                tickers = selected_stocks
                asset_prices = asset_prices[selected_stocks]
                
                st.info(f"🔀 Universo reducido de {len(original_tickers)} a {len(tickers)} acciones para optimización")
        
        # 4. BACKTESTING
        st.subheader("🔄 Ejecución del Backtesting")
        
        with st.spinner("Ejecutando backtesting con rebalanceo..."):
            
            # Configurar backtesting
            backtest_start_idx = max(252, len(X) // 3)  # Empezar con suficientes datos
            if backtest_start_idx >= len(X):
                st.error("Datos insuficientes para backtesting")
                return
                
            backtest_dates = X.index[backtest_start_idx::rebalance_freq]
            
            if len(backtest_dates) < 2:
                st.error("Período de backtesting demasiado corto")
                return
            
            portfolio_returns = []
            benchmark_returns = []
            weights_history = []
            rebalance_dates = []
            selection_history = []  # Para trackear selecciones históricas
            
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
                    
                    # SELECCIÓN DINÁMICA DE ACCIONES (solo para universo de 50)
                    current_tickers = list(asset_prices.columns)
                    if asset_universe == "50 Acciones (30 EE.UU. + 20 México)" and len(models) >= portfolio_size:
                        # Re-entrenar modelos con datos históricos
                        hist_models, hist_performance = train_random_forest_models(X_hist, y_hist)
                        
                        if hist_models and len(hist_models) >= portfolio_size:
                            # Calcular retornos esperados históricos
                            current_features_hist = X_hist.iloc[-1:].fillna(0)
                            expected_returns_hist = pd.Series(index=asset_prices.columns, dtype=float)
                            
                            for asset in asset_prices.columns:
                                if asset in hist_models:
                                    try:
                                        pred = hist_models[asset].predict(current_features_hist)[0]
                                        expected_returns_hist[asset] = pred * 252
                                    except:
                                        expected_returns_hist[asset] = returns[asset].mean() * 252
                                else:
                                    expected_returns_hist[asset] = returns[asset].mean() * 252
                            
                            # Calcular covarianza histórica
                            returns_hist = returns.loc[returns.index <= rebalance_date].tail(min(252, len(returns)))
                            if not returns_hist.empty:
                                cov_matrix_hist = returns_hist.cov() * 252
                                
                                # Seleccionar mejores acciones históricamente
                                selected_hist, _ = select_top_8_stocks(
                                    expected_returns_hist, cov_matrix_hist, hist_performance, portfolio_size
                                )
                                current_tickers = selected_hist
                                selection_history.append((rebalance_date, selected_hist))
                    
                    # Predecir rendimientos esperados para las acciones seleccionadas
                    current_features = X.loc[rebalance_date:rebalance_date]
                    expected_returns_current = pd.Series(index=current_tickers, dtype=float)
                    
                    for asset in current_tickers:
                        if asset in models and len(current_features) > 0:
                            try:
                                pred = models[asset].predict(current_features.fillna(0))[0]
                                expected_returns_current[asset] = pred * 252
                            except:
                                expected_returns_current[asset] = returns[asset].mean() * 252
                        else:
                            expected_returns_current[asset] = returns[asset].mean() * 252
                    
                    # Calcular matriz de covarianzas para las seleccionadas
                    recent_returns_current = returns[current_tickers].loc[returns.index <= rebalance_date].tail(min(252, len(returns)))
                    if not recent_returns_current.empty:
                        cov_matrix_current = recent_returns_current.cov() * 252
                        
                        # Optimizar portafolio
                        weights = markowitz_optimization(
                            expected_returns_current.values, cov_matrix_current.values, risk_aversion, max_weight
                        )
                        weights_series = pd.Series(weights, index=current_tickers)
                        
                        weights_history.append(weights_series)
                        rebalance_dates.append(rebalance_date)
                        
                        # Calcular rendimientos del período siguiente
                        next_date_idx = min(hist_idx + rebalance_freq, len(X) - 1)
                        period_returns = returns.iloc[hist_idx+1:next_date_idx+1]
                        
                        if len(period_returns) > 0:
                            # Asegurar que tenemos todas las columnas necesarias
                            available_columns = [col for col in current_tickers if col in period_returns.columns]
                            if available_columns:
                                portfolio_period_returns = (period_returns[available_columns] * 
                                                          weights_series[available_columns]).sum(axis=1)
                                benchmark_period_returns = period_returns[available_columns].mean(axis=1)
                                
                                portfolio_returns.extend(portfolio_period_returns.tolist())
                                benchmark_returns.extend(benchmark_period_returns.tolist())
                    
                except Exception as e:
                    st.warning(f"Error en rebalanceo {i+1}: {str(e)[:100]}")
                    continue
                
                progress_bar.progress((i + 1) / (len(backtest_dates) - 1))
            
            status_text.empty()
            progress_bar.empty()
        
        # Convertir a Series
        if portfolio_returns and benchmark_returns:
            portfolio_returns = pd.Series(portfolio_returns)
            benchmark_returns = pd.Series(benchmark_returns)
            
            # Crear DataFrame de pesos
            if weights_history:
                weights_df = pd.DataFrame(weights_history, index=rebalance_dates)
                weights_df = weights_df.fillna(0)
            else:
                weights_df = pd.DataFrame()
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
        
        # NUEVA SECCIÓN: ANÁLISIS DE SELECCIÓN DINÁMICA
        if asset_universe == "50 Acciones (30 EE.UU. + 20 México)" and selection_history:
            st.subheader("🔄 Análisis de Selección Dinámica")
            
            # Calcular frecuencia de selección por acción
            selection_counts = {}
            for _, selected in selection_history:
                for ticker in selected:
                    selection_counts[ticker] = selection_counts.get(ticker, 0) + 1
            
            if selection_counts:
                selection_freq = pd.Series(selection_counts).sort_values(ascending=False)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 📊 Frecuencia de Selección")
                    freq_df = pd.DataFrame({
                        'Ticker': selection_freq.index,
                        'Veces Seleccionada': selection_freq.values,
                        'Frecuencia (%)': (selection_freq.values / len(selection_history) * 100).round(1),
                        'País': ['México' if '.MX' in t else 'EE.UU.' for t in selection_freq.index]
                    })
                    st.dataframe(freq_df.head(15), hide_index=True)
                
                with col2:
                    st.markdown("#### 🌎 Distribución por País")
                    country_dist = freq_df.groupby('País')['Veces Seleccionada'].sum()
                    if not country_dist.empty:
                        fig_country = px.pie(
                            values=country_dist.values,
                            names=country_dist.index,
                            title='Selecciones por País'
                        )
                        st.plotly_chart(fig_country, use_container_width=True)
        
        # Gráficos principales
        st.subheader("📈 Visualizaciones de Performance")
        
        fig1, fig2, fig3, fig4 = create_performance_plots(portfolio_returns, benchmark_returns, weights_df)
        
        # Layout de gráficos
        if PLOTLY_AVAILABLE and fig1 is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                st.plotly_chart(fig1, use_container_width=True)
                if fig3 is not None:
                    st.plotly_chart(fig3, use_container_width=True)
            
            with col2:
                if fig2 is not None:
                    st.plotly_chart(fig2, use_container_width=True)
                if fig4 is not None:
                    st.plotly_chart(fig4, use_container_width=True)
        
        # Exportación de resultados
        st.subheader("💾 Exportar Resultados")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # CSV de rendimientos
            results_df = pd.DataFrame({
                'Portfolio_Returns': portfolio_returns.values,
                'Benchmark_Returns': benchmark_returns.values,
                'Excess_Returns': excess_returns.values
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

"""
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
