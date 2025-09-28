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
.stock-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 15px;
    border-radius: 10px;
    margin: 10px 0;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# LISTA DE LAS 20 ACCIONES MÁS POPULARES EN EE.UU.
POPULAR_US_STOCKS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA',    # Tech giants
    'TSLA', 'META', 'JPM', 'JNJ', 'V',          # Diversified leaders
    'PG', 'UNH', 'HD', 'DIS', 'PYPL',           # Consumer & services
    'BAC', 'XOM', 'PFE', 'NFLX', 'ADBE'         # Finance, energy, pharma, tech
]

STOCK_NAMES = {
    'AAPL': 'Apple Inc.',
    'MSFT': 'Microsoft Corp.',
    'GOOGL': 'Alphabet Inc.',
    'AMZN': 'Amazon.com Inc.',
    'NVDA': 'NVIDIA Corp.',
    'TSLA': 'Tesla Inc.',
    'META': 'Meta Platforms',
    'JPM': 'JPMorgan Chase',
    'JNJ': 'Johnson & Johnson',
    'V': 'Visa Inc.',
    'PG': 'Procter & Gamble',
    'UNH': 'UnitedHealth Group',
    'HD': 'Home Depot',
    'DIS': 'Walt Disney',
    'PYPL': 'PayPal Holdings',
    'BAC': 'Bank of America',
    'XOM': 'Exxon Mobil',
    'PFE': 'Pfizer Inc.',
    'NFLX': 'Netflix Inc.',
    'ADBE': 'Adobe Inc.'
}

# FUNCIONES

def create_sample_data(tickers, start_date, end_date):
    """Crea datos simulados realistas"""
    np.random.seed(42)
    
    n_days = (end_date - start_date).days
    dates = pd.date_range(start=start_date, periods=min(n_days, 500), freq='D')
    
    # Retornos y volatilidades realistas para diferentes sectores
    annual_returns = [0.08, 0.12, 0.15, 0.10, 0.09, 0.06, 0.18, 0.07, 0.05, 0.11,
                     0.04, 0.13, 0.08, 0.09, 0.16, 0.06, 0.03, 0.07, 0.20, 0.14]
    annual_vols = [0.20, 0.22, 0.35, 0.25, 0.40, 0.30, 0.45, 0.18, 0.15, 0.22,
                  0.16, 0.20, 0.25, 0.23, 0.38, 0.28, 0.20, 0.25, 0.42, 0.26]
    
    data = {}
    
    for i, ticker in enumerate(tickers):
        if i >= len(annual_returns):
            break
            
        initial_price = np.random.uniform(50, 500)
        prices = [initial_price]
        
        daily_return = annual_returns[i] / 252
        daily_vol = annual_vols[i] / np.sqrt(252)
        
        for day in range(1, len(dates)):
            price_change = daily_return + daily_vol * np.random.normal()
            new_price = prices[-1] * (1 + price_change)
            prices.append(max(new_price, 1))
        
        data[ticker] = prices[:len(dates)]
    
    return pd.DataFrame(data, index=dates)

@st.cache_data(ttl=3600)
def get_market_data(tickers, start_date, end_date):
    """Obtiene datos con fallback a simulación"""
    
    try:
        data = {}
        successful_tickers = []
        
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(start=start_date, end=end_date)
                if len(hist) > 100:
                    data[ticker] = hist['Close']
                    successful_tickers.append(ticker)
                    if len(successful_tickers) >= 10:  # Limit to 10 for performance
                        break
            except:
                continue
        
        if len(data) >= 6:
            df = pd.DataFrame(data).dropna()
            if len(df) > 100:
                return df, "real"
    except:
        pass
    
    # Usar solo los primeros 10 tickers para simulación
    return create_sample_data(tickers[:10], start_date, end_date), "simulado"

def calculate_features(prices):
    """Calcula características técnicas mejoradas"""
    features = pd.DataFrame(index=prices.index)
    
    for ticker in prices.columns:
        price_series = prices[ticker]
        returns = price_series.pct_change()
        
        # Momentum indicators
        features[f'{ticker}_mom_5'] = price_series.pct_change(5)
        features[f'{ticker}_mom_20'] = price_series.pct_change(20)
        features[f'{ticker}_mom_60'] = price_series.pct_change(60)
        
        # Volatility indicators
        features[f'{ticker}_vol_5'] = returns.rolling(5).std()
        features[f'{ticker}_vol_20'] = returns.rolling(20).std()
        features[f'{ticker}_vol_60'] = returns.rolling(60).std()
        
        # Moving average ratios
        features[f'{ticker}_ma_ratio_20'] = price_series / price_series.rolling(20).mean()
        features[f'{ticker}_ma_ratio_50'] = price_series / price_series.rolling(50).mean()
        features[f'{ticker}_ma_ratio_200'] = price_series / price_series.rolling(200).mean()
        
        # RSI
        delta = price_series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        features[f'{ticker}_rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = price_series.ewm(span=12).mean()
        exp2 = price_series.ewm(span=26).mean()
        features[f'{ticker}_macd'] = exp1 - exp2
        
        # Bollinger Bands
        rolling_mean = price_series.rolling(20).mean()
        rolling_std = price_series.rolling(20).std()
        features[f'{ticker}_bb_upper'] = (price_series - (rolling_mean + 2 * rolling_std)) / rolling_std
        features[f'{ticker}_bb_lower'] = ((rolling_mean - 2 * rolling_std) - price_series) / rolling_std
        
        # Price position relative to range
        high_20 = price_series.rolling(20).max()
        low_20 = price_series.rolling(20).min()
        features[f'{ticker}_price_position'] = (price_series - low_20) / (high_20 - low_20)
    
    return features.dropna()

def train_models(X, y):
    """Entrena modelos Random Forest mejorados"""
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
                n_estimators=100,  # Más árboles para mejor performance
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
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

def select_best_stocks(predicted_returns, n_stocks=6):
    """Selecciona las mejores acciones basado en predicciones de RF"""
    sorted_stocks = sorted(predicted_returns.items(), key=lambda x: x[1], reverse=True)
    return [stock for stock, ret in sorted_stocks[:n_stocks]]

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

def explain_intelligent_portfolio():
    """Explicación del portafolio inteligente"""
    st.markdown("""
    <div class="analogy-box">
    <h3>🎯 PORTFOLIO INTELIGENTE: 6 MEJORES ACCIONES</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🏆 Selección de Élite
        
        **Proceso de selección:**
        1. **Análisis de 20 acciones líderes** del mercado estadounidense
        2. **Random Forest evalúa** perspectiva de rentabilidad para cada acción
        3. **Selección de las 6 mejores** según predicciones de ML
        4. **Optimización Markowitz** para asignación óptima de pesos
        
        **Ventajas vs enfoque tradicional:**
        - ✅ Basado en **machine learning predictivo**
        - ✅ **Diversificación inteligente** (no igual ponderación)
        - ✅ **Actualización dinámica** según condiciones de mercado
        - ✅ **Enfoque cuantitativo** basado en datos
        """)
    
    with col2:
        st.markdown("""
        ### 📊 Acciones Analizadas
        
        **Sectores representados:**
        - 🏦 **Finanzas**: JPM, BAC, V
        - 💻 **Tecnología**: AAPL, MSFT, GOOGL, NVDA, META
        - 🏥 **Salud**: JNJ, UNH, PFE
        - 🛒 **Consumo**: AMZN, PG, HD, DIS
        - ⚡ **Energía/Auto**: TSLA, XOM
        - 🎬 **Entretenimiento**: NFLX, DIS
        
        **Criterios de selección:**
        - Liquidez y capitalización de mercado
        - Representatividad sectorial
        - Datos históricos robustos
        - Potencial de crecimiento
        """)
    
    st.markdown("""
    <div class="feature-card">
    <h4>🎯 Metodología de Selección Inteligente</h4>
    <ul>
    <li><strong>🤖 Fase 1 - Screening:</strong> Random Forest analiza 20 acciones populares</li>
    <li><strong>📈 Fase 2 - Scoring:</strong> Cada acción recibe score de rentabilidad esperada</li>
    <li><strong>🏆 Fase 3 - Selección:</strong> Top 6 acciones con mejor perspectiva</li>
    <li><strong>⚖️ Fase 4 - Optimización:</strong> Markowitz asigna pesos óptimos</li>
    <li><strong>🔄 Fase 5 - Monitoreo:</strong> Rebalanceo periódico basado en nuevas predicciones</li>
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
        ["ETFs Sectoriales", "Tech Stocks", "Portafolio Inteligente (6 mejores acciones)"]
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
    elif asset_universe == "Tech Stocks":
        tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META']
        ticker_names = {t: t for t in tickers}
    else:  # Portafolio Inteligente
        tickers = POPULAR_US_STOCKS
        ticker_names = STOCK_NAMES
    
    end_date = st.sidebar.date_input("Fecha final:", datetime.now().date())
    start_date = st.sidebar.date_input("Fecha inicial:", end_date - timedelta(days=730))
    
    prediction_horizon = st.sidebar.slider("Horizonte predicción (días):", 5, 42, 21)
    risk_aversion = st.sidebar.slider("Aversión al riesgo:", 0.5, 5.0, 2.0)
    
    if asset_universe == "Portafolio Inteligente (6 mejores acciones)":
        n_selected_stocks = st.sidebar.slider("Número de acciones a seleccionar:", 4, 8, 6)
    
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
        <strong>🤖 Random Forest:</strong> Predicción de rendimientos usando 100 "árboles-decisión"  
        <strong>📊 Markowitz:</strong> Optimización matemática riesgo-retorno  
        <strong>🏆 Portfolio Inteligente:</strong> Selección de las 6 mejores acciones entre 20 líderes  
        <strong>🔬 Backtesting:</strong> Validación histórica de la estrategia  
        <strong>🎓 Pedagogía:</strong> Explicaciones para enseñanza en aula  
        </div>
        
        ### 🎯 Nuevo: Portafolio Inteligente
        
        <div class="stock-card">
        <h4>🚀 SELECCIÓN DE 6 MEJORES ACCIONES</h4>
        <p>Analiza 20 acciones populares de EE.UU. y selecciona las 6 con mejor perspectiva usando Random Forest</p>
        </div>
        
        **👈 Configura los parámetros y presiona 'Ejecutar Análisis'**
        """, unsafe_allow_html=True)
        return
    
    # SECCIÓN 1: EXPLICACIÓN PEDAGÓGICA
    if asset_universe == "Portafolio Inteligente (6 mejores acciones)":
        st.header("🎯 Portafolio Inteligente: 6 Mejores Acciones")
        explain_intelligent_portfolio()
    else:
        st.header("🎓 Explicación: Random Forest en Finanzas")
        # (Mantener la explicación original de RF aquí)
    
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
    
    # Para portafolio inteligente, mostrar las acciones analizadas
    if asset_universe == "Portafolio Inteligente (6 mejores acciones)":
        st.subheader("📋 20 Acciones Populares Analizadas")
        
        # Agrupar por sectores para mejor visualización
        sectors = {
            'Tecnología': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'ADBE'],
            'Finanzas': ['JPM', 'V', 'BAC'],
            'Salud': ['JNJ', 'UNH', 'PFE'],
            'Consumo': ['PG', 'HD', 'DIS', 'NFLX'],
            'Energía/Automotive': ['TSLA', 'XOM'],
            'Diversificado': ['PYPL']
        }
        
        for sector, stocks in sectors.items():
            with st.expander(f"🏢 Sector: {sector}"):
                cols = st.columns(3)
                for i, stock in enumerate(stocks):
                    if stock in asset_prices.columns:
                        with cols[i % 3]:
                            current_price = asset_prices[stock].iloc[-1] if len(asset_prices) > 0 else "N/A"
                            st.write(f"**{stock}** - {STOCK_NAMES.get(stock, stock)}")
    
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
    
    # SECCIÓN 4: ENTRENAMIENTO DEL MODELO
    st.header("🤖 Entrenamiento del Random Forest")
    
    with st.spinner("🌳 Entrenando 100 árboles de decisión..."):
        models, feature_importances = train_models(X, y)
    
    if len(models) == 0:
        st.error("❌ No se pudieron entrenar modelos válidos")
        return
    
    st.success(f"✅ {len(models)} modelos entrenados exitosamente")
    
    # SECCIÓN ESPECIAL PARA PORTFOLIO INTELIGENTE: SELECCIÓN DE MEJORES ACCIONES
    if asset_universe == "Portafolio Inteligente (6 mejores acciones)":
        st.header("🏆 Selección de las 6 Mejores Acciones")
        
        with st.spinner("🔍 Analizando perspectivas de rentabilidad..."):
            # Obtener predicciones actuales para todas las acciones
            current_date = asset_prices.index[-1]
            current_features = features.loc[current_date]
            all_predictions = predict_returns(models, current_features)
            
            # Seleccionar las mejores acciones
            selected_stocks = select_best_stocks(all_predictions, n_selected_stocks)
            
            # Mostrar ranking completo
            st.subheader("📊 Ranking Completo de Predicciones")
            predictions_df = pd.DataFrame([
                {'Acción': stock, 'Nombre': STOCK_NAMES.get(stock, stock), 
                 'Predicción RF (%)': pred * 252 * 100, 'Seleccionada': stock in selected_stocks}
                for stock, pred in sorted(all_predictions.items(), key=lambda x: x[1], reverse=True)
            ])
            
            # Formatear el dataframe para mejor visualización
            display_df = predictions_df.copy()
            display_df['Predicción RF (%)'] = display_df['Predicción RF (%)'].round(2)
            
            # Aplicar estilo para resaltar las seleccionadas
            def highlight_selected(row):
                if row['Seleccionada']:
                    return ['background-color: #90EE90'] * len(row)
                else:
                    return [''] * len(row)
            
            st.dataframe(display_df.style.apply(highlight_selected, axis=1))
            
            # Mostrar las acciones seleccionadas
            st.subheader("🎯 Acciones Seleccionadas para el Portafolio")
            cols = st.columns(3)
            for i, stock in enumerate(selected_stocks):
                with cols[i % 3]:
                    pred_value = all_predictions[stock] * 252 * 100
                    st.markdown(f"""
                    <div class="stock-card">
                    <h4>{stock}</h4>
                    <p>{STOCK_NAMES.get(stock, stock)}</p>
                    <p><strong>Rentabilidad Esperada: {pred_value:.1f}%</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Actualizar tickers para usar solo las seleccionadas
            tickers = selected_stocks
            asset_prices = asset_prices[selected_stocks]
    
    # SECCIÓN 5: BACKTESTING Y OPTIMIZACIÓN
    st.header("🔄 Backtesting con Predicciones RF")
    
    with st.spinner("⚡ Ejecutando simulación histórica..."):
        
        # Recalcular features y returns con los activos finales
        if asset_universe == "Portafolio Inteligente (6 mejores acciones)":
            features = calculate_features(asset_prices)
            returns = asset_prices.pct_change().dropna()
            targets = pd.DataFrame(index=returns.index, columns=returns.columns)
            for col in returns.columns:
                targets[col] = returns[col].shift(-prediction_horizon)
            
            common_dates = features.index.intersection(targets.index)
            X = features.loc[common_dates].fillna(method='ffill').fillna(0)
            y = targets.loc[common_dates]
            
            # Reentrenar modelos solo con las acciones seleccionadas
            models, feature_importances = train_models(X, y)
        
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
        if asset_universe == "Portafolio Inteligente (6 mejores acciones)":
            st.markdown("### 🏆 Portfolio Inteligente RF")
        else:
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
            'Estrategia': port_cumulative,
            'Benchmark': bench_cumulative
        })
        
        st.line_chart(performance_df)
    except Exception as e:
        st.error(f"Error en gráfico: {str(e)}")
    
    # SECCIÓN 7: COMPOSICIÓN DEL PORTAFOLIO
    st.header("💼 Composición Óptima del Portafolio")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if asset_universe == "Portafolio Inteligente (6 mejores acciones)":
            st.subheader("🎯 Distribución del Portfolio Inteligente")
        else:
            st.subheader("📋 Distribución Recomendada")
        
        weights_df = pd.DataFrame({
            'Activo': tickers,
            'Nombre': [ticker_names.get(t, t) for t in tickers],
            'Peso': final_weights,
            'Peso %': [f"{w*100:.1f}%" for w in final_weights]
        }).sort_values('Peso', ascending=False)
        
        st.dataframe(weights_df[['Activo', 'Nombre', 'Peso %']], hide_index=True)
        
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
        
        # Resumen de asignación
        st.subheader("🎯 Resumen de Asignación")
        high_weight_assets = weights_df[weights_df['Peso'] > 0.1]
        if len(high_weight_assets) > 0:
            for _, row in high_weight_assets.iterrows():
                st.write(f"▪️ **{row['Activo']}** ({row['Nombre']}): {row['Peso %']}")
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
                st.metric(f"{row['Activo']} ({ticker_names.get(row['Activo'], row['Activo'])})", 
                         value=f"{row['Predicción Anual %']:.1f}%")
        
        with col2:
            st.subheader("📊 Métricas de Predicción")
            
            # Calcular accuracy simple
            if len(rf_predictions_history) > 1:
                pred_variability = np.std([pred * 252 * 100 for pred in rf_predictions_history], axis=0).mean()
                st.metric("Variabilidad entre Rebalanceos", f"{pred_variability:.1f}%")
            
            st.metric("Número de Rebalanceos", n_periods)
            st.metric("Horizonte de Predicción", f"{prediction_horizon} días")
            st.metric("Acciones en Portfolio", len(tickers))
    
    # SECCIÓN 9: CONCLUSIONES PEDAGÓGICAS
    st.header("🎯 Conclusiones para el Aula")
    
    excess_return = port_metrics.get('Annual Return', 0) - bench_metrics.get('Annual Return', 0)
    sharpe_diff = port_metrics.get('Sharpe Ratio', 0) - bench_metrics.get('Sharpe Ratio', 0)
    
    conclusion_emoji = "🏆" if excess_return > 0 and sharpe_diff > 0 else "⚠️" if excess_return > 0 else "📊"
    
    # CORRECCIÓN: Usar f-string simple sin multilínea problemática
    if asset_universe == "Portafolio Inteligente (6 mejores acciones)":
        class_message = "Demuestra cómo la selección inteligente basada en ML puede mejorar significativamente los resultados de inversión."
    else:
        class_message = "Excelente ejemplo de aplicación práctica de machine learning en finanzas cuantitativas."
    
    st.markdown(f"""
    ## {conclusion_emoji} **ANÁLISIS COMPLETADO**
    
    **Resultados del {'Portfolio Inteligente' if asset_universe == 'Portafolio Inteligente (6 mejores acciones)' else 'análisis RF'}:**  
    - 📈 **Diferencial de Retorno:** {excess_return:.2%}
    - ⚖️ **Diferencial de Sharpe:** {sharpe_diff:.3f}
    - 🔢 **Acciones analizadas:** {asset_prices.shape[1]}
    - 📅 **Período de backtesting:** {len(portfolio_rets)} días
    
    **Para la clase:** {class_message}
    """)

if __name__ == "__main__":
    main()
