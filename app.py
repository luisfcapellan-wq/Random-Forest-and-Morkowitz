#!/usr/bin/env python3
"""
APLICACIÓN STREAMLIT: RANDOM FOREST + MARKOWITZ PORTFOLIO OPTIMIZER
Versión corregida para problemas de datos
"""

import streamlit as st
import pandas as pd
import numpy as np
import warnings
import time
from datetime import datetime, timedelta

# Importaciones con manejo de errores
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    st.error("yfinance no está instalado")
    st.stop()

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    SKLEARN_AVAILABLE = True
except ImportError:
    st.error("scikit-learn no está instalado")
    st.stop()

try:
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError:
    st.error("scipy no está instalado")
    st.stop()

warnings.filterwarnings('ignore')

# CONFIGURACIÓN DE LA PÁGINA
st.set_page_config(
    page_title="Random Forest + Markowitz Portfolio",
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
</style>
""", unsafe_allow_html=True)

# FUNCIONES CORREGIDAS

def create_sample_data(tickers, start_date, end_date):
    """Crea datos simulados realistas"""
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    dates = dates[dates.weekday < 5]  # Solo días laborables
    
    # Parámetros realistas
    annual_returns = [0.08, 0.12, 0.10, 0.09, 0.06, 0.10, 0.08, 0.07, 0.06]
    annual_vols = [0.20, 0.25, 0.18, 0.20, 0.30, 0.22, 0.16, 0.24, 0.14]
    
    data = {}
    
    for i, ticker in enumerate(tickers[:len(annual_returns)]):
        initial_price = 50 + np.random.random() * 100
        prices = [initial_price]
        
        daily_return = annual_returns[i] / 252
        daily_vol = annual_vols[i] / np.sqrt(252)
        
        for day in range(1, len(dates)):
            market_factor = np.random.normal(0, 0.01)
            idiosyncratic = np.random.normal(0, daily_vol)
            
            price_change = daily_return + idiosyncratic + 0.5 * market_factor
            new_price = prices[-1] * (1 + price_change)
            prices.append(max(new_price, 1))  # Precio mínimo de $1
        
        data[ticker] = pd.Series(prices, index=dates[:len(prices)])
    
    return pd.DataFrame(data)

@st.cache_data(ttl=3600)
def download_market_data_safe(tickers, start_date, end_date):
    """Descarga datos con múltiples métodos de fallback"""
    
    # Intentar descargar datos reales
    try:
        st.info("Descargando datos de Yahoo Finance...")
        data = {}
        
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(start=start_date, end=end_date, auto_adjust=True)
                
                if not hist.empty and len(hist) > 50:
                    data[ticker] = hist['Close']
                    
            except Exception as e:
                continue
        
        if len(data) >= 3:
            df = pd.DataFrame(data).dropna()
            if len(df) > 100:
                st.success(f"Datos reales descargados: {len(data)} activos, {len(df)} días")
                return df
        
        # Si no hay suficientes datos reales, intentar con tickers más comunes
        st.warning("Intentando con tickers alternativos...")
        common_tickers = ['SPY', 'QQQ', 'IWM', 'VTI', 'EFA', 'EEM', 'AGG', 'GLD', 'SLV']
        data = {}
        
        for ticker in common_tickers[:len(tickers)]:
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(start=start_date, end=end_date, auto_adjust=True)
                
                if not hist.empty and len(hist) > 50:
                    data[tickers[len(data)]] = hist['Close']  # Usar nombre original
                    
                if len(data) >= len(tickers):
                    break
                    
            except:
                continue
        
        if len(data) >= 3:
            df = pd.DataFrame(data).dropna()
            if len(df) > 100:
                st.success(f"Datos alternativos descargados: {len(data)} activos")
                return df
    
    except Exception as e:
        st.warning(f"Error en descarga: {str(e)[:100]}")
    
    # Fallback final: datos simulados
    st.info("Usando datos simulados para demostración")
    return create_sample_data(tickers, start_date, end_date)

def calculate_technical_indicators(prices):
    """Calcula indicadores técnicos de manera robusta"""
    features = pd.DataFrame(index=prices.index)
    
    for ticker in prices.columns:
        try:
            price_series = prices[ticker].dropna()
            
            if len(price_series) < 100:
                continue
            
            # Momentum
            if len(price_series) >= 21:
                features[f'{ticker}_momentum_1m'] = price_series.pct_change(21)
            if len(price_series) >= 63:
                features[f'{ticker}_momentum_3m'] = price_series.pct_change(63)
            
            # Volatilidad
            features[f'{ticker}_volatility'] = price_series.pct_change().rolling(20).std()
            
            # Moving Average Ratios
            if len(price_series) >= 50:
                features[f'{ticker}_ma_ratio'] = price_series / price_series.rolling(50).mean()
            
            # RSI simplificado
            delta = price_series.diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / (loss + 1e-10)  # Evitar división por cero
            features[f'{ticker}_rsi'] = 100 - (100 / (1 + rs))
            
        except Exception as e:
            st.warning(f"Error calculando indicadores para {ticker}: {str(e)[:50]}")
            continue
    
    return features.dropna()

def train_random_forest_models(X, y, test_size=0.2):
    """Entrena modelos Random Forest de manera robusta"""
    models = {}
    performance = {}
    
    if len(X) < 100:
        st.error("Datos insuficientes para entrenamiento")
        return {}, {}
    
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    progress_bar = st.progress(0)
    
    for i, asset in enumerate(y.columns):
        try:
            # Filtrar datos válidos
            mask = ~(y_train[asset].isna() | X_train.isna().any(axis=1))
            if mask.sum() < 50:
                continue
                
            X_asset = X_train[mask]
            y_asset = y_train[asset][mask]
            
            # Modelo más simple para evitar overfitting
            rf = RandomForestRegressor(
                n_estimators=30,
                max_depth=5,
                min_samples_split=10,
                random_state=42,
                n_jobs=1
            )
            
            rf.fit(X_asset, y_asset)
            models[asset] = rf
            
            # Evaluación básica
            train_score = rf.score(X_asset, y_asset)
            performance[asset] = {'R2_train': train_score}
            
        except Exception as e:
            st.warning(f"Error entrenando modelo para {asset}: {str(e)[:50]}")
            continue
        
        progress_bar.progress((i + 1) / len(y.columns))
    
    progress_bar.empty()
    return models, performance

def markowitz_optimization_safe(expected_returns, cov_matrix, risk_aversion=2, max_weight=0.25):
    """Optimización de Markowitz con manejo de errores"""
    try:
        n_assets = len(expected_returns)
        
        # Verificar que la matriz de covarianzas es válida
        if np.any(np.isnan(cov_matrix)) or np.any(np.isinf(cov_matrix)):
            # Usar matriz identidad como fallback
            cov_matrix = np.eye(n_assets) * 0.01
        
        def objective(weights):
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
            return -(portfolio_return - (risk_aversion/2) * portfolio_variance)
        
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bounds = [(0.01, max_weight) for _ in range(n_assets)]  # Mínimo 1%
        
        result = minimize(
            objective,
            x0=np.ones(n_assets) / n_assets,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 100}
        )
        
        if result.success:
            return result.x
        else:
            # Fallback: pesos iguales
            return np.ones(n_assets) / n_assets
            
    except Exception as e:
        st.warning(f"Error en optimización: {str(e)[:50]}")
        return np.ones(len(expected_returns)) / len(expected_returns)

def calculate_portfolio_metrics_safe(returns):
    """Calcula métricas con manejo de errores"""
    try:
        returns_clean = returns.dropna()
        
        if len(returns_clean) == 0:
            return {}
        
        total_return = (1 + returns_clean).prod() - 1
        annual_return = (1 + returns_clean.mean())**252 - 1
        annual_volatility = returns_clean.std() * np.sqrt(252)
        sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0
        
        # Max Drawdown
        cumulative = (1 + returns_clean).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        return {
            'Total Return': total_return,
            'Annual Return': annual_return,
            'Annual Volatility': annual_volatility,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown': max_drawdown,
            'Win Rate': (returns_clean > 0).mean()
        }
    except Exception as e:
        st.error(f"Error calculando métricas: {str(e)}")
        return {}

# INTERFAZ PRINCIPAL SIMPLIFICADA
def main():
    # Título
    st.markdown("""
    <div class="main-header">
        <h1>📊 Random Forest + Markowitz Portfolio</h1>
        <p>Práctica Académica Interactiva</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("Configuración")
    
    # Parámetros básicos
    asset_universe = st.sidebar.selectbox(
        "Universo:",
        ["ETFs Sectoriales", "Tech Stocks", "Personalizado"]
    )
    
    if asset_universe == "ETFs Sectoriales":
        tickers = ['XLF', 'XLK', 'XLV', 'XLI', 'XLE']  # Reducido para mayor robustez
    elif asset_universe == "Tech Stocks":
        tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META']
    else:
        ticker_input = st.sidebar.text_input("Tickers:", "AAPL,GOOGL,MSFT")
        tickers = [t.strip().upper() for t in ticker_input.split(',')]
    
    end_date = st.sidebar.date_input("Fecha final:", datetime.now().date())
    start_date = st.sidebar.date_input("Fecha inicial:", end_date - timedelta(days=365))
    
    prediction_horizon = st.sidebar.slider("Horizonte (días):", 5, 63, 21)
    risk_aversion = st.sidebar.slider("Aversión al riesgo:", 0.5, 10.0, 2.0)
    max_weight = st.sidebar.slider("Peso máximo (%):", 10, 50, 30) / 100
    
    run_analysis = st.sidebar.button("🚀 Ejecutar Análisis", type="primary")
    
    if not run_analysis:
        st.markdown("""
        ## Bienvenido a la Práctica
        
        Esta aplicación combina:
        - Random Forest para predicción de rendimientos
        - Optimización de Markowitz para construcción de portafolios
        - Backtesting con métricas de performance
        
        Configure los parámetros y presione "Ejecutar Análisis"
        """)
        return
    
    # ANÁLISIS PRINCIPAL
    
    # 1. DATOS
    st.subheader("📊 Datos del Mercado")
    
    with st.spinner("Obteniendo datos..."):
        asset_prices = download_market_data_safe(tickers, start_date, end_date)
        
        if asset_prices is None or asset_prices.empty:
            st.error("No se pudieron obtener datos")
            return
    
    st.success(f"Datos obtenidos: {asset_prices.shape[1]} activos, {len(asset_prices)} días")
    
    # Mostrar datos básicos
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Precios Finales:**")
        try:
            final_prices = asset_prices.iloc[-1]
            # Crear DataFrame con nombres limpios para evitar errores de Altair
            chart_data = pd.DataFrame({
                'Ticker': final_prices.index,
                'Price': final_prices.values
            }).set_index('Ticker')
            st.bar_chart(chart_data)
        except Exception as e:
            # Fallback: mostrar como tabla
            st.dataframe(asset_prices.iloc[-1:].T)
    
    with col2:
        returns = asset_prices.pct_change().dropna()
        annual_rets = returns.mean() * 252
        st.write("**Rendimientos Anualizados:**")
        try:
            # Crear DataFrame con nombres limpios
            returns_data = pd.DataFrame({
                'Ticker': annual_rets.index,
                'Annual_Return': annual_rets.values
            }).set_index('Ticker')
            st.bar_chart(returns_data)
        except Exception as e:
            # Fallback: mostrar como tabla
            annual_rets_df = pd.DataFrame({'Retorno Anual': annual_rets})
            st.dataframe(annual_rets_df)
    
    # 2. CARACTERÍSTICAS
    st.subheader("🔧 Características Técnicas")
    
    with st.spinner("Calculando indicadores..."):
        features = calculate_technical_indicators(asset_prices)
        
        # Crear targets
        targets = pd.DataFrame(index=returns.index, columns=returns.columns)
        for col in returns.columns:
            targets[col] = returns[col].shift(-prediction_horizon)
        
        # Alinear datos
        common_dates = features.index.intersection(targets.index)
        if len(common_dates) < 100:
            st.error("Datos insuficientes después del procesamiento")
            return
            
        X = features.loc[common_dates].fillna(0)
        y = targets.loc[common_dates]
    
    st.success(f"Dataset final: {len(X)} observaciones, {X.shape[1]} características")
    
    # 3. MODELOS
    st.subheader("🤖 Random Forest")
    
    with st.spinner("Entrenando modelos..."):
        models, performance = train_random_forest_models(X, y)
    
    if not models:
        st.error("No se pudieron entrenar modelos")
        return
    
    st.success(f"{len(models)} modelos entrenados")
    
    if performance:
        perf_df = pd.DataFrame(performance).T
        st.dataframe(perf_df.round(4))
    
    # 4. BACKTESTING SIMPLIFICADO
    st.subheader("🔄 Backtesting")
    
    with st.spinner("Ejecutando análisis..."):
        
        # Configuración simple
        split_point = int(len(X) * 0.8)
        test_returns = returns.iloc[split_point:]
        
        if len(test_returns) < 50:
            st.error("Período de prueba muy corto")
            return
        
        # Simulación de rebalanceo mensual
        portfolio_returns = []
        benchmark_returns = []
        portfolio_weights_history = []
        rebalance_dates = []
        
        n_rebalances = max(3, len(test_returns) // 21)  # Mínimo 3 rebalanceos
        
        for rebal in range(n_rebalances):
            start_idx = rebal * 21
            end_idx = min((rebal + 1) * 21, len(test_returns))
            
            if start_idx >= len(test_returns):
                break
            
            # Usar datos hasta este punto para predicción
            hist_data = returns.iloc[:split_point + start_idx]
            
            if len(hist_data) < 100:
                continue
            
            # Predicciones usando modelos o media histórica
            expected_rets = pd.Series(index=returns.columns)
            for asset in returns.columns:
                if asset in models and len(X) > split_point + start_idx:
                    try:
                        # Usar características más recientes para predicción
                        recent_features = X.iloc[split_point + start_idx - 1:split_point + start_idx]
                        if len(recent_features) > 0:
                            pred = models[asset].predict(recent_features.values.reshape(1, -1))[0]
                            expected_rets[asset] = pred * 252
                        else:
                            expected_rets[asset] = hist_data[asset].tail(63).mean() * 252
                    except:
                        expected_rets[asset] = hist_data[asset].tail(63).mean() * 252
                else:
                    expected_rets[asset] = hist_data[asset].tail(63).mean() * 252
            
            # Matriz de covarianzas
            cov_matrix = hist_data.tail(126).cov() * 252
            
            # Optimizar
            weights = markowitz_optimization_safe(expected_rets, cov_matrix, risk_aversion, max_weight)
            weights_series = pd.Series(weights, index=returns.columns)
            
            # Guardar información del rebalanceo
            portfolio_weights_history.append(weights_series.copy())
            rebalance_dates.append(f"Rebalanceo {rebal + 1}")
            
            # Rendimientos del período
            period_rets = test_returns.iloc[start_idx:end_idx]
            if len(period_rets) > 0:
                # Calcular rendimientos diarios del portafolio
                daily_port_rets = (period_rets * weights_series).sum(axis=1)
                daily_bench_rets = period_rets.mean(axis=1)
                
                portfolio_returns.extend(daily_port_rets.tolist())
                benchmark_returns.extend(daily_bench_rets.tolist())
    
    if not portfolio_returns:
        st.error("No se pudieron calcular rendimientos del backtest")
        return
    
    # Convertir a series con índices apropiados
    portfolio_returns = pd.Series(portfolio_returns)
    benchmark_returns = pd.Series(benchmark_returns)
    
    # Mostrar composición del portafolio
    st.subheader("💼 Composición del Portafolio")
    
    if portfolio_weights_history:
        # Pesos promedio durante el período
        avg_weights = pd.DataFrame(portfolio_weights_history, index=rebalance_dates).mean()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Pesos Promedio del Portafolio:**")
            weights_df = pd.DataFrame({
                'Activo': avg_weights.index,
                'Peso (%)': (avg_weights * 100).round(1),
                'Inversión en $1000': (avg_weights * 1000).round(0).astype(int)
            })
            st.dataframe(weights_df, hide_index=True)
            
        with col2:
            st.write("**Evolución de Pesos:**")
            try:
                weights_chart_data = pd.DataFrame(portfolio_weights_history, 
                                                index=rebalance_dates)
                # Transponer para que cada activo sea una línea
                st.line_chart(weights_chart_data.T)
            except:
                st.write("No se pudo mostrar el gráfico de evolución")
        
        # Recomendación práctica
        st.write("**💡 Para invertir $1,000:**")
        investment_breakdown = ""
        for asset, weight in avg_weights.items():
            amount = weight * 1000
            if amount >= 10:  # Solo mostrar inversiones significativas
                investment_breakdown += f"- **{asset}**: ${amount:.0f} ({weight*100:.1f}%)\n"
        
        if investment_breakdown:
            st.markdown(investment_breakdown)
        else:
            st.write("Pesos muy pequeños para inversión de $1,000")
    
    if not portfolio_returns:
        st.error("No se pudieron calcular rendimientos del backtest")
        return
    
    # Convertir a series
    portfolio_returns = pd.Series(portfolio_returns)
    benchmark_returns = pd.Series(benchmark_returns)
    
    # 5. RESULTADOS
    st.subheader("📊 Resultados")
    
    port_metrics = calculate_portfolio_metrics_safe(portfolio_returns)
    bench_metrics = calculate_portfolio_metrics_safe(benchmark_returns)
    
    if port_metrics and bench_metrics:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🤖 RF + Markowitz")
            for key, value in port_metrics.items():
                if isinstance(value, float):
                    st.metric(key, f"{value:.2%}" if 'Rate' in key or 'Return' in key or 'Volatility' in key or 'Drawdown' in key else f"{value:.3f}")
        
        with col2:
            st.markdown("### 📈 Benchmark")
            for key, value in bench_metrics.items():
                if isinstance(value, float):
                    st.metric(key, f"{value:.2%}" if 'Rate' in key or 'Return' in key or 'Volatility' in key or 'Drawdown' in key else f"{value:.3f}")
        
        # Gráficos básicos
        st.subheader("📈 Performance")
        
        # Verificar que tenemos datos válidos
        if len(portfolio_returns) == 0 or len(benchmark_returns) == 0:
            st.error("No hay datos suficientes para el gráfico")
            return
        
        # Performance acumulada con datos válidos
        try:
            # Asegurar que ambas series tengan la misma longitud
            min_length = min(len(portfolio_returns), len(benchmark_returns))
            port_returns_clean = portfolio_returns[:min_length]
            bench_returns_clean = benchmark_returns[:min_length]
            
            # Calcular performance acumulada
            port_cum = (1 + port_returns_clean).cumprod()
            bench_cum = (1 + bench_returns_clean).cumprod()
            
            # Verificar que los valores no sean constantes
            if port_cum.std() > 0.001 and bench_cum.std() > 0.001:
                chart_data = pd.DataFrame({
                    'RF_Markowitz': port_cum.values,
                    'Benchmark': bench_cum.values
                }, index=range(len(port_cum)))
                
                st.line_chart(chart_data)
                
                # Mostrar valores finales
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Valor Final RF+Markowitz", f"${port_cum.iloc[-1]:.3f}")
                with col2:
                    st.metric("Valor Final Benchmark", f"${bench_cum.iloc[-1]:.3f}")
            else:
                st.warning("Los datos de rendimiento parecen ser constantes")
                st.write(f"RF+Markowitz rendimiento promedio: {port_returns_clean.mean():.4f}")
                st.write(f"Benchmark rendimiento promedio: {bench_returns_clean.mean():.4f}")
            
        except Exception as e:
            # Fallback detallado
            st.warning(f"Error creando gráfico: {str(e)}")
            
            # Mostrar estadísticas básicas en su lugar
            col1, col2 = st.columns(2)
            with col1:
                st.write("**RF + Markowitz:**")
                st.write(f"Rendimientos: {len(portfolio_returns)} observaciones")
                st.write(f"Promedio: {portfolio_returns.mean():.4f}")
                st.write(f"Desv. Estándar: {portfolio_returns.std():.4f}")
                
            with col2:
                st.write("**Benchmark:**")
                st.write(f"Rendimientos: {len(benchmark_returns)} observaciones")
                st.write(f"Promedio: {benchmark_returns.mean():.4f}")
                st.write(f"Desv. Estándar: {benchmark_returns.std():.4f}")
        
        # Métricas de valor añadido
        excess_return = port_metrics.get('Total Return', 0) - bench_metrics.get('Total Return', 0)
        outperform_rate = (portfolio_returns > benchmark_returns).mean()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Exceso de Retorno", f"{excess_return:.2%}")
        with col2:
            st.metric("% Períodos Ganadores", f"{outperform_rate:.1%}")
        
        # Conclusión
        if excess_return > 0:
            st.success("🏆 Estrategia superó al benchmark")
        else:
            st.warning("⚠️ Estrategia no superó al benchmark en este período")
    
    else:
        st.error("Error calculando métricas finales")

if __name__ == "__main__":
    main()
