import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

# Configuración de página de Streamlit con tema oscuro
st.set_page_config(page_title="Crypto DCA Quant Dashboard",
                   layout="wide", initial_sidebar_state="expanded")

# Estilizar con CSS para asemejar la estética oscura y técnica de TradingView
st.markdown("""
    <style>
    .main { background-color: #131722; color: #d1d4dc; }
    .stMetric { background-color: #1c2030; padding: 15px; border-radius: 8px; border: 1px solid #2a2e39; }
    div[data-testid="stMetricValue"] { color: #2962ff; font-family: 'Courier New', monospace; }
    div[data-testid="stSidebar"] { background-color: #1c2030; border-right: 1px solid #2a2e39; }
    h1, h2, h3 { color: #ffffff; font-family: 'Trebuchet MS', sans-serif; }
    </style>
""", unsafe_allow_html=True)

# --- 1. LÓGICA CORE DE DATOS Y BACKTESTING ---


@st.cache_data(show_spinner=False)
def load_and_simulate(ticker, start_date, investment, fee_rate):
    # Descarga desde Yahoo Finance
    df_raw = yf.Ticker(ticker).history(start=start_date, interval="1d")
    if df_raw.empty:
        return pd.DataFrame(), pd.DataFrame()

    df = df_raw.reset_index()
    df['date'] = df['Date'].dt.strftime('%Y-%m-%d')
    df = df.rename(columns={'Open': 'open', 'Close': 'close'}).sort_values(
        'date').reset_index(drop=True)

    # Simulación del DCA
    trades = []
    usd_acumulado = 0.0
    btc_acumulado = 0.0
    market_data = dict(zip(df['date'], df['open']))
    available_dates = set(df['date'].tolist())

    ideal_dates = pd.date_range(start=df['date'].min(
    ), end=df['date'].max(), freq='D').strftime('%Y-%m-%d').tolist()

    for date_str in ideal_dates:
        dt = datetime.strptime(date_str, '%Y-%m-%d')
        if dt.day in [1, 15]:
            planned_date = date_str
            execution_date = planned_date if planned_date in available_dates else None

            if not execution_date:
                for future_days in range(1, 10):
                    check_date = (
                        dt + pd.Timedelta(days=future_days)).strftime('%Y-%m-%d')
                    if check_date in available_dates:
                        execution_date = check_date
                        break

            if execution_date:
                price = float(market_data[execution_date])
                comision = investment * fee_rate
                net_investment = investment - comision
                btc_comprado = net_investment / price

                usd_acumulado += investment
                btc_acumulado += btc_comprado
                equity_actual = btc_acumulado * price
                dd = ((equity_actual - usd_acumulado) / usd_acumulado) * \
                    100 if usd_acumulado > 0 else 0

                trades.append({
                    'planned_date': planned_date,
                    'execution_date': execution_date,
                    'monto_usd': investment,
                    'comision_usd': comision,
                    'precio_btc': price,
                    'btc_comprado': btc_comprado,
                    'usd_acumulado': usd_acumulado,
                    'btc_acumulado': btc_acumulado,
                    'equity_usd': equity_actual,
                    'drawdown_pct': dd
                })

    trades_df = pd.DataFrame(trades)

    # Generación de la curva diaria
    curve_data = []
    trades_dict = trades_df.set_index('execution_date').to_dict(
        'index') if not trades_df.empty else {}
    current_usd_invested = 0.0
    current_btc_balance = 0.0

    for _, row in df.iterrows():
        current_date = row['date']
        price = row['close']

        if current_date in trades_dict:
            current_usd_invested = trades_dict[current_date]['usd_acumulado']
            current_btc_balance = trades_dict[current_date]['btc_acumulado']

        equity_usd = current_btc_balance * \
            price if current_btc_balance > 0 else current_usd_invested
        curve_data.append({
            'date': current_date,
            'precio_btc': price,
            'usd_invertido': current_usd_invested,
            'equity_usd': equity_usd
        })

    curve_df = pd.DataFrame(curve_data)
    return trades_df, curve_df

# --- 2. ENTORNO VISUAL Y CONTROLES (SIDEBAR) ---


st.sidebar.title("🤖 Quant DCA Engine")
st.sidebar.markdown("Configura los parámetros del algoritmo:")

asset_choice = st.sidebar.selectbox(
    "Activo a evaluar", ["BTC-USD", "ETH-USD"], index=0)
fecha_inicio = st.sidebar.date_input("Fecha de Inicio", datetime.strptime(
    "2021-07-01", "%Y-%m-%d")).strftime('%Y-%m-%d')
monto_dca = st.sidebar.number_input(
    "Monto por aporte ($)", min_value=10.0, value=2000.0, step=50.0)
fee_pct = st.sidebar.slider("Comisión del Exchange (%)",
                            min_value=0.0, max_value=1.0, value=0.1, step=0.05) / 100

with st.spinner("Procesando datos históricos de mercado..."):
    trades, curve = load_and_simulate(
        asset_choice, fecha_inicio, monto_dca, fee_pct)

if curve.empty or trades.empty:
    st.error("No hay suficientes datos disponibles para los parámetros seleccionados.")
else:
    # --- 3. CÁLCULO DE MÉTRICAS QUANT AVANZADAS ---
    final_invested = curve['usd_invertido'].iloc[-1]
    final_equity = curve['equity_usd'].iloc[-1]
    retorno_total = ((final_equity - final_invested) / final_invested) * 100

    # Calcular Rendimiento Diario para ratios de riesgo
    curve['daily_return'] = curve['equity_usd'].pct_change().fillna(0)

    # Sharpe Ratio (Asumiendo Risk-Free Rate = 0 para simplificar cripto)
    avg_return = curve['daily_return'].mean()
    std_return = curve['daily_return'].std()
    sharpe_ratio = (avg_return / std_return) * \
        np.sqrt(365) if std_return != 0 else 0

    # Drawdown Histórico Real
    curve['peak'] = curve['equity_usd'].cummax()
    curve['drawdown'] = (
        (curve['equity_usd'] - curve['peak']) / curve['peak']) * 100
    max_drawdown = curve['drawdown'].min()

    # CAGR (Tasa de Crecimiento Anual Compuesto)
    dias_totales = (pd.to_datetime(
        curve['date'].iloc[-1]) - pd.to_datetime(curve['date'].iloc[0])).days
    anios = dias_totales / 365.25
    cagr = (((final_equity / final_invested) ** (1 / anios)) - 1) * \
        100 if final_invested > 0 and anios > 0 else 0

    # --- 4. RENDERIZADO DE LA INTERFAZ ---

    st.title(f"📊 Dashboard Técnico DCA: {asset_choice}")
    st.markdown("Estadísticas de rendimiento avanzado e histórico de equidad.")

    # Fila de Métricas Principales (KPIs)
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Invertido", f"${final_invested:,.2f}")
    col2.metric("Valor Portafolio", f"${final_equity:,.2f}")
    col3.metric("Retorno Total", f"{retorno_total:+.2f}%")
    col4.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
    col5.metric("Max Drawdown", f"{max_drawdown:.2f}%")

    # Pestañas de Visualización (Dashboard General / Desglose por Operación)
    tab1, tab2 = st.tabs(["📈 Análisis de Curvas y Riesgo",
                         "🔍 Auditoría de Inversiones Individuales"])

    with tab1:
        st.subheader("Evolución del Portafolio vs Aportes Netos")

        # Gráfico Principal Interactivo
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=curve['date'], y=curve['usd_invertido'],
                      name="Capital Aportado", line=dict(color='#ff9800', width=2)))
        fig.add_trace(go.Scatter(x=curve['date'], y=curve['equity_usd'],
                      name="Valor del Portafolio (Equity)", line=dict(color='#2962ff', width=2.5)))

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor='#131722', plot_bgcolor='#131722',
            xaxis=dict(gridcolor='#2a2e39', title="Fecha"),
            yaxis=dict(gridcolor='#2a2e39', title="USD ($)"),
            hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)

        # Fila Inferior: Gráfico de Drawdown y Datos de Mercado
        c_left, c_right = st.columns(2)

        with c_left:
            st.subheader("Zona de Estrés: Histórico de Drawdown (%)")
            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(x=curve['date'], y=curve['drawdown'],
                             name="Drawdown", fill='tozeroy', line=dict(color='#ef5350', width=1.5)))
            fig_dd.update_layout(
                template="plotly_dark", paper_bgcolor='#131722', plot_bgcolor='#131722',
                xaxis=dict(gridcolor='#2a2e39'), yaxis=dict(gridcolor='#2a2e39', title="Porcentaje (%)"),
                margin=dict(t=20, b=20)
            )
            st.plotly_chart(fig_dd, use_container_width=True)

        with c_right:
            st.subheader("Métricas de Consistencia Quant")
            st.markdown(f"""
            - **CAGR (Rendimiento Anual Compuesto):** `{cagr:.2f}%`
            - **Precio Inicial de Compra:** `${trades['precio_btc'].iloc[0]:,.2f}`
            - **Precio Actual de Mercado:** `${curve['precio_btc'].iloc[-1]:,.2f}`
            - **Precio Promedio Ponderado de Compra:** `${final_invested / trades['btc_comprado'].sum():,.2f}`
            - **Total de Activos Acumulados:** `{trades['btc_comprado'].sum():.4f} {asset_choice.split("-")[0]}`
            """)

    with tab2:
        st.subheader("Desglose Individual de Compras Registradas")
        st.markdown(
            "Tabla analítica completa de las operaciones quincenales calculadas por el algoritmo:")

        # Formatear el DataFrame para visualización amigable
        display_trades = trades.copy()
        display_trades.columns = ['Fecha Planeada', 'Fecha Ejecución', 'Inversión USD', 'Comisión USD', 'Precio Entrada',
                                  'Activo Comprado', 'Inversión Acumulada', 'Balance Acumulado', 'Equity Actual', 'Drawdown Temp (%)']

        st.dataframe(
            display_trades.style.format({
                'Inversión USD': '${:,.2f}', 'Comisión USD': '${:,.2f}', 'Precio Entrada': '${:,.2f}',
                'Activo Comprado': '{:.6f}', 'Inversión Acumulada': '${:,.2f}', 'Balance Acumulado': '{:.6f}',
                'Equity Actual': '${:,.2f}', 'Drawdown Temp (%)': '{:.2f}%'
            }),
            use_container_width=True
        )
