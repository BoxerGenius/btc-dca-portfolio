"""
BTC DCA (Dollar Cost Average) Portfolio - Repo 1
Estrategia: Invertir cantidad fija cada día/semana en Bitcoin.
Objetivo: Mostrar código transparente + métricas de rendimiento.
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

# ============================
# 1. DESCARGAR DATOS HISTÓRICOS DE BTC (Binance API)
# ============================


def get_btc_historical_data():
    """
    Descarga datos históricos de BTCUSDT (diario) desde Binance API.
    No requiere API key para datos públicos.
    """
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": "BTCUSDT",
        "interval": "1d",           # 1 día
        "limit": 1000                # últimos 1000 días
    }

    response = requests.get(url, params=params)
    data = response.json()

    # Convertir a DataFrame
    df = pd.DataFrame(data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])

    # Convertir tipos
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['close'] = df['close'].astype(float)
    df['open'] = df['open'].astype(float)

    # Solo nos interesa la fecha y el precio de cierre
    df = df[['timestamp', 'close']].copy()
    df = df.sort_values('timestamp').reset_index(drop=True)

    return df

# ============================
# 2. SIMULAR DCA
# ============================


def simulate_dca(df, investment_per_period=100, period_days=7):
    """
    Simula DCA: invertir una cantidad fija cada X días.

    Args:
        df: DataFrame con fecha y precio de cierre.
        investment_per_period: Cantidad en USDT a invertir cada periodo.
        period_days: Número de días entre cada inversión (7 = semanal, 1 = diario).

    Returns:
        df_dca: DataFrame con el historial de DCA.
    """
    df_dca = df.copy()

    # Inicializar variables
    total_invested = 0
    btc_held = 0
    equity = []

    # Iterar por cada día
    for i in range(len(df_dca)):
        price = df_dca['close'][i]
        date = df_dca['timestamp'][i]

        # ¿Es día de invertir?
        if i % period_days == 0:
            # Invertir cantidad fija
            btc_bought = investment_per_period / price
            btc_held += btc_bought
            total_invested += investment_per_period

        # Calcular equity (valor actual de la posición)
        current_equity = btc_held * price
        equity.append(current_equity)

    df_dca['equity'] = equity
    df_dca['total_invested'] = total_invested
    df_dca['btc_held'] = btc_held

    return df_dca

# ============================
# 3. CALCULAR MÉTRICAS
# ============================


def calculate_metrics(df_dca):
    """
    Calcula métricas clave: PnL, Sharpe, drawdown, CAGR.
    """
    equity = df_dca['equity'].values
    invested = df_dca['total_invested'].iloc[-1]

    # PnL total
    pnl_total = equity[-1] - invested
    pnl_percent = (pnl_total / invested) * 100

    # Retornos diarios
    returns = np.log(equity / np.roll(equity, 1))[1:]
    returns = np.where(returns == 0, 0, returns)  # Eliminar ceros

    # Sharpe ratio (anualizado, sin considerar tasa libre de riesgo)
    sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(365)

    # Drawdown
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak
    max_drawdown = np.min(drawdown) * 100

    # CAGR (Compound Annual Growth Rate)
    days = len(df_dca)
    years = days / 365
    cagr = ((equity[-1] / invested) ** (1 / years)) - 1

    metrics = {
        'PnL Total (USDT)': pnl_total,
        'PnL Percent (%)': pnl_percent,
        'Sharpe Ratio': sharpe,
        'Máximo Drawdown (%)': max_drawdown,
        'CAGR (%)': cagr * 100,
        'BTC Held': df_dca['btc_held'].iloc[-1],
        'Total Invested (USDT)': invested,
        'Final Equity (USDT)': equity[-1]
    }

    return metrics

# ============================
# 4. GENERAR GRÁFICOS
# ============================


def plot_equity(df_dca, metrics):
    """
    Genera gráfico de equity curve + inversión total.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(df_dca['timestamp'], df_dca['equity'],
             label='Equity', color='blue', linewidth=2)
    plt.plot(df_dca['timestamp'], df_dca['total_invested'],
             label='Total Invested', color='red', linestyle='--', linewidth=2)

    plt.title('BTC DCA - Equity Curve', fontsize=16)
    plt.xlabel('Fecha')
    plt.ylabel('USDT')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Añadir métricas en el gráfico
    metrics_text = f"""
    PnL Total: {metrics['PnL Total (USDT)']:.2f} USDT
    PnL %: {metrics['PnL Percent (%)']:.2f}%
    Sharpe: {metrics['Sharpe Ratio']:.2f}
    Max Drawdown: {metrics['Máximo Drawdown (%)']:.2f}%
    CAGR: {metrics['CAGR (%)']:.2f}%
    """
    plt.text(0.02, 0.98, metrics_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig('equity_curve.png', dpi=300)
    plt.show()

# ============================
# 5. EJECUTAR TODO
# ============================


if __name__ == "__main__":
    print("=== BTC DCA PORTFOLIO - REPO 1 ===")
    print("1. Descargando datos históricos de BTC...")
    df = get_btc_historical_data()
    print(
        f"   ✓ Datos descargados: {len(df)} días desde {df['timestamp'].min()} hasta {df['timestamp'].max()}")

    print("2. Simulando DCA semanal ($100 USDT cada 7 días)...")
    df_dca = simulate_dca(df, investment_per_period=100, period_days=7)

    print("3. Calculando métricas...")
    metrics = calculate_metrics(df_dca)

    print("4. Métricas resultantes:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"   - {key}: {value:.2f}")
        else:
            print(f"   - {key}: {value}")

    print("5. Generando gráfico de equity curve...")
    plot_equity(df_dca, metrics)

    print("\n✅ Proceso completado!")
    print("   - Gráfico: equity_curve.png")
    print("   - Datos: df_dca (puedes exportar a CSV si quieres)")
