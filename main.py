import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import yfinance as yf

START_DATE = '2021-07-01'
INVESTMENT = 2000.0
FEE_RATE = 0.001  # 0.1% comisión simulada (Maker/Taker promedio)
TICKER_BTC = 'BTC-USD'


def fetch_klines(ticker=TICKER_BTC, start_date=START_DATE):
    """
    Descarga datos históricos diarios desde Yahoo Finance de forma directa.
    Ideal para saltar restricciones geográficas y obtener históricos profundos sin paginación manual.
    """
    print(
        f"Descargando datos históricos para {ticker} desde {start_date} vía Yahoo Finance...")

    try:
        # Descarga el histórico diario directo
        ticker_obj = yf.Ticker(ticker)
        df_raw = ticker_obj.history(start=start_date, interval="1d")
    except Exception as e:
        raise RuntimeError(f"Error crítico al conectar con Yahoo Finance: {e}")

    if df_raw.empty:
        raise ValueError(
            "No se obtuvieron datos. Verifica que el rango de fechas o el ticker sean correctos.")

    # Resetear el índice para tratar la fecha como una columna estándar
    df = df_raw.reset_index()

    # Formatear la fecha a string YYYY-MM-DD
    df['date'] = df['Date'].dt.strftime('%Y-%m-%d')

    # Renombrar columnas para mantener compatibilidad con tu estructura original
    df = df.rename(columns={'Open': 'open', 'Close': 'close'})

    # Asegurar orden cronológico y limpieza
    df = df.sort_values('date').reset_index(drop=True)

    print(
        f"Descarga completada. Registros obtenidos: {len(df)}. Rango real: {df['date'].min()} a {df['date'].max()}")
    return df[['date', 'open', 'close']]


def simulate_dca(df, investment=INVESTMENT, fee_rate=FEE_RATE):
    """
    Simula compras programadas quincenales (días 01 y 15 de cada mes).
    Alinea las compras estrictamente con las fechas cronológicas reales del mercado.
    """
    trades = []
    usd_acumulado = 0.0
    btc_acumulado = 0.0

    # Diccionario rápido de búsqueda {fecha: precio_apertura}
    market_data = dict(zip(df['date'], df['open']))
    available_dates = set(df['date'].tolist())

    # Generar rango de días ideales desde el inicio real hasta el final
    min_date = df['date'].min()
    max_date = df['date'].max()
    ideal_dates = pd.date_range(
        start=min_date, end=max_date, freq='D').strftime('%Y-%m-%d').tolist()

    for date_str in ideal_dates:
        dt = datetime.strptime(date_str, '%Y-%m-%d')

        # Ejecutar estrategia DCA: Días 1 y 15 de cada mes
        if dt.day in [1, 15]:
            planned_date = date_str
            execution_date = None

            # Buscar fecha de ejecución real (si cae fin de semana en mercados tradicionales,
            # aunque en cripto hay velas diarias siempre en Yahoo Finance)
            if planned_date in available_dates:
                execution_date = planned_date
            else:
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
                drawdown = ((equity_actual - usd_acumulado) /
                            usd_acumulado) * 100 if usd_acumulado > 0 else 0

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
                    'drawdown_pct': drawdown
                })

    return pd.DataFrame(trades)


def make_end_equity_series(df, trades_df):
    """
    Construye la evolución diaria del portafolio.
    Mapea de forma realista cómo crece el capital invertido y varía la equidad con el cierre diario.
    """
    curve_data = []
    trades_dict = trades_df.set_index('execution_date').to_dict('index')

    current_usd_invested = 0.0
    current_btc_balance = 0.0

    for _, row in df.iterrows():
        current_date = row['date']
        price = row['close']

        # Si hubo una ejecución de compra este día, actualizamos los balances acumulados
        if current_date in trades_dict:
            current_usd_invested = trades_dict[current_date]['usd_acumulado']
            current_btc_balance = trades_dict[current_date]['btc_acumulado']

        # El valor del portafolio diario es tu balance de BTC multiplicado por el precio de cierre de ese día
        equity_usd = current_btc_balance * \
            price if current_btc_balance > 0 else current_usd_invested

        curve_data.append({
            'date': current_date,
            'precio_btc': price,
            'usd_invertido': current_usd_invested,
            'equity_usd': equity_usd
        })

    return pd.DataFrame(curve_data)


def plot_curve(curve, out='equity_curve.png'):
    """
    Genera el gráfico comparativo del rendimiento del DCA frente al capital neto aportado.
    """
    plt.figure(figsize=(14, 7))
    dates = pd.to_datetime(curve['date'])

    plt.plot(dates, curve['usd_invertido'],
             label='Capital Invertido Acumulado (Aportes)', color='orange', linewidth=2)
    plt.plot(dates, curve['equity_usd'],
             label='Valor Real del Portafolio BTC (Equity)', color='blue', linewidth=2)

    plt.title(
        'Backtesting BTC DCA - Julio 2021 a Presente (Yahoo Finance Data)', fontsize=14)
    plt.xlabel('Fecha')
    plt.ylabel('Monto en USD')
    plt.grid(True, alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()


def main():
    try:
        df = fetch_klines()
        trades = simulate_dca(df)

        if trades.empty:
            print("No se generaron trades. Revisa el rango de datos.")
            return

        curve = make_end_equity_series(df, trades)

        # Guardar reportes locales en CSV y exportar la gráfica limpia
        trades.to_csv('dca_trades.csv', index=False)
        curve.to_csv('equity_curve.csv', index=False)
        plot_curve(curve)

        print("\n=== PRIMEROS TRADES REGISTRADOS (2021) ===")
        print(trades[['planned_date', 'execution_date', 'precio_btc',
              'usd_acumulado', 'btc_acumulado']].head(5).to_string(index=False))

        print("\n=== ÚLTIMOS TRADES REGISTRADOS ===")
        print(trades[['planned_date', 'execution_date', 'precio_btc',
              'usd_acumulado', 'btc_acumulado']].tail(5).to_string(index=False))

        final_invested = curve['usd_invertido'].iloc[-1]
        final_equity = curve['equity_usd'].iloc[-1]
        rendimiento = ((final_equity - final_invested) /
                       final_invested) * 100 if final_invested > 0 else 0

        print("\n=== RESUMEN DE BACKTESTING DE DCA ===")
        print(f"Fecha de inicio real: {curve['date'].min()}")
        print(f"Fecha de fin:         {curve['date'].iloc[-1]}")
        print(f"Total USD Invertido:  ${final_invested:,.2f}")
        print(f"Valor Final Portafolio: ${final_equity:,.2f}")
        print(f"Rendimiento Total:    {rendimiento:.2f}%")

    except Exception as e:
        print(f"Ocurrió un error en la ejecución: {e}")


if __name__ == '__main__':
    main()
