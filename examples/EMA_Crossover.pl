{=============================================================================
 Simple EMA Crossover Strategy
 PowerLanguage example for the EasyBacktest engine

 Entries:
   Long on fast EMA crossing above slow EMA.
   Short on fast EMA crossing below slow EMA, when enabled.
   Positions reverse or exit at the next bar's market price.

 Suggested chart: 5-minute OHLCV bars. The script also works on other
 timeframes supported by the backtest engine.
=============================================================================}
// ENGINE_BAR_TIMEFRAME=5min
// ENGINE_SESSION_TIMEZONE=America/New_York

Inputs:
    FastLength(9),
    SlowLength(21),
    NumContracts(1),
    EnableLongs(true),
    EnableShorts(true);

Variables:
    FastEMA(0),
    SlowEMA(0);

FastEMA = XAverage(Close, FastLength);
SlowEMA = XAverage(Close, SlowLength);

If FastEMA Crosses Over SlowEMA Then Begin
    If EnableLongs And MarketPosition <= 0 Then
        Buy ("EMA-Long") NumContracts contracts Next Bar At Market
    Else If MarketPosition = -1 Then
        BuyToCover ("EMA-Short-Exit") Next Bar At Market;
End;

If FastEMA Crosses Under SlowEMA Then Begin
    If EnableShorts And MarketPosition >= 0 Then
        SellShort ("EMA-Short") NumContracts contracts Next Bar At Market
    Else If MarketPosition = 1 Then
        Sell ("EMA-Long-Exit") Next Bar At Market;
End;
