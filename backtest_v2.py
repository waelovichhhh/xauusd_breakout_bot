"""
Backtest v2 - Stratégie "CASSURE D'UN NIVEAU MAJEUR" (XAUUSD, et autres marchés avec valider_marches.py)
=========================================================================================================

POURQUOI CE FICHIER (octobre 2026)
----------------------------------
L'ancien backtest (backtest_xauusd_breakout.py) "trichait" sans le vouloir : par endroits, il se
servait de prix pas encore connus au moment de décider (la bougie d'entrée comptée en entier, une
clôture M15 utilisée avant la fin de la bougie, le choix du trade à garder fait en connaissant la
suite). Rejouée honnêtement (uniquement avec le passé, comme le bot en réel), l'ancienne stratégie
cassure + retest de zone n'a aucun avantage sur 5 ans. Ce fichier teste la stratégie de
remplacement, la seule qui a tenu sur 2021-2026.

LA RÈGLE (volontairement simple, 3 réglages seulement)
------------------------------------------------------
1. Niveau majeur = le plus haut et le plus bas des N_DAYS derniers jours de cotation
   (N_DAYS x 276 bougies M5), bougie en cours exclue.
2. Signal = une bougie M5 CLÔTURE au-dessus du plus haut (-> achat) ou sous le plus bas (-> vente).
   Au maximum un signal par sens toutes les 276 bougies (environ un jour).
3. Entrée au marché juste après la clôture. PAS d'attente de retest (testé : le retest détruit l'avantage).
4. Stop = SL_SIGMA x "sigma 1h" depuis le prix d'entrée, où sigma 1h = volatilité récente
   (écart-type des variations M5 sur 5 jours, multiplié par racine de 12).
5. Pas de TP : le trade est fermé au bout de HOLD_MINUTES si le stop n'a pas été touché.
6. Un seul trade à la fois.

AUCUNE TRICHE POSSIBLE : chaque décision à la bougie t n'utilise que des bougies <= t.

USAGE
-----
    python backtest_v2.py --m5 XAUUSD_M5.csv
    python backtest_v2.py --m5 XAUUSD_M5.csv --m1 XAUUSD_M1.csv      (plus précis)

Les signaux viennent toujours des bougies M5. Ce qui change entre les deux modes, c'est la simulation du trade :
- avec le M5 seul : entrée à l'ouverture de la bougie M5 qui commence 5 minutes après la clôture du signal,
  stop et sortie vérifiés bougie M5 par bougie M5 ;
- avec le M1 en plus : entrée 1 minute après la clôture du signal, stop et sortie vérifiés minute par minute.
  Si le fichier M1 ne couvre pas un trade (historique M1 plus court), ce trade est simulé sur les bougies M5
  et le programme le signale.
Le bot en réel entre plus tôt : dès que la bougie suivante apparaît, 1 à 2 secondes après la clôture. Rejoué ainsi
sur 2020-2026, le résultat est un peu meilleur (+0.16 R par trade au lieu de +0.12), mais l'écart tient à une
poignée de trades : ne compte pas dessus.

Les CSV sont ceux exportés par MT5 (Symboles > Barres > Exporter), avec assez d'historique :
il faut au moins N_DAYS + 5 jours de bougies avant le premier trade possible.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# RÉGLAGES DE LA STRATÉGIE (figés AVANT la validation sur 2025-2026 ; ne pas optimiser dessus)
# ----------------------------------------------------------------------
N_DAYS = 10              # profondeur du niveau majeur, en jours de cotation
BARS_PER_DAY = 276       # nombre de bougies M5 dans une journée de cotation de l'or (23 h)
VOL_BARS = 1380          # fenêtre de volatilité : 5 jours de bougies M5
SL_SIGMA = 1.5           # distance du stop, en multiples de la volatilité sur 1 heure
HOLD_MINUTES = 120       # durée maximale d'un trade (sortie au marché ensuite)
SIGNAL_GAP_BARS = 276    # un seul signal par sens toutes les 276 bougies

# ----------------------------------------------------------------------
# HYPOTHÈSES DE COÛTS (prudentes ; spread mesuré sur le compte MetaQuotes-Demo : ~0.35 $ en sept. 2026)
# ----------------------------------------------------------------------
SPREAD_MIN_USD = 0.20          # spread minimum en dollars
SPREAD_BP = 8.5e-5             # spread proportionnel au prix (0.85 point de base) -> 0.35 $ à 4150 $
SLIPPAGE_ENTRY_USD = 0.02      # glissement à l'entrée (ordre au marché)
SLIPPAGE_STOP_USD = 0.05       # glissement sur un stop
SLIPPAGE_TIMEOUT_USD = 0.02    # glissement sur la sortie au temps (ordre au marché)
ENTRY_DELAY_SECONDS = 60       # délai entre la clôture de la bougie signal et l'exécution réelle
MAX_WAIT_SECONDS = 300         # si le marché est fermé plus longtemps que ça après le signal : pas de trade


@dataclass(frozen=True)
class Costs:
    """
    Coûts de trading d'un marché, en unités de prix : spread(prix) = max(spread_min, spread_bp x prix),
    plus des glissements fixes à l'entrée, sur un stop et sur la sortie au temps.
    Par défaut : les coûts de l'or ci-dessus. valider_marches.py en construit un pour chaque autre marché,
    à partir des spreads réels du broker.
    """
    spread_min: float = SPREAD_MIN_USD
    spread_bp: float = SPREAD_BP
    slip_entry: float = SLIPPAGE_ENTRY_USD
    slip_stop: float = SLIPPAGE_STOP_USD
    slip_timeout: float = SLIPPAGE_TIMEOUT_USD

    def spread(self, price: float) -> float:
        return max(self.spread_min, self.spread_bp * price)


GOLD_COSTS = Costs()


def load_mt5_csv(path: str) -> pd.DataFrame:
    """CSV exporté depuis MT5 (tabulations, colonnes <DATE> <TIME> <OPEN> <HIGH> <LOW> <CLOSE> ...)."""
    df = pd.read_csv(path, sep="\t")
    df.columns = [c.strip("<>").lower() for c in df.columns]
    df["time"] = pd.to_datetime(df["date"] + " " + df["time"], format="%Y.%m.%d %H:%M:%S")
    return df[["time", "open", "high", "low", "close"]].sort_values("time").reset_index(drop=True)


def add_indicators(df5: pd.DataFrame) -> pd.DataFrame:
    """Niveaux majeurs et volatilité, calculés uniquement avec le passé de chaque bougie."""
    df5 = df5.copy()
    lookback = N_DAYS * BARS_PER_DAY
    df5["level_high"] = df5["high"].rolling(lookback).max().shift(1)   # shift(1) : bougie courante exclue
    df5["level_low"] = df5["low"].rolling(lookback).min().shift(1)
    df5["sigma_1h"] = df5["close"].diff().fillna(0.0).rolling(VOL_BARS).std() * np.sqrt(12)
    return df5


def find_signals(df5: pd.DataFrame, last_up: int = -10**9, last_down: int = -10**9) -> np.ndarray:
    """
    +1 = achat, -1 = vente, 0 = rien, pour chaque bougie M5 (décision prise à SA clôture).
    last_up / last_down : position (dans df5) du dernier signal déjà pris dans chaque sens, si on le connaît
    (utilisé par le bot en réel pour reprendre exactement où il en était ; inutile pour un backtest complet).
    """
    close = df5["close"].values
    up = close > df5["level_high"].values       # NaN -> False tant que l'historique est insuffisant
    down = close < df5["level_low"].values
    signal = np.zeros(len(df5), dtype=np.int8)
    for t in np.nonzero(up | down)[0]:
        if up[t] and t - last_up >= SIGNAL_GAP_BARS:
            signal[t] = 1
            last_up = t
        elif down[t] and t - last_down >= SIGNAL_GAP_BARS:
            signal[t] = -1
            last_down = t
    return signal


def spread_at(price: float) -> float:
    """Spread de l'or (gardé pour compatibilité)."""
    return GOLD_COSTS.spread(price)


def _arrays(df: pd.DataFrame, step: int) -> tuple:
    """(heures en secondes, open, high, low, close, durée d'une bougie) sous forme de tableaux numpy."""
    times = df["time"].values.astype("datetime64[s]").astype(np.int64)
    return times, df["open"].values, df["high"].values, df["low"].values, df["close"].values, step


def _m1_covers(t1: np.ndarray, t5: np.ndarray, start: int, deadline: int) -> bool:
    """
    Le fichier M1 couvre-t-il tout le trade (de la clôture du signal jusqu'à la sortie au temps) ?
    Oui si chaque bougie M5 de cet intervalle contient au moins une bougie M1 : les deux viennent des mêmes
    ticks, donc une bougie M5 sans aucune M1 dedans = un trou dans le fichier M1.
    """
    a = int(np.searchsorted(t5, start, side="left"))
    b = int(np.searchsorted(t5, deadline, side="left"))       # 1re bougie M5 à l'échéance ou après : celle de la sortie
    bars = t5[a:b + 1]
    if len(t1) == 0 or len(bars) == 0:
        return False
    k = np.minimum(np.searchsorted(t1, bars, side="left"), len(t1) - 1)
    return bool(np.all((t1[k] >= bars) & (t1[k] < bars + 300)))


def run_backtest(df5: pd.DataFrame, df1: pd.DataFrame | None = None, costs: Costs = GOLD_COSTS,
                 one_at_a_time: bool = True) -> pd.DataFrame:
    """
    Rejoue la stratégie bougie par bougie. Les prix des CSV MT5 sont des prix BID :
    - un achat s'exécute au ASK (bid + spread) et se ferme au bid ;
    - une vente s'exécute au bid et se ferme au ASK (son stop est donc touché quand bid + spread l'atteint).

    Simulation d'un trade sur les bougies "fines" (M1 si df1 est fourni et couvre le trade, sinon M5) :
    - entrée à l'ouverture de la première bougie fine qui commence au moins ENTRY_DELAY_SECONDS après la
      clôture du signal (donc +1 min en M1, +5 min en M5) ;
    - sortie au temps à l'ouverture de la première bougie fine qui commence à l'échéance ou après ;
    - avant l'échéance, le stop est considéré touché dès que la bougie l'atteint (plus bas pour un achat,
      plus haut + spread pour une vente), au pire des deux prix entre le stop et l'ouverture de la bougie.
    Il n'y a pas de TP, donc aucune ambiguïté "stop ou TP en premier" dans une même bougie.
    La colonne "simulation" du résultat indique, pour chaque trade, les bougies utilisées (M1 ou M5).

    costs : coûts du marché (l'or par défaut).
    one_at_a_time=False : simule TOUS les signaux, même ceux qui tombent pendant un trade déjà ouvert
    (sert à simulate_portfolio, qui applique ensuite les règles du bot sur plusieurs marchés à la fois).
    """
    df5 = add_indicators(df5)
    signal = find_signals(df5)
    m5 = _arrays(df5, 300)
    m1 = _arrays(df1, 60) if df1 is not None and len(df1) else None
    t5 = m5[0]
    sigma = df5["sigma_1h"].values

    trades = []
    next_free = 0
    for t in np.nonzero(signal)[0]:
        direction = int(signal[t])
        signal_close = t5[t] + 300
        if (one_at_a_time and signal_close < next_free) or not np.isfinite(sigma[t]):
            continue                                         # déjà en position, ou pas assez d'historique
        deadline = signal_close + HOLD_MINUTES * 60
        use_m1 = m1 is not None and _m1_covers(m1[0], t5, signal_close, deadline)
        ft, fo, fh, fl, fc, step = m1 if use_m1 else m5
        m = int(np.searchsorted(ft, signal_close + ENTRY_DELAY_SECONDS, side="left"))
        if m >= len(ft):
            break
        if ft[m] - (signal_close + ENTRY_DELAY_SECONDS) > MAX_WAIT_SECONDS:
            continue                                         # marché fermé : l'ordre ne serait pas passé
        bid = fo[m]
        fill = bid + costs.spread(bid) + costs.slip_entry if direction == 1 else bid - costs.slip_entry
        risk = SL_SIGMA * sigma[t]
        stop = fill - risk if direction == 1 else fill + risk
        outcome, exit_price, exit_time = None, None, None
        for k in range(m, len(ft)):
            if ft[k] >= deadline:                            # sortie au temps, au marché
                exit_price = fo[k] - costs.slip_timeout if direction == 1 else fo[k] + costs.spread(fo[k]) + costs.slip_timeout
                outcome, exit_time = "temps", ft[k]
                break
            if direction == 1:
                if fl[k] <= stop:                            # stop touché (ou sauté par un trou de cotation)
                    exit_price = min(fo[k], stop) - costs.slip_stop
                    outcome, exit_time = "stop", ft[k] + step
                    break
            else:
                sp = costs.spread(fc[k])
                if fh[k] + sp >= stop:
                    exit_price = max(fo[k] + sp, stop) + costs.slip_stop
                    outcome, exit_time = "stop", ft[k] + step
                    break
        if outcome is None:
            break                                            # fin des données avant la sortie
        pnl = (exit_price - fill) if direction == 1 else (fill - exit_price)
        trades.append({
            "signal_time": pd.Timestamp(signal_close, unit="s"), "direction": "achat" if direction == 1 else "vente",
            "entry_time": pd.Timestamp(ft[m], unit="s"), "entry_price": round(fill, 3), "stop": round(stop, 3),
            "risk_usd": round(risk, 3), "exit_time": pd.Timestamp(exit_time, unit="s"), "exit_price": round(exit_price, 3),
            "outcome": outcome, "pnl_usd_per_oz": round(pnl, 3), "R": pnl / risk, "simulation": "M1" if use_m1 else "M5",
        })
        next_free = exit_time
    return pd.DataFrame(trades)


def simulate_portfolio(candidates: dict, groups: dict, max_open: int, max_per_group: int) -> pd.DataFrame:
    """
    Rejoue plusieurs marchés ensemble, avec les règles du bot :
    - un seul trade à la fois par marché ;
    - au plus `max_open` positions ouvertes en tout, et `max_per_group` par famille de marchés corrélés.
    `candidates` : {marché: run_backtest(..., one_at_a_time=False)} ; `groups` : {marché: famille}.
    Les signaux sont traités dans l'ordre chronologique (à égalité : dans l'ordre de `candidates`), comme le bot.
    Tous les trades risquent la même fraction du capital : le résultat se lit en R.
    """
    events = []
    for rank, (market, trades) in enumerate(candidates.items()):
        for row in trades.itertuples(index=False):
            events.append((row.signal_time, rank, market, row))
    events.sort(key=lambda e: (e[0], e[1]))
    open_positions = []            # (heure de sortie, marché, famille)
    taken = []
    for signal_time, _, market, row in events:
        open_positions = [p for p in open_positions if p[0] > signal_time]
        group = groups.get(market, market)
        if any(p[1] == market for p in open_positions):
            continue                                   # ce marché a déjà son trade
        if len(open_positions) >= max_open or sum(p[2] == group for p in open_positions) >= max_per_group:
            continue                                   # plafond global ou plafond de la famille atteint
        open_positions.append((row.exit_time, market, group))
        taken.append(dict(row._asdict(), market=market, group=group))
    return pd.DataFrame(taken)


def print_stats(trades: pd.DataFrame) -> None:
    if trades.empty:
        print("Aucun trade (pas assez d'historique ?). Il faut au moins", N_DAYS + 5, "jours de bougies M5.")
        return

    def row(g: pd.DataFrame) -> dict:
        wins = g[g["R"] > 0]
        losses = g[g["R"] <= 0]
        pf = wins["pnl_usd_per_oz"].sum() / max(1e-9, -losses["pnl_usd_per_oz"].sum())
        return {"trades": len(g), "gagnants %": round(100 * len(wins) / len(g), 1), "R moyen": round(g["R"].mean(), 3),
                "R total": round(g["R"].sum(), 1), "facteur de profit": round(pf, 2)}

    table = pd.DataFrame({str(y): row(g) for y, g in trades.groupby(trades["signal_time"].dt.year)}).T
    table.loc["TOTAL"] = row(trades)
    print("\n===== STRATÉGIE v2 : CASSURE D'UN NIVEAU MAJEUR =====")
    print(f"Réglages : niveau {N_DAYS} jours | stop {SL_SIGMA} x sigma 1h | durée max {HOLD_MINUTES} min")
    print(table.to_string())
    r = trades["R"].values
    cumulative = np.concatenate([[0.0], np.cumsum(r)])       # départ à 0 : une série qui commence par des pertes compte aussi
    drawdown = (cumulative - np.maximum.accumulate(cumulative)).min()
    t_stat = r.mean() / (r.std(ddof=1) / np.sqrt(len(r))) if len(r) > 2 else float("nan")
    print(f"\nR = gain ou perte d'un trade, en multiples du risque pris (1 R = la perte si le stop est touché).")
    print(f"Pire baisse cumulée (depuis un plus haut) : {drawdown:.1f} R | t-stat du R moyen : {t_stat:.2f}")
    print("Repère : un t-stat au-dessus de 2 est rarement dû au seul hasard, mais ce n'est pas une preuve ;")
    print("         cette règle a été retenue après en avoir testé une douzaine, ce qui rend le chiffre un peu flatteur.")
    print("Exemple : en risquant 1 % du capital par trade, +10 R sur une année = environ +10 %.")
    if "simulation" in trades.columns:
        n_m1, n_m5 = int((trades["simulation"] == "M1").sum()), int((trades["simulation"] == "M5").sum())
        if n_m1 and n_m5:
            print(f"ATTENTION : le fichier M1 ne couvre pas toute la période. {n_m1} trades simulés à la minute, "
                  f"{n_m5} simulés sur les bougies M5 (entrée 5 min après le signal au lieu de 1 min).")
        elif n_m5:
            print("Simulation sur les bougies M5 (entrée à l'ouverture de la bougie qui commence 5 min après le signal).")
        else:
            print("Simulation à la minute (entrée 1 min après la clôture du signal).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--m5", required=True, help="CSV M5 exporté de MT5")
    parser.add_argument("--m1", default=None, help="CSV M1 exporté de MT5 (optionnel, sorties plus précises)")
    parser.add_argument("--out", default="trades_v2.csv")
    args = parser.parse_args()

    df5 = load_mt5_csv(args.m5)
    df1 = load_mt5_csv(args.m1) if args.m1 else None
    print(f"Données : {len(df5)} bougies M5, du {df5['time'].iloc[0]} au {df5['time'].iloc[-1]}"
          + (f" | {len(df1)} bougies M1" if df1 is not None else ""))
    result = run_backtest(df5, df1)
    if df1 is not None and (result.empty or not (result["simulation"] == "M1").any()):
        print("ATTENTION : le fichier M1 ne couvre aucun trade (périodes différentes ?) : tout est simulé sur les bougies M5.")
    print_stats(result)
    if not result.empty:
        result.to_csv(args.out, index=False)
        print(f"\nDétail des trades : {args.out}")
