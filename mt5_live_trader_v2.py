"""
Bot v2 pour MetaTrader 5 - Stratégie "CASSURE D'UN NIVEAU MAJEUR" (or, et autres marchés validés)
==================================================================================================

À LIRE AVANT DE LANCER
----------------------
1) Ce bot REMPLACE mt5_live_trader.py. Ne fais pas tourner les deux en même temps.
   L'ancienne stratégie (cassure + retest de zone) n'a aucun avantage une fois rejouée sans
   regarder le futur : -0.055 R par trade sur 5 269 trades (2021-2026).

2) La règle (détail dans backtest_v2.py, qui doit rester dans le MÊME dossier) :
   - niveau majeur = plus haut / plus bas des 10 derniers jours de cotation ;
   - si une bougie M5 CLÔTURE au-dessus du plus haut -> achat ; sous le plus bas -> vente ;
   - stop à 1.5 x la volatilité d'une heure, PAS de TP, sortie au bout de 120 minutes ;
   - un seul trade à la fois par marché, au plus un signal par sens et par jour.
   Sur l'or seul : environ UN trade par semaine.

3) Ce que le backtest honnête dit sur l'or (2020 -> sept. 2026, 435 trades) : +0.12 R par trade en
   moyenne, soit environ +8 R par an. Avantage MODESTE et pas certain. Compte démo uniquement.

4) PLUSIEURS MARCHÉS : lance d'abord  python valider_marches.py  (une fois). Il teste la règle, sans
   changer ses réglages, sur une quinzaine de marchés de ton broker et écrit marches_valides.json si elle
   tient. Au démarrage, le bot lit ce fichier : il trade alors les marchés validés, avec le risque par
   trade conseillé, au plus MAX_OPEN_POSITIONS positions en même temps et une seule par famille de marchés
   corrélés (or et argent = même pari). Sans ce fichier, il trade l'or seul.

5) GARDE-FOU D'USURE : le bot note le résultat (en R) de chacun de ses trades. S'il fait nettement pire que
   ce que le backtest juge possible (moins de 1 % de chances si la stratégie marchait encore), il arrête
   d'ouvrir des trades et l'écrit dans le log. Les positions déjà ouvertes restent gérées.

6) Sécurité : le bot REFUSE de trader sur un compte réel tant que ALLOW_REAL_ACCOUNT = False
   (vérifié au démarrage, puis en continu). Attention : un compte de prop firm (FundedNext...) est vu
   par MT5 comme un compte démo, ce garde-fou ne le bloque donc pas.

7) Le bot n'utilise PAS l'horloge du PC pour décider. Il regarde chaque seconde si MT5 affiche une
   nouvelle bougie M5 : la bougie précédente est alors terminée de façon certaine, et il la traite
   une seule fois. Laisse le PC allumé (pas de mise en veille), MT5 ouvert et "Algo Trading" activé.

INSTALLATION (PC Windows, MT5 ouvert et connecté) :  pip install MetaTrader5 pandas numpy
LANCEMENT :  python mt5_live_trader_v2.py
"""

from __future__ import annotations

import json
import logging
import sys
import time as time_module
from pathlib import Path

import numpy as np
import pandas as pd

import backtest_v2 as strat   # la règle de trading vit dans UN seul fichier, partagé avec le backtest

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None  # permet de lire/tester ce fichier hors Windows


# ----------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------
DRY_RUN = False                 # True = n'envoie aucun ordre, écrit seulement ce qu'il aurait fait
ALLOW_REAL_ACCOUNT = False      # tant que False : le bot ne trade QUE sur un compte démo
SYMBOL = "XAUUSD"               # nom de l'or chez ton broker ("GOLD", "XAUUSD.a"...) ; sert tant que
                                # marches_valides.json n'existe pas (ensuite, ce fichier donne les noms)
MAGIC_NUMBER = 20261003         # identifiant des trades de CE bot (différent de l'ancien bot)

ALLOW_LONGS = True              # achats sur cassure du plus haut de 10 jours
ALLOW_SHORTS = True             # ventes sur cassure du plus bas de 10 jours.
                                # Info : sur l'or 2020-2026, tout le gain est venu des ACHATS (+0.20 R par trade) ;
                                # les ventes ont fait zéro (-0.01 R). L'écart n'est pas assez net pour conclure,
                                # d'où les deux sens par défaut. Mets False si tu préfères n'avoir que les achats.

RISK_PERCENT_PER_TRADE = 1.0    # % du capital perdu si le stop est touché (or seul). Ne monte pas au-dessus de 1 %.
USE_RECOMMENDED_RISK = True     # si marches_valides.json existe : risque conseillé par valider_marches.py
DAILY_RISK_BUDGET_PERCENT = 3.0 # pas de nouvelle entrée si perte du jour + risque du nouveau trade dépasse ce %
MAX_OPEN_POSITIONS = 4          # (plusieurs marchés) positions ouvertes en même temps, tous marchés confondus
MAX_POSITIONS_PER_GROUP = 1     # (plusieurs marchés) une seule position à la fois par famille de marchés corrélés

EDGE_GUARD_ENABLED = True       # garde-fou d'usure (point 5 en haut du fichier)
EDGE_GUARD_Z = 2.5              # sévérité : pause si le résultat cumulé tombe 2.5 écarts-types sous ce que prévoit
                                # le backtest (fausse alerte : environ 5 % des cas sur plusieurs années)
EDGE_GUARD_MIN_TRADES = 20      # le garde-fou ne juge qu'à partir de 20 trades
GOLD_BACKTEST_MEAN_R = 0.122    # backtest de l'or 2020-2026 : résultat moyen par trade...
GOLD_BACKTEST_SD_R = 1.20       # ... et sa dispersion (remplacés par ceux de marches_valides.json s'il existe)

HISTORY_BARS = 8000             # bougies M5 chargées pour le calcul (10 j de niveau + 5 j de volatilité + marge)
POLL_SECONDS = 1.0              # le bot regarde chaque seconde si une nouvelle bougie M5 est apparue
MAX_SIGNAL_AGE_SECONDS = 150    # un signal vu plus de 150 s après la clôture de sa bougie est ignoré
MAX_SPREAD_TO_RISK = 0.25       # pas d'entrée si le spread dépasse 25 % de la distance du stop (annonce, marché fin)
ORDER_DEVIATION_POINTS = 30     # glissement toléré à l'envoi de l'ordre (en points)
ORDER_ATTEMPTS = 3              # essais d'entrée si le broker répond "prix modifié" (avec le prix du moment à chaque fois)
ERROR_PAUSE_SECONDS = 10        # pause après une erreur ou une liaison MT5 perdue, avant de réessayer
IDLE_NOTICE_SECONDS = 900       # sans nouvelle bougie depuis 15 min : le bot écrit qu'il attend (marché fermé ?)

HERE = Path(__file__).parent
LOG_FILE = HERE / "live_trader_v2.log"
STATE_FILE = HERE / "live_trader_v2_state.json"   # derniers signaux par marché + résultats des trades (garde-fou)
MARKETS_FILE = HERE / "marches_valides.json"      # écrit par valider_marches.py ; absent = or seul

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("mt5_live_trader_v2")

# Réponses de MT5 à un ordre
RETCODES_OK = (10008, 10009, 10010)          # ordre placé / exécuté / exécuté partiellement
RETCODE_PARTIAL = 10010
RETCODE_INVALID_FILL = 10030
RETCODES_RETRY = (10004, 10020, 10021)       # prix modifié / prix changé / pas de cotation : rien n'a été exécuté
RETCODE_HELP = {
    10004: "prix modifié (requote)", 10006: "ordre rejeté par le broker", 10013: "requête invalide",
    10014: "volume invalide", 10015: "prix invalide", 10016: "stop invalide ou trop proche",
    10017: "trading désactivé sur ce symbole", 10018: "marché fermé", 10019: "marge insuffisante",
    10020: "prix changé", 10021: "pas de cotation", 10024: "trop de requêtes",
    10026: "trading automatique interdit par le serveur", 10027: 'bouton "Algo Trading" désactivé dans MT5',
    10030: "mode d'exécution refusé", 10031: "pas de connexion au serveur", 10036: "position déjà fermée",
}


# ----------------------------------------------------------------------
# OUTILS
# ----------------------------------------------------------------------
_monotonic = time_module.monotonic   # chronomètre interne : sert seulement à espacer les messages, jamais à décider
_last_logged = {}


def log_once(key: str, level: int, message: str, every: float = 300.0, exc_info: bool = False) -> None:
    """Écrit le message au plus une fois toutes les `every` secondes pour une même clé (évite d'inonder le log)."""
    now = _monotonic()
    if now - _last_logged.get(key, -1e18) >= every:
        _last_logged[key] = now
        log.log(level, message, exc_info=exc_info)


def api_error() -> str | None:
    """Dernière erreur de l'API MT5, ou None si le dernier appel a réussi (code 1 = succès)."""
    err = mt5.last_error()
    if not err or err[0] == 1:
        return None
    return f"{err[0]} {err[1]}"


def describe(result) -> str:
    """Réponse de MT5 à un ordre, en clair."""
    if result is None:
        return f"aucune réponse de MT5 ({mt5.last_error()})"
    text = RETCODE_HELP.get(result.retcode) or getattr(result, "comment", "") or "voir la documentation MT5"
    return f"code {result.retcode} : {text}"


def order_accepted(result) -> bool:
    return result is not None and result.retcode in RETCODES_OK


# ----------------------------------------------------------------------
# MARCHÉS TRADÉS ET RÉGLAGES
# ----------------------------------------------------------------------
_cfg = None


def load_markets() -> dict:
    """Marchés à trader : ceux de marches_valides.json (écrit par valider_marches.py) s'il existe, sinon l'or seul."""
    cfg = {"markets": [{"market": "XAUUSD", "symbol": SYMBOL, "group": "metaux"}],
           "risk": RISK_PERCENT_PER_TRADE, "risk_source": "réglage RISK_PERCENT_PER_TRADE",
           "max_open": MAX_OPEN_POSITIONS, "max_per_group": MAX_POSITIONS_PER_GROUP,
           "guard_mean": GOLD_BACKTEST_MEAN_R, "guard_sd": GOLD_BACKTEST_SD_R,
           "source": f"or seul ({MARKETS_FILE.name} absent)"}
    if not MARKETS_FILE.exists():
        return cfg
    try:
        data = json.loads(MARKETS_FILE.read_text(encoding="utf-8"))
        markets = [{"market": str(m["market"]), "symbol": str(m["symbol"]), "group": str(m.get("group") or m["market"])}
                   for m in data["markets"]]
        if not markets:
            raise ValueError("aucun marché dans le fichier")
        cfg["markets"] = markets
        cfg["source"] = f"{MARKETS_FILE.name} du {data.get('cree_le', '?')}"
        if USE_RECOMMENDED_RISK and data.get("risk_percent_per_trade"):
            cfg["risk"] = min(1.0, float(data["risk_percent_per_trade"]))
            cfg["risk_source"] = "conseillé par valider_marches.py"
        cfg["max_open"] = int(data.get("max_open_positions") or MAX_OPEN_POSITIONS)
        cfg["max_per_group"] = int(data.get("max_positions_per_group") or MAX_POSITIONS_PER_GROUP)
        if data.get("guard_mean_r") is not None and data.get("guard_sd_r"):
            cfg["guard_mean"], cfg["guard_sd"] = float(data["guard_mean_r"]), float(data["guard_sd_r"])
    except (OSError, ValueError, KeyError, TypeError) as e:
        log.warning(f"{MARKETS_FILE.name} illisible ({e}) : le bot reste sur l'or seul.")
        cfg["source"] = f"or seul ({MARKETS_FILE.name} illisible)"
    return cfg


def cfg() -> dict:
    global _cfg
    if _cfg is None:
        _cfg = load_markets()
    return _cfg


def group_of(symbol: str) -> str:
    for m in cfg()["markets"]:
        if m["symbol"] == symbol:
            return m["group"]
    return symbol


# ----------------------------------------------------------------------
# HEURE SERVEUR
# ----------------------------------------------------------------------
def tick_time(symbol: str) -> float | None:
    """
    Heure du serveur au dernier prix reçu pour ce marché (même référentiel que les bougies et que pos.time).
    Elle peut être en retard sur la réalité (marché calme ou fermé), jamais en avance.
    """
    tick = mt5.symbol_info_tick(symbol)
    if tick is None or not tick.time:
        return None
    msc = getattr(tick, "time_msc", 0) or 0
    return msc / 1000.0 if msc else float(tick.time)


def server_time(symbols: list | None = None) -> float | None:
    """
    Heure du serveur du broker : la plus récente des heures des derniers prix reçus.
    L'horloge du PC n'intervient dans AUCUNE décision : qu'elle avance ou qu'elle retarde, le bot fait pareil.
    """
    times = [t for t in (tick_time(s) for s in (symbols or [m["symbol"] for m in cfg()["markets"]])) if t is not None]
    return max(times) if times else None


# ----------------------------------------------------------------------
# CONNEXION ET GARDE-FOUS
# ----------------------------------------------------------------------
def is_authorized(account) -> bool:
    """Compte démo obligatoire tant que ALLOW_REAL_ACCOUNT = False (sauf en DRY_RUN, où aucun ordre ne part)."""
    return DRY_RUN or ALLOW_REAL_ACCOUNT or account.trade_mode == mt5.ACCOUNT_TRADE_MODE_DEMO


def trading_enabled(account) -> bool:
    """MT5 et le compte acceptent-ils les ordres d'un programme, là, maintenant ?"""
    problems = []
    terminal = mt5.terminal_info()
    if terminal is not None:
        if not terminal.trade_allowed:
            problems.append('le bouton "Algo Trading" de MT5 est désactivé')
        if not getattr(terminal, "connected", True):
            problems.append("MT5 n'est pas connecté au serveur du broker")
    if not account.trade_allowed:
        problems.append("le trading est désactivé sur ce compte (connexion avec le mot de passe investisseur ?)")
    if not getattr(account, "trade_expert", True):
        problems.append("le broker interdit le trading automatique sur ce compte")
    if problems:
        log_once("trading_off", logging.WARNING, "Ordres impossibles pour l'instant : " + " ; ".join(problems) + ".", every=600)
    return not problems


def connect_mt5() -> bool:
    if mt5 is None:
        log.error("Le package MetaTrader5 n'est pas installé, ou ce script tourne hors Windows. Abandon.")
        return False
    if not mt5.initialize():
        log.error(f"Échec de connexion à MT5 : {mt5.last_error()}")
        return False
    account = mt5.account_info()
    if account is None:
        log.error("MT5 initialisé mais aucun compte connecté.")
        return False
    is_demo = account.trade_mode == mt5.ACCOUNT_TRADE_MODE_DEMO
    log.info(f"Connecté à MT5 : compte {account.login} ({account.server}), "
             f"{'DÉMO' if is_demo else 'RÉEL'}, solde {account.balance:.2f} {account.currency}")
    if not is_authorized(account):
        log.error("Ce compte n'est PAS un compte démo et ALLOW_REAL_ACCOUNT = False : le bot s'arrête par sécurité.")
        return False
    trading_enabled(account)   # simple avertissement au démarrage ; revérifié avant chaque ordre
    return True


def reconnect() -> None:
    """MT5 ne répond plus (terminal fermé ou redémarré) : on relance la liaison, puis on réessaie au passage suivant."""
    log_once("reconnect", logging.ERROR, "MT5 ne répond plus : tentative de reconnexion (vérifie que MT5 est ouvert et connecté).")
    try:
        mt5.shutdown()
    except Exception:
        pass
    if mt5.initialize() and mt5.account_info() is not None:
        for m in cfg()["markets"]:
            mt5.symbol_select(m["symbol"], True)
        log.info("Liaison avec MT5 rétablie.")
        _last_logged.pop("reconnect", None)
    else:
        time_module.sleep(ERROR_PAUSE_SECONDS)


# ----------------------------------------------------------------------
# DONNÉES
# ----------------------------------------------------------------------
def last_closed_bar_epoch(symbol: str = SYMBOL) -> int | None:
    """
    Heure d'ouverture de la dernière bougie M5 CLÔTURÉE de ce marché.
    MT5 renvoie toujours la bougie en cours en dernier : celle d'avant est donc terminée de façon certaine,
    sans avoir besoin d'aucune horloge. Quand le marché est fermé, la dernière bougie connue est terminée elle
    aussi, mais rien ne permet de le savoir : elle sera traitée à l'apparition de la bougie suivante, c'est-à-dire
    au premier prix de la réouverture (avant, aucun ordre ne passerait de toute façon).
    """
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, 2)
    if rates is None or len(rates) < 2:
        return None
    return int(rates[-2]["time"])


def fetch_closed_m5_bars(symbol: str = SYMBOL) -> pd.DataFrame:
    """Les HISTORY_BARS dernières bougies M5 CLÔTURÉES (la dernière renvoyée par MT5, en cours, est retirée)."""
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, HISTORY_BARS + 1)
    if rates is None or len(rates) < 2:
        raise RuntimeError(f"Aucune donnée reçue de MT5 pour {symbol} (symbole mal orthographié ou non visible ?) : {mt5.last_error()}")
    df = pd.DataFrame(rates)
    df["time_epoch"] = df["time"].astype(np.int64)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df[["time", "time_epoch", "open", "high", "low", "close"]].sort_values("time").reset_index(drop=True)
    return df.iloc[:-1].reset_index(drop=True)


# ----------------------------------------------------------------------
# MÉMOIRE DU BOT (fichier d'état)
# ----------------------------------------------------------------------
def empty_state() -> dict:
    return {"signals": {}, "open_risk": {}, "closed": [], "guard_paused": False}


def load_state() -> dict:
    """
    signals : dernier signal pris dans chaque sens, par marché (pour reprendre la même suite de signaux après
    un redémarrage) ; open_risk : risque engagé sur chaque position ouverte ; closed : résultat en R de chaque
    trade fermé (pour le garde-fou d'usure).
    """
    state = empty_state()
    if not STATE_FILE.exists():
        return state
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            saved = json.load(f)
        if "last_up_bar_epoch" in saved:                       # ancien format (or seul)
            saved = {"signals": {SYMBOL: saved}}

        def as_int(value):
            return int(value) if value is not None else None
        for symbol, s in (saved.get("signals") or {}).items():
            state["signals"][str(symbol)] = {"last_up_bar_epoch": as_int(s.get("last_up_bar_epoch")),
                                             "last_down_bar_epoch": as_int(s.get("last_down_bar_epoch"))}
        for ticket, v in (saved.get("open_risk") or {}).items():
            state["open_risk"][str(ticket)] = {"symbol": str(v["symbol"]), "risk": float(v["risk"])}
        state["closed"] = [{"ticket": int(x["ticket"]), "symbol": str(x["symbol"]), "r": float(x["r"]), "time": int(x["time"])}
                           for x in (saved.get("closed") or [])]
        state["guard_paused"] = bool(saved.get("guard_paused", False))
    except (OSError, ValueError, TypeError, AttributeError, KeyError):
        log.warning("Fichier d'état illisible : on repart de l'historique des bougies (résultats passés oubliés).")
        state = empty_state()
    return state


def save_state(state: dict) -> None:
    """Écriture en deux temps (fichier temporaire puis remplacement) : une coupure de courant ne peut pas abîmer le fichier."""
    tmp = STATE_FILE.with_name(STATE_FILE.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f)
    tmp.replace(STATE_FILE)


_state = None


def state() -> dict:
    global _state
    if _state is None:
        _state = load_state()
    return _state


# ----------------------------------------------------------------------
# SIGNAL (strictement le même code que le backtest)
# ----------------------------------------------------------------------
def _index_of_bar(time_epoch: np.ndarray, bar_epoch) -> int:
    """Position dans la fenêtre de la bougie ouverte à bar_epoch ; très ancien (-10**9) si elle n'y est plus."""
    if bar_epoch is None:
        return -10**9
    i = int(np.searchsorted(time_epoch, bar_epoch, side="left"))
    if i < len(time_epoch) and time_epoch[i] == bar_epoch:
        return i
    return -10**9 if i == 0 else i - 1   # bougie absente (trou de cotation) : la précédente fait foi


def compute_signal(df5: pd.DataFrame, sig_state: dict) -> dict | None:
    """
    Renvoie le signal de la DERNIÈRE bougie clôturée, ou None.
    Utilise strat.add_indicators et strat.find_signals : aucune logique de trading n'est dupliquée ici.
    La règle "un signal par sens toutes les 276 bougies" repart du dernier signal mémorisé dans `sig_state`
    (mis à jour ici), pour que le bot reprenne exactement la même suite de signaux après un redémarrage.
    """
    min_bars = strat.N_DAYS * strat.BARS_PER_DAY + 2
    if len(df5) < max(min_bars, strat.VOL_BARS + 2):
        log.warning(f"Historique insuffisant ({len(df5)} bougies M5) : il en faut au moins {max(min_bars, strat.VOL_BARS + 2)}.")
        return None
    ind = strat.add_indicators(df5[["time", "open", "high", "low", "close"]])
    epochs = df5["time_epoch"].values
    last_up = _index_of_bar(epochs, sig_state.get("last_up_bar_epoch"))
    last_down = _index_of_bar(epochs, sig_state.get("last_down_bar_epoch"))
    signal = strat.find_signals(ind, last_up=last_up, last_down=last_down)
    ups, downs = np.nonzero(signal == 1)[0], np.nonzero(signal == -1)[0]
    if len(ups):
        sig_state["last_up_bar_epoch"] = int(epochs[ups[-1]])
    if len(downs):
        sig_state["last_down_bar_epoch"] = int(epochs[downs[-1]])
    last = len(ind) - 1
    row = ind.iloc[last]
    info = {
        "direction": int(signal[last]), "close": float(row["close"]), "level_high": float(row["level_high"]),
        "level_low": float(row["level_low"]), "sigma_1h": float(row["sigma_1h"]),
        "bar_close_epoch": int(df5["time_epoch"].iloc[last]) + 300, "bar_time": row["time"],
    }
    return info


# ----------------------------------------------------------------------
# POSITIONS ET RISQUE
# ----------------------------------------------------------------------
def bot_positions(symbol: str | None = None) -> list:
    """Positions ouvertes par CE bot (sur un marché, ou sur tous). Une panne de MT5 n'est jamais prise pour "aucune position"."""
    positions = mt5.positions_get()
    if positions is None:
        error = api_error()
        if error:
            raise RuntimeError(f"MT5 n'a pas pu donner les positions ouvertes ({error}).")
        return []
    return [p for p in positions if p.magic == MAGIC_NUMBER and (symbol is None or p.symbol == symbol)]


def realized_pnl_today(now: float) -> float | None:
    """
    Résultat net (profit + commission + swap + frais) des trades de CE bot depuis minuit, heure serveur, tous
    marchés confondus. Un deal est reconnu par la POSITION à laquelle il appartient (ouverte par ce bot), pas
    seulement par son numéro magique : chez certains brokers, la clôture par le stop porte un numéro magique à 0.
    Renvoie None si MT5 n'a pas pu donner l'historique (le bot n'ouvre alors rien, par prudence).
    """
    midnight = int(now // 86400) * 86400
    deals = mt5.history_deals_get(midnight - 7 * 86400, int(now) + 86400)   # 7 jours en arrière : couvre un week-end
    if deals is None:
        if api_error():
            return None
        deals = ()
    ours = {d.position_id for d in deals if d.magic == MAGIC_NUMBER and d.entry == mt5.DEAL_ENTRY_IN}
    total = 0.0
    for d in deals:
        if d.time >= midnight and (d.position_id in ours or d.magic == MAGIC_NUMBER):
            total += d.profit + getattr(d, "commission", 0.0) + getattr(d, "swap", 0.0) + getattr(d, "fee", 0.0)
    return total


def loss_per_lot(order_type, price: float, stop: float, symbol: str = SYMBOL) -> float:
    """Perte (devise du compte) pour 1 lot si le stop est touché. Demandée à MT5 : gère la devise et la taille de contrat."""
    value = mt5.order_calc_profit(order_type, symbol, 1.0, price, stop)
    if value is None or value >= 0:
        return 0.0
    return -value


def compute_lot(equity: float, per_lot_loss: float, info, risk_percent: float | None = None) -> float:
    target = equity * (cfg()["risk"] if risk_percent is None else risk_percent) / 100.0
    step = info.volume_step or 0.01
    lot = np.floor(target / per_lot_loss / step + 1e-9) * step
    lot = min(lot, info.volume_max)
    if lot < info.volume_min:
        # le lot minimum du broker risquerait plus que prévu : accepté seulement si ça reste sous 2x la cible
        if info.volume_min * per_lot_loss <= 2.0 * target:
            lot = info.volume_min
        else:
            return 0.0
    decimals = max(0, int(round(-np.log10(step)))) if step < 1 else 0
    return round(float(lot), decimals)


# ----------------------------------------------------------------------
# ORDRES
# ----------------------------------------------------------------------
def send_market_order(request: dict):
    """Envoie l'ordre ; si le broker refuse le mode d'exécution, essaie les autres (IOC, puis FOK, puis RETURN)."""
    result = None
    for filling in (mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_FOK, mt5.ORDER_FILLING_RETURN):
        result = mt5.order_send(dict(request, type_filling=filling))
        if result is None or result.retcode != RETCODE_INVALID_FILL:
            break
    return result


def close_position(pos, comment: str) -> bool:
    """Ferme une position au marché. Renvoie True si MT5 a accepté l'ordre."""
    tick = mt5.symbol_info_tick(pos.symbol)
    if tick is None:
        return False
    is_buy = pos.type == mt5.ORDER_TYPE_BUY
    request = {
        "action": mt5.TRADE_ACTION_DEAL, "symbol": pos.symbol, "volume": pos.volume,
        "type": mt5.ORDER_TYPE_SELL if is_buy else mt5.ORDER_TYPE_BUY, "position": pos.ticket,
        "price": tick.bid if is_buy else tick.ask, "deviation": ORDER_DEVIATION_POINTS, "magic": MAGIC_NUMBER,
        "comment": comment, "type_time": mt5.ORDER_TIME_GTC,
    }
    result = send_market_order(request)
    if not order_accepted(result):
        log_once(f"close_failed:{pos.ticket}", logging.ERROR,
                 f"{pos.symbol} : échec de la fermeture de la position {pos.ticket} ({describe(result)}). "
                 "Le bot réessaie chaque seconde.", every=60)
        return False
    if result.retcode == RETCODE_PARTIAL:
        log.warning(f"{pos.symbol} : position {pos.ticket} fermée en partie seulement, le reste sera fermé au passage suivant.")
    return True


def ensure_stop(order_ticket: int, stop: float, point: float, symbol: str = SYMBOL):
    """
    Vérifie, juste après l'ouverture, que le stop est bien enregistré chez le broker. S'il manque, le bot le pose ;
    si le broker refuse, la position est refermée tout de suite : ce bot ne garde JAMAIS une position sans stop.
    Renvoie la position trouvée (ou None).
    """
    position = None
    for _ in range(5):
        mine = bot_positions(symbol)
        if mine:
            position = next((p for p in mine if p.ticket == order_ticket), max(mine, key=lambda p: p.time))
            break
        time_module.sleep(0.2)
    if position is None:
        log.warning(f"{symbol} : position introuvable juste après l'ouverture (stop déjà touché, ou MT5 en retard) : vérifie dans MT5.")
        return None
    if position.sl and abs(position.sl - stop) <= 2 * point:
        return position
    log.warning(f"{symbol} : le broker n'a pas enregistré le stop (stop lu : {position.sl}). Pose du stop à {stop}.")
    result = mt5.order_send({"action": mt5.TRADE_ACTION_SLTP, "symbol": symbol, "position": position.ticket,
                             "sl": stop, "tp": 0.0, "magic": MAGIC_NUMBER})
    if order_accepted(result):
        log.info(f"{symbol} : stop posé à {stop} sur la position {position.ticket}.")
        return position
    log.error(f"{symbol} : impossible de poser le stop ({describe(result)}) : fermeture immédiate de la position par sécurité.")
    if close_position(position, "sans_stop"):
        log.info(f"{symbol} : position {position.ticket} refermée (pas de stop possible).")
    return None


def _prepare_entry(sig: dict, equity: float, day_pnl: float, day_start_equity: float, symbol: str):
    """Construit l'ordre d'entrée avec les prix du moment. Renvoie (ordre, description, pas de prix, risque) ou None si annulé."""
    direction = sig["direction"]
    info = mt5.symbol_info(symbol)
    tick = mt5.symbol_info_tick(symbol)
    if info is None or tick is None:
        log.error(f"{symbol} : infos du symbole ou prix indisponibles, pas d'entrée.")
        return None
    order_type = mt5.ORDER_TYPE_BUY if direction == 1 else mt5.ORDER_TYPE_SELL
    price = tick.ask if direction == 1 else tick.bid
    spread = tick.ask - tick.bid
    risk_dist = strat.SL_SIGMA * sig["sigma_1h"]
    if not np.isfinite(risk_dist) or risk_dist <= 0:
        log.info(f"{symbol} : volatilité indisponible, pas d'entrée.")
        return None
    if spread > MAX_SPREAD_TO_RISK * risk_dist:
        log.info(f"{symbol} : signal ANNULÉ, spread {spread:.5g} trop large par rapport au stop ({risk_dist:.5g}).")
        return None
    if abs(price - sig["close"]) > risk_dist:
        log.info(f"{symbol} : signal ANNULÉ, le prix a déjà bougé de {abs(price - sig['close']):.5g} depuis la clôture "
                 f"(plus que la distance du stop, {risk_dist:.5g}).")
        return None
    stop = price - risk_dist if direction == 1 else price + risk_dist
    stop = round(stop, info.digits)
    min_dist = (info.trade_stops_level or 0) * info.point
    if abs(price - stop) < min_dist:
        log.info(f"{symbol} : signal ANNULÉ, stop trop proche pour le broker (minimum {min_dist:.5g}).")
        return None

    per_lot = loss_per_lot(order_type, price, stop, symbol)
    if per_lot <= 0:
        log.error(f"{symbol} : impossible de calculer la perte par lot (order_calc_profit), pas d'entrée.")
        return None
    lot = compute_lot(equity, per_lot, info)
    if lot <= 0:
        log.info(f"{symbol} : signal ANNULÉ, même le lot minimum ({info.volume_min}) risquerait plus de 2x la cible.")
        return None
    trade_risk = lot * per_lot
    budget = day_start_equity * DAILY_RISK_BUDGET_PERCENT / 100.0
    if max(0.0, -day_pnl) + trade_risk > budget:
        log.info(f"{symbol} : signal BLOQUÉ par le budget de risque du jour : perte réalisée {max(0.0, -day_pnl):.2f} + "
                 f"risque du trade {trade_risk:.2f} > budget {budget:.2f}.")
        return None

    side = "ACHAT" if direction == 1 else "VENTE"
    detail = (f"{symbol} : {side} {lot} lot(s) @ {price}, stop {stop} (à {risk_dist:.5g}, risque {trade_risk:.2f} "
              f"= {100 * trade_risk / equity:.2f}% du capital), sortie au plus tard dans {strat.HOLD_MINUTES} min")
    request = {
        "action": mt5.TRADE_ACTION_DEAL, "symbol": symbol, "volume": lot, "type": order_type, "price": price,
        "sl": stop, "tp": 0.0, "deviation": ORDER_DEVIATION_POINTS, "magic": MAGIC_NUMBER,
        "comment": "cassure_niveau_majeur", "type_time": mt5.ORDER_TIME_GTC,
    }
    return request, detail, info.point, trade_risk


def open_position(sig: dict, equity: float, day_pnl: float, day_start_equity: float, symbol: str = SYMBOL) -> bool:
    """Ouvre la position du signal (stop posé depuis le prix d'entrée, pas de TP). Renvoie True si elle est ouverte."""
    for attempt in range(1, ORDER_ATTEMPTS + 1):
        prepared = _prepare_entry(sig, equity, day_pnl, day_start_equity, symbol)
        if prepared is None:
            return False
        request, detail, point, trade_risk = prepared
        if DRY_RUN:
            log.info(f"[DRY_RUN] Aurait ouvert : {detail}")
            return True
        result = send_market_order(request)
        if order_accepted(result):
            executed = getattr(result, "volume", 0.0) or request["volume"]
            partial = f" - exécution PARTIELLE : {executed} lot(s) seulement" if result.retcode == RETCODE_PARTIAL else ""
            log.info(f"Position ouverte (ticket {result.order}) : {detail}{partial}")
            position = ensure_stop(result.order, request["sl"], point, symbol)
            if position is not None:                 # risque réellement engagé, pour mesurer le trade en R
                volume = getattr(position, "volume", executed) or executed
                state()["open_risk"][str(position.ticket)] = {"symbol": symbol, "risk": trade_risk * volume / request["volume"]}
                save_state(state())
            return True
        if result is not None and result.retcode in RETCODES_RETRY and attempt < ORDER_ATTEMPTS:
            log.warning(f"{symbol} : ordre refusé ({describe(result)}), nouvel essai avec le prix du moment ({attempt + 1}/{ORDER_ATTEMPTS}).")
            time_module.sleep(0.3)
            continue
        log.error(f"{symbol} : échec de l'ouverture de position, {describe(result)}")
        return False
    return False


def close_expired_positions(now: float | None = None, can_trade: bool = True) -> None:
    """
    Ferme au marché toute position du bot dont la durée maximale est dépassée.
    Le chrono part de la clôture de la bougie signal (heure d'ouverture de la position ramenée au début de sa
    période de 5 minutes), comme dans le backtest. L'heure de référence est celle du dernier prix du marché de
    la position (`now` sert aux tests) : pendant une fermeture de ce marché, rien n'est tenté.
    """
    for pos in bot_positions():
        current = now if now is not None else tick_time(pos.symbol)
        if current is None:
            continue
        signal_close = int(pos.time // 300) * 300
        if current < signal_close + strat.HOLD_MINUTES * 60:
            continue
        age_min = (current - pos.time) / 60.0
        if DRY_RUN:
            log_once(f"dry_close:{pos.ticket}", logging.INFO,
                     f"[DRY_RUN] Aurait fermé la position {pos.ticket} sur {pos.symbol} (durée max atteinte, {age_min:.0f} min).", every=3600)
            continue
        if not can_trade:
            log_once(f"cannot_close:{pos.ticket}", logging.ERROR,
                     f"{pos.symbol} : la position {pos.ticket} a dépassé sa durée maximale mais MT5 refuse les ordres : ferme-la à la main.")
            continue
        if close_position(pos, "sortie_temps"):
            log.info(f"{pos.symbol} : position {pos.ticket} fermée (durée max atteinte, {age_min:.0f} min), PnL flottant {pos.profit:.2f}.")


# ----------------------------------------------------------------------
# GARDE-FOU D'USURE
# ----------------------------------------------------------------------
_missing_since = {}


def guard_limit(n: int) -> float:
    """Résultat cumulé (en R) sous lequel, après n trades, la stratégie ne se comporte plus comme dans le backtest."""
    return cfg()["guard_mean"] * n - EDGE_GUARD_Z * cfg()["guard_sd"] * np.sqrt(n)


def guard_summary() -> str:
    closed = state()["closed"]
    n = len(closed)
    if n == 0:
        return "aucun trade clôturé pour l'instant"
    total = sum(x["r"] for x in closed)
    text = (f"{n} trades clôturés, {total:+.1f} R au total, {total / n:+.2f} R par trade "
            f"(backtest : {cfg()['guard_mean']:+.2f})")
    if n >= EDGE_GUARD_MIN_TRADES:
        text += f" ; le garde-fou se déclencherait sous {guard_limit(n):+.1f} R"
    else:
        text += f" ; le garde-fou commence à juger à {EDGE_GUARD_MIN_TRADES} trades"
    return text


def record_closed_trades(account) -> None:
    """Repère les positions du bot qui viennent de se fermer (stop, sortie au temps ou à la main) et note leur résultat en R."""
    st = state()
    if not st["open_risk"]:
        return
    open_tickets = {str(p.ticket) for p in bot_positions()}
    for ticket, info in list(st["open_risk"].items()):
        if ticket in open_tickets:
            _missing_since.pop(ticket, None)
            continue
        deals = mt5.history_deals_get(position=int(ticket))
        if deals is None:
            if api_error():
                return                                        # historique illisible : on réessaie au passage suivant
            deals = ()
        exits = [d for d in deals if d.entry != mt5.DEAL_ENTRY_IN]
        if not exits:                                         # la fermeture n'est pas encore dans l'historique
            first_seen = _missing_since.setdefault(ticket, _monotonic())
            if _monotonic() - first_seen > 600:
                log.warning(f"{info['symbol']} : résultat de la position {ticket} introuvable, ignoré par le garde-fou.")
                del st["open_risk"][ticket]
                save_state(st)
            continue
        pnl = sum(d.profit + getattr(d, "commission", 0.0) + getattr(d, "swap", 0.0) + getattr(d, "fee", 0.0) for d in deals)
        r = pnl / info["risk"] if info["risk"] > 0 else 0.0
        st["closed"].append({"ticket": int(ticket), "symbol": info["symbol"], "r": round(r, 4), "time": int(max(d.time for d in exits))})
        del st["open_risk"][ticket]
        _missing_since.pop(ticket, None)
        save_state(st)
        log.info(f"{info['symbol']} : trade clôturé, {r:+.2f} R ({pnl:+.2f} {account.currency}). Bilan : {guard_summary()}.")
        check_guard()


def check_guard() -> None:
    st = state()
    if not EDGE_GUARD_ENABLED or st["guard_paused"]:
        return
    n = len(st["closed"])
    if n < EDGE_GUARD_MIN_TRADES:
        return
    total = sum(x["r"] for x in st["closed"])
    if total < guard_limit(n):
        st["guard_paused"] = True
        save_state(st)
        log.error(f"GARDE-FOU D'USURE DÉCLENCHÉ : après {n} trades, le bot est à {total:+.1f} R alors que le backtest prévoyait "
                  f"{cfg()['guard_mean'] * n:+.1f} R, et qu'il jugeait improbable de passer sous {guard_limit(n):+.1f} R. "
                  "La stratégie ne se comporte plus comme prévu : le bot n'ouvre plus de trades (les positions ouvertes "
                  "restent gérées). Pour repartir de zéro après avoir compris pourquoi : arrête le bot, supprime "
                  f"{STATE_FILE.name}, relance.")


# ----------------------------------------------------------------------
# BOUCLE PRINCIPALE
# ----------------------------------------------------------------------
_day_state = {"day": None, "start_equity": None}
_last_bar_processed = {}      # par marché : heure d'ouverture de la dernière bougie clôturée déjà traitée
_last_bar_seen_at = None      # chronomètre interne au dernier traitement d'une bougie (pour signaler une longue attente)


def process_market(market: dict, account, now: float, can_trade: bool) -> None:
    """Traite la dernière bougie M5 clôturée de ce marché, si elle est nouvelle : signal, contrôles, entrée."""
    global _last_bar_seen_at
    symbol = market["symbol"]
    last_closed = last_closed_bar_epoch(symbol)
    if last_closed is None:
        log_once(f"no_bars:{symbol}", logging.WARNING, f"MT5 ne renvoie aucune bougie M5 pour {symbol} (historique en cours de "
                 f"chargement, ou symbole absent de la fenêtre 'Observation du marché') : le bot attend.", every=600)
        return
    if last_closed == _last_bar_processed.get(symbol):
        return                                   # pas de nouvelle bougie clôturée depuis le dernier passage
    st = state()
    sig_state = st["signals"].setdefault(symbol, {"last_up_bar_epoch": None, "last_down_bar_epoch": None})
    df5 = fetch_closed_m5_bars(symbol)
    sig = compute_signal(df5, sig_state)
    save_state(st)
    _last_bar_processed[symbol] = int(df5["time_epoch"].iloc[-1])
    _last_bar_seen_at = _monotonic()
    _last_logged.pop("idle", None)
    if sig is None:
        return
    tick = mt5.symbol_info_tick(symbol)
    spread_txt = f", spread {tick.ask - tick.bid:.5g}" if tick else ""
    log.info(f"{symbol} | bougie {sig['bar_time']} clôture {sig['close']:.5g} | plus haut 10 j {sig['level_high']:.5g} "
             f"(à {sig['level_high'] - sig['close']:+.5g}) | plus bas 10 j {sig['level_low']:.5g} "
             f"(à {sig['level_low'] - sig['close']:+.5g}) | stop prévu {strat.SL_SIGMA * sig['sigma_1h']:.5g}{spread_txt}")

    direction = sig["direction"]
    if direction == 0:
        return
    side = "ACHAT (clôture au-dessus du plus haut de 10 jours)" if direction == 1 else "VENTE (clôture sous le plus bas de 10 jours)"
    log.info(f"SIGNAL {symbol} : {side}")
    if (direction == 1 and not ALLOW_LONGS) or (direction == -1 and not ALLOW_SHORTS):
        log.info("Ce sens est désactivé dans la configuration : signal ignoré.")
        return
    age = max(0.0, (tick_time(symbol) or now) - sig["bar_close_epoch"])   # heure relue : le calcul a pris un instant
    if age > MAX_SIGNAL_AGE_SECONDS:
        log.info(f"Signal trop ancien ({age:.0f} s après la clôture de la bougie) : ignoré.")
        return
    if EDGE_GUARD_ENABLED and st["guard_paused"]:
        log.info("Signal ignoré : le garde-fou d'usure a mis le bot en pause (voir le message plus haut dans le log).")
        return
    mine = bot_positions()
    if any(p.symbol == symbol for p in mine):
        log.info("Une position du bot est déjà ouverte sur ce marché : signal ignoré (un trade à la fois par marché).")
        return
    c = cfg()
    if len(mine) >= c["max_open"]:
        log.info(f"Signal ignoré : déjà {len(mine)} positions ouvertes (maximum {c['max_open']}).")
        return
    same_group = [p.symbol for p in mine if group_of(p.symbol) == market["group"]]
    if len(same_group) >= c["max_per_group"]:
        log.info(f"Signal ignoré : la famille '{market['group']}' a déjà une position ouverte ({', '.join(same_group)}) ; "
                 "ce serait doubler le même pari.")
        return
    day_pnl = realized_pnl_today(now)
    if day_pnl is None:
        log.error(f"Signal NON EXÉCUTÉ : MT5 n'a pas pu donner l'historique des trades du jour ({mt5.last_error()}), "
                  "le budget de risque ne peut pas être vérifié.")
        return
    if not can_trade and not DRY_RUN:
        log.error("Signal NON EXÉCUTÉ : MT5 refuse les ordres pour l'instant (voir l'avertissement plus haut).")
        return
    open_position(sig, account.equity, day_pnl, _day_state["start_equity"], symbol)


def run_cycle() -> None:
    """
    Un passage, appelé chaque seconde : contrôle du compte, sorties au temps, résultats des trades fermés
    (garde-fou d'usure), puis, pour chaque marché, traitement de sa dernière bougie M5 clôturée SI elle est
    nouvelle (chaque bougie n'est donc traitée qu'une seule fois).
    """
    account = mt5.account_info()
    if account is None:
        reconnect()
        return
    if not is_authorized(account):
        log_once("unauthorized", logging.ERROR, f"Le compte connecté ({account.login}) n'est PAS un compte démo et "
                 "ALLOW_REAL_ACCOUNT = False : le bot ne fait plus rien tant que le compte démo n'est pas rétabli.")
        return
    now = server_time()
    if now is None:
        log_once("no_tick", logging.WARNING, "Aucun prix reçu pour les marchés du bot : le bot attend.", every=600)
        return
    today = int(now // 86400)
    if _day_state["day"] is None or today > _day_state["day"]:
        _day_state["day"], _day_state["start_equity"] = today, account.equity
        log.info(f"Nouveau jour de trading (heure serveur). Capital de référence : {account.equity:.2f} {account.currency}. "
                 f"Bilan du bot : {guard_summary()}.")

    can_trade = trading_enabled(account)
    close_expired_positions(None, can_trade)
    record_closed_trades(account)
    for market in cfg()["markets"]:
        process_market(market, account, now, can_trade)


def report_if_idle() -> None:
    """Écrit une ligne quand aucune bougie n'arrive depuis longtemps, pour qu'un log silencieux ne soit pas un mystère."""
    if _last_bar_seen_at is None:
        return
    idle = _monotonic() - _last_bar_seen_at
    if idle > IDLE_NOTICE_SECONDS:
        log_once("idle", logging.INFO, f"Aucune nouvelle bougie depuis {idle / 60:.0f} min : marché fermé (pause du soir, "
                 "week-end) ou MT5 déconnecté. Le bot attend et reprendra tout seul.", every=3600)


def main() -> None:
    log.info(f"Démarrage du bot v2 (cassure d'un niveau majeur). DRY_RUN={DRY_RUN}. Achats={ALLOW_LONGS}, ventes={ALLOW_SHORTS}.")
    if not connect_mt5():
        sys.exit(1)
    c = cfg()
    for market in list(c["markets"]):
        info = mt5.symbol_info(market["symbol"])
        if info is None:
            log.error(f"Le symbole '{market['symbol']}' n'existe pas chez ce broker ou n'est pas visible : marché ignoré.")
            c["markets"].remove(market)
            continue
        if not info.visible:
            mt5.symbol_select(market["symbol"], True)
    if not c["markets"]:
        log.error("Aucun marché utilisable : vérifie SYMBOL (ou marches_valides.json).")
        sys.exit(1)
    names = ", ".join(m["symbol"] for m in c["markets"])
    log.info(f"Marchés tradés : {names} ({c['source']}). Risque par trade : {c['risk']:.2f} % ({c['risk_source']}).")
    if len(c["markets"]) > 1:
        log.info(f"Au plus {c['max_open']} positions en même temps, et {c['max_per_group']} par famille de marchés corrélés.")
    log.info(f"Garde-fou d'usure : {'actif' if EDGE_GUARD_ENABLED else 'DÉSACTIVÉ'} ; {guard_summary()}.")
    if EDGE_GUARD_ENABLED and state()["guard_paused"]:
        log.error(f"Le garde-fou d'usure est toujours déclenché : le bot n'ouvrira pas de trades (voir {STATE_FILE.name}).")
    log.info("Le bot surveille le marché : une ligne par bougie M5 et par marché. Ctrl+C pour l'arrêter.")
    try:
        while True:
            try:
                run_cycle()
            except Exception as e:  # une erreur ponctuelle ne doit pas arrêter le bot
                log_once(f"cycle_error:{type(e).__name__}:{e}", logging.ERROR,
                         f"Erreur pendant le passage (le bot continue) : {e}", exc_info=True)
                time_module.sleep(ERROR_PAUSE_SECONDS)
            report_if_idle()
            time_module.sleep(POLL_SECONDS)
    except KeyboardInterrupt:
        log.info("Arrêt demandé par l'utilisateur (Ctrl+C).")
    finally:
        mt5.shutdown()
        log.info("Déconnecté de MT5.")


if __name__ == "__main__":
    main()
