import os
import sqlite3
from datetime import datetime, timezone, timedelta
import datetime as _dt
import random
import traceback

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(BASE_DIR, "sip_sync.db")
VENUE_ID_DEFAULT = 1

BETA_FORCED = 0.07
GAMA_FORCED = 0.01
BASE_DEMAND_PCT = 0.05 

NEON_PRIMARY = "#D41876"
NEON_SECONDARY = "#BB17BB"
NEON_ACCENT = "#FF4800"
NEON_GREEN = "#4714FF"
DARK_BG = "#0A0A0A"
CARD_BG = "#1A1A1A"
TEXT_LIGHT = "#F1ECEC"
NEON_COLORS = [NEON_PRIMARY, NEON_SECONDARY, NEON_GREEN, NEON_ACCENT, "#1FA767", '#4ECDC4']

st.markdown(f"""
<style>
    .main .block-container {{
        padding-top: 0.5rem;
        padding-bottom: 0.5rem;
        background-color: {DARK_BG};
        color: {TEXT_LIGHT};
    }}
    .stApp {{ background: {DARK_BG}; }}
    h1,h2,h3,h4 {{ color: {TEXT_LIGHT} !important; text-shadow: 0 0 5px {NEON_PRIMARY}; border-bottom: 1px solid {NEON_PRIMARY}33; padding-bottom:0.2rem; margin-bottom:0.5rem !important; margin-top:0.5rem !important; }}
    h1 {{ font-size:1.4rem !important; }}
    .stButton>button {{ background: transparent; color: {NEON_PRIMARY}; border:1px solid {NEON_PRIMARY}; border-radius:3px; padding:0.2rem 0.6rem; font-size:0.8rem; font-weight:bold; }}
    .dataframe {{ background-color: {CARD_BG}; border: 1px solid {NEON_GREEN}22; font-size:0.75rem; }}
</style>
""", unsafe_allow_html=True)

def get_conn():
    return sqlite3.connect(DB, detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES, check_same_thread=False)

def fetch_df(query, params=()):
    conn = get_conn()
    try:
        df = pd.read_sql_query(query, conn, params=params)
    finally:
        conn.close()
    return df

def _sanitize_value(v):
    if v is None:
        return None
    try:
        if hasattr(pd, "_libs") and hasattr(pd._libs, "tslibs"):
            nat_type = getattr(pd._libs.tslibs.nattype, "NaTType", None)
            if nat_type is not None and isinstance(v, nat_type):
                return None
    except Exception:
        pass
    if isinstance(v, pd.Timestamp):
        if pd.isna(v):
            return None
        dt = v.to_pydatetime()
        if dt.tzinfo is not None:
            dt = dt.astimezone(_dt.timezone.utc).replace(tzinfo=None)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(v, (np.datetime64,)):
        try:
            ts = pd.to_datetime(v)
            if pd.isna(ts):
                return None
            dt = ts.to_pydatetime()
            if dt.tzinfo is not None:
                dt = dt.astimezone(_dt.timezone.utc).replace(tzinfo=None)
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            pass
    if isinstance(v, _dt.datetime):
        dt = v
        if dt.tzinfo is not None:
            dt = dt.astimezone(_dt.timezone.utc).replace(tzinfo=None)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, (list, tuple)):
        sanitized = [_sanitize_value(x) for x in v]
        return sanitized if isinstance(v, list) else tuple(sanitized)
    return v

def _sanitize_params(params):
    if params is None:
        return None
    if isinstance(params, dict):
        return {k: _sanitize_value(v) for k, v in params.items()}
    try:
        seq = tuple(params)
    except Exception:
        return (_sanitize_value(params),)
    return tuple(_sanitize_value(x) for x in seq)

def execute(query, params=()):
    conn = get_conn()
    c = conn.cursor()
    safe_params = _sanitize_params(params)
    try:
        if isinstance(safe_params, dict):
            c.execute(query, safe_params)
        else:
            c.execute(query, safe_params or ())
        conn.commit()
    finally:
        conn.close()

def execute_many(query, rows):
    conn = get_conn()
    c = conn.cursor()
    safe_rows = []
    for r in rows:
        if hasattr(r, "tolist") and not isinstance(r, (list, tuple)):
            r = r.tolist()
        safe = _sanitize_params(r)
        if safe is None:
            safe_rows.append(())
        else:
            if isinstance(safe, dict):
                safe_rows.append(safe)
            else:
                safe_rows.append(tuple(safe))
    try:
        c.executemany(query, safe_rows)
        conn.commit()
    finally:
        conn.close()

def get_pricing_rule(venue_id):
    q = "SELECT alpha, beta, gama, max_up_pct, max_down_pct, min_step FROM pricing_rules WHERE venue_id=? LIMIT 1"
    df = fetch_df(q, (venue_id,))
    if df.empty:
        return (1.0, BETA_FORCED, GAMA_FORCED, 0.60, 0.50, 0.05)
    row = df.iloc[0]
    alpha = float(row['alpha'])
    return (alpha, float(row.get('beta', BETA_FORCED)), float(row.get('gama', GAMA_FORCED)),
            float(row['max_up_pct']), float(row['max_down_pct']), float(row['min_step']))

def compute_expected_rate_hour(venue_id, drink_id):
    q_expected = """
        SELECT COALESCE(AVG(hourly), 0) as avg_hour FROM (
            SELECT strftime('%Y-%m-%d %H', timestamp) as hour, SUM(qty) as hourly
            FROM orders
            WHERE venue_id=? AND drink_id=?
            GROUP BY hour
        )
    """
    df = fetch_df(q_expected, (venue_id, drink_id))
    if df.empty:
        return 3.0
    expected = df['avg_hour'].iloc[0]
    if expected <= 0:
        return 3.0
    return float(expected)

def compute_event_factor(venue_id, manual_override=None):
    """
    Devuelve E y desc. Asegura que E sea uno de {0.0, 0.5, 1.0}.
    Si manual_override no está en el conjunto permitido, se ignora.
    """
    allowed = {0.0, 0.5, 1.0}
    q = "SELECT event_factor, noise_level, description, tick FROM event_context WHERE venue_id=? ORDER BY tick DESC LIMIT 1"
    df = fetch_df(q, (venue_id,))
    if df.empty:
        E = 0.0
        desc = ""
    else:
        E = float(df['event_factor'].iloc[0])
        desc = df['description'].iloc[0] if 'description' in df.columns else ""
    if manual_override is not None:
        try:
            m = float(manual_override)
            if m in allowed:
                E = m
        except Exception:
            pass
    if E not in allowed:
        E = 0.0
    return E, desc

def compute_inventory_indicator(venue_id, drink_id):
    q = "SELECT stock_units, stock_target FROM inventory WHERE venue_id=? AND drink_id=?"
    df = fetch_df(q, (venue_id, drink_id))
    if df.empty:
        return 1.0
    stock, target = df.iloc[0]['stock_units'], df.iloc[0]['stock_target']
    if target is None or target <= 0:
        target = 1.0
    I = float(stock / target)
    I = float(np.clip(I, 0.0, 2.0))
    return I

def get_last_price(venue_id, drink_id):
    q = "SELECT current_price FROM pricing_state WHERE venue_id=? AND drink_id=? ORDER BY tick DESC LIMIT 1"
    df = fetch_df(q, (venue_id, drink_id))
    if df.empty:
        q2 = "SELECT base_price FROM drink_master WHERE drink_id=?"
        df2 = fetch_df(q2, (drink_id,))
        if df2.empty:
            return 1.0
        return float(df2['base_price'].iloc[0])
    return float(df['current_price'].iloc[0])

def round_step(price, step):
    if step is None or step == 0:
        return float(price)
    return float(round(price / step) * step)

def _enforce_min_step_change(last_price, candidate_price, min_step):
    if min_step is None or min_step <= 0:
        return float(candidate_price)
    diff = candidate_price - last_price
    if abs(diff) < min_step and abs(diff) > 1e-9:
        if diff > 0:
            return float(last_price + min_step)
        else:
            return float(max(0.0, last_price - min_step))
    return float(candidate_price)

def recalc_prices_realtime(venue_id, window_minutes=10, manual_E=None, weight_share=1.0, tick_time=None):
    """
    Recalcula precios usando la fórmula:
      Pt = P_base * (1 + effective_demand_pct + beta * E - gama * I)
    donde:
      effective_demand_pct = alpha * BASE_DEMAND_PCT * D
    y D = blend(D_share_ratio, D_rel) pero weight_share ahora es binario (0 o 1).
    Nota: beta y gama se forzan a BETA_FORCED y GAMA_FORCED en la aplicación.
    """
    if tick_time is None:
        tick_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    alpha_db, _, _, max_up_pct, max_down_pct, min_step = get_pricing_rule(venue_id)
    beta = BETA_FORCED
    gama = GAMA_FORCED
    alpha = alpha_db

    drinks = fetch_df("SELECT drink_id, base_price, name FROM drink_master WHERE is_active=1")
    if drinks.empty:
        return 0
    drink_ids = [int(x) for x in drinks['drink_id'].tolist()]

    now = datetime.now(timezone.utc)
    window_start = (now - timedelta(minutes=window_minutes)).strftime("%Y-%m-%d %H:%M:%S")
    now_s = now.strftime("%Y-%m-%d %H:%M:%S")
    q_recent = """
        SELECT drink_id, COALESCE(SUM(qty),0) as qty FROM orders
        WHERE venue_id=? AND timestamp BETWEEN ? AND ?
        GROUP BY drink_id
    """
    df_recent = fetch_df(q_recent, (venue_id, window_start, now_s))
    recent_qty_map = {d: 0 for d in drink_ids}
    for _, r in df_recent.iterrows():
        recent_qty_map[int(r['drink_id'])] = int(r['qty'])
    recent_rates_per_hour = {d: recent_qty_map.get(d, 0) * (60.0 / max(1.0, window_minutes)) for d in drink_ids}

    expected_rates = {d: compute_expected_rate_hour(venue_id, d) for d in drink_ids}
    total_recent = float(sum(recent_rates_per_hour.values()))
    total_expected = float(sum(expected_rates.values()))
    if total_expected <= 0:
        total_expected = float(len(drink_ids)) * 1.0
        expected_rates = {d: 1.0 for d in drink_ids}

    E, desc = compute_event_factor(venue_id, manual_E)

    rows_state = []
    debug_rows = []
    for _, row in drinks.iterrows():
        drink_id = int(row['drink_id'])
        base_price = float(row['base_price'])
        last_price = get_last_price(venue_id, drink_id)

        recent = recent_rates_per_hour.get(drink_id, 0.0)
        expected = expected_rates.get(drink_id, 1.0)

        demand_share = (recent / total_recent) if total_recent > 0 else 0.0
        expected_share = (expected / total_expected) if total_expected > 0 else (1.0 / max(1, len(drink_ids)))
        if expected_share <= 0:
            expected_share = 1.0 / max(1, len(drink_ids))

        D_share_ratio = (demand_share / expected_share) if expected_share > 0 else 1.0
        D_rel = (recent / expected) if expected > 0 else 1.0

        D = float(weight_share * D_share_ratio + (1.0 - weight_share) * D_rel)

        effective_demand_pct = float(alpha * BASE_DEMAND_PCT * D)

        I = compute_inventory_indicator(venue_id, drink_id)
        raw_price = base_price * (1.0 + effective_demand_pct + beta * E - gama * I)

        max_price = last_price * (1.0 + max_up_pct)
        min_price = last_price * (1.0 - max_down_pct)

        candidate = _enforce_min_step_change(last_price, raw_price, min_step)
        capped_price = float(np.clip(candidate, min_price, max_price))
        rounded_price = round_step(capped_price, min_step)
        final_price = rounded_price

        rows_state.append((tick_time, venue_id, drink_id, final_price, D, E, I))
        debug_rows.append((tick_time, venue_id, drink_id, last_price, raw_price, capped_price, rounded_price, final_price,
                           f"D_share_ratio={D_share_ratio:.3f},D_rel={D_rel:.3f},D={D:.3f},effective_demand_pct={effective_demand_pct:.4f},beta={beta:.3f},gama={gama:.3f},evt_desc={desc}"))

    execute_many("""
        INSERT INTO pricing_state(tick, venue_id, drink_id, current_price, D, E, I)
        VALUES (?,?,?,?,?,?,?)
    """, rows_state)
    execute_many("""
        INSERT INTO debug_logs(tick, venue_id, drink_id, last_price, raw_price, capped_price, rounded_price, final_price, reason)
        VALUES (?,?,?,?,?,?,?,?,?)
    """, debug_rows)
    return len(rows_state)

st.set_page_config(page_title="SipSync", layout="wide", page_icon="⚡")

col1, col1 = st.columns([1, 10000])
with col1:
    st.markdown(f"<h1 style='color: {NEON_PRIMARY}; text-shadow: 0 0 10px {NEON_PRIMARY}; margin: 0; font-size: 1.4rem; line-height: 1;'>SIP SYNC • Dynamic Pricing Prototype</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='color: {TEXT_LIGHT}; opacity: 0.7; margin: 1; font-size: 0.7rem;'>Dynamic Pricing • Real-time</p>", unsafe_allow_html=True)

with st.sidebar:
    st.markdown(f"<h3 style='color: {NEON_SECONDARY};'>CONTROLS</h3>", unsafe_allow_html=True)
    venue_id = st.number_input("Venue ID", value=VENUE_ID_DEFAULT, min_value=1, key="venue_compact")
    window_minutes_ui = st.slider("Window (min)", min_value=1, max_value=60, value=10, key="window_compact")

    e_label_map = {
        0.0: "0.0 — Normal day (no event)",
        0.5: "0.5 — Busy night / Weekend",
        1.0: "1.0 — Peak / High-intensity (DJ, football, etc.)"
    }
    e_options = [0.0, 0.5, 1.0]
    manual_E = st.selectbox("Event factor (E)", options=e_options,
                            format_func=lambda v: e_label_map.get(v, str(v)),
                            index=0, key="E_compact")

    weight_choice = st.radio("Blend (mode)", options=[0,1], index=1,
                             format_func=lambda v: "Share-based (1) — redistribute demand / others drop" if v==1 else "Per-drink ratio (0) — per-bottle reaction",
                             key="blend_compact")
    weight_share = float(weight_choice)

tab1, tab2, tab3 = st.tabs(["PRICES", "ORDERS", "GRAF"])

with tab1:
    col_prices, col_stats = st.columns([3, 1])
    with col_prices:
        q_prices = """
            SELECT pm.drink_id, pm.name, pm.base_price,
                   ps.current_price, ps.D, ps.E, ps.I
            FROM drink_master pm
            LEFT JOIN (
                SELECT * FROM pricing_state WHERE (venue_id=?) AND id IN (
                    SELECT MAX(id) FROM pricing_state WHERE venue_id=? GROUP BY drink_id
                )
            ) ps ON pm.drink_id = ps.drink_id
            WHERE pm.is_active=1
        """
        price_df = fetch_df(q_prices, (venue_id, venue_id))
        if not price_df.empty:
            price_df['base_price'] = pd.to_numeric(price_df['base_price'], errors='coerce')
            price_df['current_price'] = pd.to_numeric(price_df['current_price'], errors='coerce')
            price_df['current_price'] = price_df['current_price'].where(price_df['current_price'].notnull(), price_df['base_price'])
            price_df['change_pct'] = ((price_df['current_price'] - price_df['base_price']) / price_df['base_price'] * 100).round(1)
        st.markdown(f"<h4 style='color: {NEON_GREEN};'>Actual Prices</h4>", unsafe_allow_html=True)
        if not price_df.empty:
            display_df = price_df[['name', 'base_price', 'current_price', 'change_pct', 'D']].round(3)
            display_df.columns = ['Drink', 'Base', 'Actual', 'Δ%', 'D']
            st.dataframe(display_df, use_container_width=True, height=180)
    with col_stats:
        alpha, _, _, max_up_pct, max_down_pct, min_step = get_pricing_rule(venue_id)
        st.markdown(f"<h4 style='color: {NEON_ACCENT};'>Params</h4>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:0.8rem;'>α (weight): <b style='color:{NEON_PRIMARY}'>{alpha:.3f}</b><br>β (forced): <b style='color:{NEON_PRIMARY}'>{BETA_FORCED:.3f}</b><br>γ (forced): <b style='color:{NEON_PRIMARY}'>{GAMA_FORCED:.3f}</b></div>", unsafe_allow_html=True)
        total_orders = fetch_df("SELECT COUNT(*) as count FROM orders WHERE venue_id=?", (venue_id,))['count'].iloc[0]
        today_orders = fetch_df("SELECT COUNT(*) as count FROM orders WHERE venue_id=? AND DATE(timestamp)=DATE('now')", (venue_id,))['count'].iloc[0]
        st.metric("Total Orders", total_orders)
        st.metric("Today", today_orders)

with tab2:
    col_order = st.columns([1])[0]
    with col_order:
        st.markdown(f"<h4 style='color: {NEON_GREEN};'>New Order</h4>", unsafe_allow_html=True)
        drinks_df = fetch_df("SELECT drink_id, name, base_price FROM drink_master WHERE is_active=1")
        customers_df = fetch_df("SELECT customer_id, name FROM customers ORDER BY customer_id")
        customer_options = [(int(r['customer_id']), r['name']) for _, r in customers_df.iterrows()]
        customer_choices = [f"{cid}: {name}" for cid, name in customer_options]
        customer_choices.insert(0, "New Client...")
        with st.form("order_form_compact", clear_on_submit=True):
            col_drink, col_qty = st.columns([2, 1])
            with col_drink:
                drink_sel = st.selectbox("Drink", options=drinks_df['drink_id'].tolist(),
                                        format_func=lambda x: drinks_df[drinks_df['drink_id']==x]['name'].iloc[0])
            with col_qty:
                qty = st.number_input("Cant", min_value=1, max_value=10, value=1)
            customer_choice = st.selectbox("Client", options=customer_choices)
            new_customer_name = None
            if customer_choice == "New Client...":
                new_customer_name = st.text_input("Name", placeholder="Name...")
            submitted = st.form_submit_button("Create Order", use_container_width=True)
            if submitted:
                try:
                    if customer_choice == "New Client...":
                        if not new_customer_name or new_customer_name.strip() == "":
                            st.error("Name required for new client.")
                        else:
                            q_max = "SELECT COALESCE(MAX(customer_id),0)+1 as nextid FROM customers"
                            nextid_df = fetch_df(q_max)
                            nextid = int(nextid_df['nextid'].iloc[0]) if not nextid_df.empty else random.randint(1000,9999)
                            execute("INSERT OR REPLACE INTO customers(customer_id, name) VALUES (?,?)", (nextid, new_customer_name.strip()))
                            customer_id = nextid
                    else:
                        parts = customer_choice.split(":")
                        customer_id = int(parts[0])
                    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
                    execute("INSERT INTO orders(venue_id, drink_id, customer_id, timestamp, qty) VALUES (?,?,?,?,?)",
                            (venue_id, int(drink_sel), int(customer_id), ts, int(qty)))
                    # call recalc with manual_E and binary weight_share
                    n = recalc_prices_realtime(venue_id, window_minutes=window_minutes_ui, manual_E=manual_E, weight_share=weight_share)
                    st.success(f"Order + price recalculated ({n} drinks).")
                except Exception as e:
                    st.error(f"Error: {e}")
    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown(f"<h4 style='color: {NEON_ACCENT};'>Last 20 saved orders</h4>", unsafe_allow_html=True)
    last_orders = fetch_df("""
        SELECT order_id, venue_id, drink_id, customer_id, timestamp, qty
        FROM orders
        WHERE venue_id=?
        ORDER BY order_id DESC
        LIMIT 20
    """, (venue_id,))
    if last_orders.empty:
        st.info("No orders yet.")
    else:
        drinks_map = fetch_df("SELECT drink_id, name FROM drink_master")
        customers_map = fetch_df("SELECT customer_id, name FROM customers")
        last_orders = last_orders.merge(drinks_map, on='drink_id', how='left').rename(columns={'name': 'drink_name'})
        last_orders = last_orders.merge(customers_map, on='customer_id', how='left').rename(columns={'name': 'customer_name'})
        display_cols = ['order_id', 'timestamp', 'drink_id', 'drink_name', 'customer_id', 'customer_name', 'qty']
        st.dataframe(last_orders[display_cols].rename(columns={
            'order_id': 'Order ID', 'timestamp': 'Timestamp', 'drink_id': 'Drink ID', 'drink_name': 'Drink',
            'customer_id': 'Customer ID', 'customer_name': 'Client', 'qty': 'Quantity'
        }).reset_index(drop=True), use_container_width=True, height=350)

with tab3:
    st.markdown(f"<h4 style='color: {NEON_PRIMARY};'>Price Evolution</h4>", unsafe_allow_html=True)
    plot_df = fetch_df("SELECT tick, drink_id, current_price FROM pricing_state WHERE venue_id=? ORDER BY tick", (venue_id,))
    if plot_df.empty:
        st.info("There is no price history yet. Run some ticks to generate history.")
    else:
        drinks_map = fetch_df("SELECT drink_id, name FROM drink_master WHERE is_active=1")
        plot_df['tick_dt'] = pd.to_datetime(plot_df['tick'])
        unique_ticks = sorted(plot_df['tick_dt'].unique())
        DEFAULT_LAST_N = 20
        if len(unique_ticks) > 0:
            if len(unique_ticks) > DEFAULT_LAST_N:
                start_default = unique_ticks[-DEFAULT_LAST_N]
            else:
                start_default = unique_ticks[0]
            end_default = unique_ticks[-1]
        else:
            start_default = None
            end_default = None

        last_n = st.selectbox("Show last:", [20, 50, 100, "All"], index=0, key="last_n")
        if last_n == "All":
            x_start = unique_ticks[0]
            x_end = unique_ticks[-1]
        else:
            n = int(last_n)
            if len(unique_ticks) > n:
                x_start = unique_ticks[-n]
            else:
                x_start = unique_ticks[0]
            x_end = unique_ticks[-1]

        fig_hist = go.Figure()
        view_option = st.radio("View:", ["All the drinks", "Specific Drink"], horizontal=True, key="view_opt")
        if view_option == "Specific Drink":
            selected_drink = st.selectbox("Drink", options=drinks_map['drink_id'].tolist(),
                                         format_func=lambda x: drinks_map[drinks_map['drink_id']==x]['name'].iloc[0], key="single_plot")
            df_plot = plot_df[plot_df['drink_id'] == selected_drink].copy()
            if not df_plot.empty:
                fig_hist.add_trace(go.Scatter(x=pd.to_datetime(df_plot['tick']), y=df_plot['current_price'],
                                              mode='lines+markers', name=drinks_map[drinks_map['drink_id']==selected_drink]['name'].iloc[0],
                                              line=dict(color=NEON_PRIMARY)))
        else:
            for i, d_id in enumerate(sorted(plot_df['drink_id'].unique())):
                sub = plot_df[plot_df['drink_id'] == d_id].copy()
                if sub.empty:
                    continue
                fig_hist.add_trace(go.Scatter(x=pd.to_datetime(sub['tick']), y=sub['current_price'],
                                              mode='lines', name=drinks_map[drinks_map['drink_id']==d_id]['name'].iloc[0],
                                              line=dict(color=NEON_COLORS[i % len(NEON_COLORS)], width=1.5)))
        fig_hist.update_layout(template='plotly_dark', height=420, margin=dict(l=20, r=20, t=30, b=20),
                                yaxis_title='Price    (€)')
        fig_hist.update_xaxes(rangeslider=dict(visible=True))
        if x_start is not None and x_end is not None:
            fig_hist.update_xaxes(range=[x_start.isoformat(), x_end.isoformat()])
        st.plotly_chart(fig_hist, use_container_width=True)

st.markdown(f"""
<div style='text-align: center; margin-top: 0.5rem; padding: 0.5rem; border-top: 1px solid {NEON_PRIMARY}22;'>
    <p style='color: {TEXT_LIGHT}; opacity: 0.5; font-size: 0.6rem; margin: 0;'>
        SipSync • P ​ =P    ​ ×(1+αD+βE−γI) •
    </p>
</div>

""", unsafe_allow_html=True)
