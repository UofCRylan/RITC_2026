
# Requirements:  pip install requests matplotlib

import tkinter as tk
import requests
import threading
import time
import math
import traceback
import re

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# ─── CONFIG ───────────────────────────────────────────────────────────────
RIT_API_BASE = "http://localhost:9995/v1"
API_KEY = "RYL2000"
POLL_MS = 250

BG       = "#0e1117"
BG_PANEL = "#161b22"
BG_CARD  = "#1c2333"
BG_INPUT = "#0d1117"
BORDER   = "#30363d"
FG       = "#e6edf3"
FG_DIM   = "#7d8590"
GREEN    = "#2ea043"
RED      = "#da3633"
YELLOW   = "#d29922"
BLUE     = "#388bfd"
PURPLE   = "#8957e5"
CYAN     = "#39d353"

COMM = {
    "RITC": 0.01, "COMP": 0.02,
    "TRNT": 0.01, "MTRL": 0.01,
    "BLU": 0.04,  "RED": 0.03,  "GRN": 0.02,
    "WDY": 0.02,  "BZZ": 0.02,  "BNN": 0.03,
    "VNS": 0.02,  "MRS": 0.02,  "JPTR": 0.02, "STRN": 0.02,
}
MAX_ORDER = {"RITC": 10000, "COMP": 15000}
DEFAULT_MAX = 10000
FONT = "Segoe UI"
MONO = "Consolas"


# ─── RIT API ──────────────────────────────────────────────────────────────
class RIT:
    def __init__(self):
        self.s = requests.Session()
        self.s.headers.update({"X-API-Key": API_KEY})
        self.base = RIT_API_BASE

    def _get(self, path, params=None):
        try:
            r = self.s.get(f"{self.base}{path}", params=params, timeout=1)
            return r.json() if r.status_code == 200 else None
        except Exception:
            return None

    def _post(self, path, params=None):
        try:
            r = self.s.post(f"{self.base}{path}", params=params, timeout=1.5)
            if r.status_code in (200, 201):
                return r.json()
            return {"_error": True, "status": r.status_code, "body": r.text}
        except Exception as e:
            return {"_error": True, "msg": str(e)}

    def _delete(self, path, params=None):
        try:
            r = self.s.delete(f"{self.base}{path}", params=params, timeout=1.5)
            return r.status_code in (200, 201, 204)
        except Exception:
            return False

    def case(self):            return self._get("/case")
    def trader(self):          return self._get("/trader")
    def securities(self):      return self._get("/securities")
    def book(self, tk, n=20):  return self._get("/securities/book", {"ticker": tk, "limit": n})
    def tenders(self):         return self._get("/tenders")

    def accept_tender(self, tid, price=None):
        p = {}
        if price is not None:
            p["price"] = price
        return self._post(f"/tenders/{tid}", p if p else None)

    def decline_tender(self, tid):
        return self._delete(f"/tenders/{tid}")

    def market_order(self, ticker, qty, action):
        mx = MAX_ORDER.get(ticker, DEFAULT_MAX)
        rem = abs(int(qty))
        results = []
        while rem > 0:
            c = min(rem, mx)
            results.append(self._post("/orders", {
                "ticker": ticker, "type": "MARKET",
                "quantity": c, "action": action}))
            rem -= c
        return results


# ─── VALIDATED ENTRIES ────────────────────────────────────────────────────
class IntEntry(tk.Entry):
    def __init__(self, master, **kw):
        self._var = tk.StringVar()
        self._var.trace_add("write", self._validate)
        super().__init__(master, textvariable=self._var, **kw)

    def _validate(self, *_):
        v = self._var.get()
        c = re.sub(r'[^0-9]', '', v)
        if c != v:
            self._var.set(c)

    def get_int(self):
        try: return int(self._var.get())
        except: return 0

    def set_val(self, n):
        self._var.set(str(int(n)))


class FloatEntry(tk.Entry):
    def __init__(self, master, **kw):
        self._var = tk.StringVar()
        self._var.trace_add("write", self._validate)
        super().__init__(master, textvariable=self._var, **kw)

    def _validate(self, *_):
        v = self._var.get()
        c = re.sub(r'[^0-9.]', '', v)
        parts = c.split('.')
        if len(parts) > 2:
            c = parts[0] + '.' + ''.join(parts[1:])
        if c != v:
            self._var.set(c)

    def get_float(self):
        try: return float(self._var.get())
        except: return 0.0

    def set_val(self, f):
        self._var.set(f"{f:.2f}")


# ─── SECURITY PANEL ──────────────────────────────────────────────────────
class SecurityPanel(tk.Frame):
    def __init__(self, parent, ticker, rit, **kw):
        super().__init__(parent, bg=BG_PANEL, bd=0, highlightbackground=BORDER,
                         highlightthickness=1, **kw)
        self.ticker = ticker
        self.rit = rit
        self.comm = COMM.get(ticker, 0.02)
        self.last_price = None
        self.bid_price = None
        self.ask_price = None
        self.position = 0
        self.tender = None
        self.current_tick = 0
        self._exit_action = "SELL"
        self._tender_count = 0
        self._wta_price_set = False
        self._build()

    def _build(self):
        # ── Header ──
        hdr = tk.Frame(self, bg=BG_PANEL)
        hdr.pack(fill="x", padx=10, pady=(8, 4))

        self.lbl_tick = tk.Label(hdr, text=self.ticker, font=(MONO, 16, "bold"),
                                  fg="white", bg=BG_PANEL)
        self.lbl_tick.pack(side="left")
        self.lbl_price = tk.Label(hdr, text="—", font=(MONO, 13),
                                   fg=CYAN, bg=BG_PANEL)
        self.lbl_price.pack(side="left", padx=(14, 0))
        self.lbl_pos = tk.Label(hdr, text="Pos: 0", font=(MONO, 11),
                                 fg=FG, bg=BG_PANEL)
        self.lbl_pos.pack(side="left", padx=(14, 0))
        self.lbl_comm = tk.Label(hdr, text=f"${self.comm:.2f}/sh",
                                  font=(FONT, 8), fg=FG_DIM, bg=BG_PANEL)
        self.lbl_comm.pack(side="right")

        tk.Frame(self, bg=BORDER, height=1).pack(fill="x", padx=8)

        # ── Volume chart ──
        chart_fr = tk.Frame(self, bg=BG_PANEL)
        chart_fr.pack(fill="both", expand=True, padx=6, pady=(4, 0))
        self.fig = Figure(figsize=(3.0, 1.4), dpi=90, facecolor=BG_PANEL)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=chart_fr)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self._draw_empty()

        tk.Frame(self, bg=BORDER, height=1).pack(fill="x", padx=8, pady=(2, 0))

        # ── Tender card ──
        # Use a container with grid so button order is ALWAYS fixed
        self.tender_fr = tk.Frame(self, bg=BG_CARD)
        self.tender_fr.pack(fill="x", padx=6, pady=(4, 4))

        # Row 0: tender type + multi-tender warning
        type_row = tk.Frame(self.tender_fr, bg=BG_CARD)
        type_row.pack(fill="x", pady=(6, 2), padx=8)

        self.lbl_multi = tk.Label(type_row, text="", font=(FONT, 10),
                                   fg=YELLOW, bg=BG_CARD)
        self.lbl_multi.pack(side="left")

        self.lbl_ttype = tk.Label(type_row, text="NO ACTIVE TENDER",
                                   font=(FONT, 10, "bold"), fg=FG_DIM,
                                   bg=BG_CARD)
        self.lbl_ttype.pack(side="left", expand=True)

        # Row 1: info
        self.lbl_tinfo = tk.Label(self.tender_fr, text="", font=(MONO, 9),
                                   fg=FG, bg=BG_CARD, anchor="center")
        self.lbl_tinfo.pack(fill="x", padx=8)

        # Row 2: P&L
        self.lbl_pnl = tk.Label(self.tender_fr, text="", font=(MONO, 13, "bold"),
                                 fg=FG, bg=BG_CARD)
        self.lbl_pnl.pack(pady=(4, 2))

        # Row 3: timer
        self.lbl_timer = tk.Label(self.tender_fr, text="", font=(MONO, 9),
                                   fg=YELLOW, bg=BG_CARD)
        self.lbl_timer.pack(pady=(0, 2))

        # Row 4: recommendation text
        self.lbl_rec = tk.Label(self.tender_fr, text="", font=(FONT, 8),
                                 fg=FG_DIM, bg=BG_CARD, anchor="center")
        self.lbl_rec.pack(padx=8, pady=(0, 2))

        # Row 5: WTA price entry (always in layout, shown/hidden via widgets inside)
        self.wta_fr = tk.Frame(self.tender_fr, bg=BG_CARD)
        self.wta_fr.pack(fill="x", padx=8, pady=(0, 4))

        self.wta_price = FloatEntry(self.wta_fr, font=(MONO, 11),
                                     bg=BG_INPUT, fg="white",
                                     insertbackground="white",
                                     justify="center",
                                     highlightbackground=BLUE,
                                     highlightthickness=1)
        self.btn_wta_submit = tk.Button(self.wta_fr, text="Submit Bid",
                                         font=(FONT, 9, "bold"),
                                         bg=BLUE, fg="white",
                                         activebackground="#1a7af8",
                                         command=self._submit_wta,
                                         cursor="hand2", relief="flat")
        # These get grid'd when visible, hidden otherwise
        self._wta_visible = False
        self._hide_wta()

        # Row 6: ACCEPT — ALWAYS at this position
        self.btn_accept = tk.Button(self.tender_fr, text="ACCEPT",
                                     font=(FONT, 10, "bold"),
                                     bg="#238636", fg="white",
                                     activebackground="#2ea043",
                                     command=self._accept, state="disabled",
                                     cursor="hand2", relief="flat", bd=0)
        self.btn_accept.pack(fill="x", padx=8, pady=(2, 2))

        # Row 7: REJECT — ALWAYS below accept
        self.btn_reject = tk.Button(self.tender_fr, text="REJECT",
                                     font=(FONT, 9),
                                     bg="#21262d", fg="#da3633",
                                     activebackground="#30363d",
                                     command=self._reject, state="disabled",
                                     cursor="hand2", relief="flat", bd=0)
        self.btn_reject.pack(fill="x", padx=8, pady=(0, 6))

        tk.Frame(self, bg=BORDER, height=1).pack(fill="x", padx=8)

        # ── Trade row (50/50 grid, NEVER auto-updates) ──
        trade_fr = tk.Frame(self, bg=BG_PANEL)
        trade_fr.pack(fill="x", padx=8, pady=(6, 2))
        trade_fr.columnconfigure(0, weight=1, uniform="t")
        trade_fr.columnconfigure(1, weight=1, uniform="t")

        self.btn_trade = tk.Button(trade_fr, text="—",
                                    font=(FONT, 10, "bold"),
                                    bg="#21262d", fg=FG_DIM,
                                    activebackground="#30363d",
                                    command=self._manual_trade,
                                    state="disabled", cursor="hand2",
                                    relief="flat", bd=0)
        self.btn_trade.grid(row=0, column=0, sticky="nsew", padx=(0, 2), ipady=5)

        self.qty_entry = IntEntry(trade_fr, font=(MONO, 12),
                                   bg=BG_INPUT, fg="white",
                                   insertbackground="white",
                                   justify="center",
                                   highlightbackground=BORDER,
                                   highlightthickness=1)
        self.qty_entry.grid(row=0, column=1, sticky="nsew", padx=(2, 0), ipady=5)
        self.qty_entry.set_val(0)
        self.qty_entry.bind("<Return>", lambda e: self._manual_trade())

        # ── Exit position ──
        self.btn_exit = tk.Button(self, text="EXIT POSITION",
                                   font=(FONT, 9, "bold"),
                                   bg=PURPLE, fg="white",
                                   activebackground="#6e40c9",
                                   command=self._exit_position,
                                   state="disabled", cursor="hand2",
                                   relief="flat", bd=0)
        self.btn_exit.pack(fill="x", padx=8, pady=(2, 8), ipady=3)

    def _show_wta(self):
        if not self._wta_visible:
            self.wta_price.pack(side="left", fill="x", expand=True, padx=(0, 4))
            self.btn_wta_submit.pack(side="right")
            self._wta_visible = True

    def _hide_wta(self):
        if self._wta_visible or True:  # always safe to call
            self.wta_price.pack_forget()
            self.btn_wta_submit.pack_forget()
            self._wta_visible = False

    # ── Chart ────────────────────────────────────────────────────────────

    def _draw_empty(self):
        self.ax.clear()
        self.ax.set_facecolor(BG_PANEL)
        self.ax.text(0.5, 0.5, "Waiting...", ha="center", va="center",
                     transform=self.ax.transAxes, color=FG_DIM, fontsize=9)
        for sp in self.ax.spines.values():
            sp.set_visible(False)
        self.ax.tick_params(colors=FG_DIM, length=0)
        self.fig.tight_layout(pad=0.3)
        self.canvas.draw_idle()

    def update_volume(self, bv, av):
        self.ax.clear()
        self.ax.set_facecolor(BG_PANEL)
        total = bv + av
        bp = (bv / total * 100) if total > 0 else 50
        ap = 100 - bp

        bars = self.ax.bar(["Buy", "Sell"], [bv, av],
                            color=[GREEN, RED], edgecolor="none", width=0.5)
        peak = max(bv, av, 1)
        for bar, val, pct in zip(bars, [bv, av], [bp, ap]):
            self.ax.text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + peak * 0.03,
                         f"{val:,.0f} ({pct:.0f}%)", ha="center", va="bottom",
                         color=FG, fontsize=7.5, fontweight="bold")

        self.ax.set_title(f"{self.ticker} Book Volume", fontsize=8,
                           color=FG_DIM, fontweight="bold", pad=2)
        self.ax.tick_params(colors=FG_DIM, labelsize=7, length=0)
        for sp in self.ax.spines.values():
            sp.set_visible(False)
        self.fig.tight_layout(pad=0.4)
        self.canvas.draw_idle()

    # ── Market ───────────────────────────────────────────────────────────

    def update_market(self, price, bid, ask, position):
        self.last_price = price
        self.bid_price = bid
        self.ask_price = ask
        self.position = position

        self.lbl_price.config(text=f"${price:.2f}" if price else "—")
        fg = CYAN if position > 0 else (RED if position < 0 else FG)
        self.lbl_pos.config(text=f"Pos: {position:,.0f}", fg=fg)

        # Update trade button direction (but NEVER touch qty_entry)
        if position > 0:
            self.btn_trade.config(text="SELL", bg="#b62324", fg="white", state="normal")
        elif position < 0:
            self.btn_trade.config(text="BUY", bg="#1a7f37", fg="white", state="normal")
        else:
            self.btn_trade.config(text="—", bg="#21262d", fg=FG_DIM, state="disabled")

        self.btn_exit.config(state="normal" if position != 0 else "disabled")

        if self.tender:
            self._update_pnl()

    # ── Tender ───────────────────────────────────────────────────────────

    def update_tender(self, tender, current_tick, tender_count=1):
        self.current_tick = current_tick
        self._tender_count = tender_count
        self.tender = tender

        # Multi-tender warning
        if tender_count > 1:
            self.lbl_multi.config(text=f"  +{tender_count - 1} ")
        else:
            self.lbl_multi.config(text="")

        if not tender:
            self.lbl_ttype.config(text="NO ACTIVE TENDER", fg=FG_DIM)
            self.lbl_tinfo.config(text="")
            self.lbl_pnl.config(text="", bg=BG_CARD)
            self.lbl_timer.config(text="")
            self.lbl_rec.config(text="")
            self._hide_wta()
            self._wta_price_set = False
            self.btn_accept.config(state="disabled", bg="#21262d", fg=FG_DIM)
            self.btn_reject.config(state="disabled")
            return

        caption  = str(tender.get("caption", "") or "")
        action   = str(tender.get("action", "") or "").upper()
        qty      = tender.get("quantity", 0) or 0
        price    = tender.get("price", None)
        expires  = tender.get("expires", 0) or 0
        is_fixed = tender.get("is_fixed_bid", True)

        cap_low = caption.lower()
        is_wta = "winner" in cap_low or "take all" in cap_low or "take-all" in cap_low
        is_comp = "competitive" in cap_low or "auction" in cap_low
        is_special = is_wta or is_comp or not is_fixed

        if is_wta:
            ttype = "WINNER-TAKE-ALL"
        elif is_comp:
            ttype = "COMPETITIVE AUCTION"
        else:
            ttype = "PRIVATE TENDER"
        self.lbl_ttype.config(text=ttype, fg=YELLOW if is_special else BLUE)

        if action == "BUY":
            dir_text = "You BUY (long)"
            self._exit_action = "SELL"
        else:
            dir_text = "You SELL (short)"
            self._exit_action = "BUY"

        price_str = f"${price:.2f}" if price is not None else "—"
        self.lbl_tinfo.config(
            text=f"{dir_text}   Qty: {qty:,.0f}   Price: {price_str}")

        remaining = max(0, int(expires - current_tick))
        self.lbl_timer.config(text=f"{remaining}s remaining")

        # WTA/Competitive: show price entry
        if is_special:
            self._show_wta()
            lbl = "Submit Bid" if action == "BUY" else "Submit Offer"
            self.btn_wta_submit.config(text=lbl, state="normal")
            if self.bid_price and self.ask_price:
                self._show_rec(action, is_wta)
                # Only auto-fill price once per tender
                if not self._wta_price_set:
                    c = self.comm
                    if action == "BUY":
                        self.wta_price.set_val(round(self.bid_price - c - 0.01, 2))
                    else:
                        self.wta_price.set_val(round(self.ask_price + c + 0.01, 2))
                    self._wta_price_set = True
        else:
            self._hide_wta()
            self.lbl_rec.config(text="")

        self._update_pnl()

        # ALWAYS enable accept and reject
        self.btn_accept.config(state="normal", bg="#238636", fg="white")
        self.btn_reject.config(state="normal")

    def _show_rec(self, action, is_wta):
        bid, ask, c = self.bid_price, self.ask_price, self.comm
        if action == "BUY":
            be = bid - c
            tip = f"Breakeven: ${be:.2f}  (bid - comm)" + \
                  ("  |  Highest bid wins" if is_wta else "  |  Your price if > reserve")
        else:
            be = ask + c
            tip = f"Breakeven: ${be:.2f}  (ask + comm)" + \
                  ("  |  Lowest offer wins" if is_wta else "  |  Your price if > reserve")
        self.lbl_rec.config(text=tip)

    def _update_pnl(self):
        t = self.tender
        if not t:
            return
        action   = str(t.get("action", "") or "").upper()
        qty      = t.get("quantity", 0) or 0
        price    = t.get("price", None)
        is_fixed = t.get("is_fixed_bid", True)

        if not is_fixed or price is None:
            self.lbl_pnl.config(text="", bg=BG_CARD)
            return

        bid, ask = self.bid_price, self.ask_price
        if not bid or not ask:
            self.lbl_pnl.config(text="…", fg=FG_DIM, bg=BG_CARD)
            return

        c = self.comm
        pnl = round(((bid - price) * qty - c * qty) if action == "BUY"
                     else ((price - ask) * qty - c * qty), 2)

        if pnl > 0:
            self.lbl_pnl.config(text=f"${pnl:+,.0f}", fg="white", bg=GREEN)
        elif pnl < 0:
            self.lbl_pnl.config(text=f"${pnl:+,.0f}", fg="white", bg=RED)
        else:
            self.lbl_pnl.config(text=f"${pnl:+,.0f}", fg="black", bg=YELLOW)

    # ── Actions ──────────────────────────────────────────────────────────

    def _tid(self):
        return (self.tender.get("tender_id") or self.tender.get("id")) \
               if self.tender else None

    def _accept(self):
        tid = self._tid()
        if tid is None:
            return
        self.btn_accept.config(state="disabled", text="ACCEPTING…")

        def do():
            t = self.tender
            is_fixed = t.get("is_fixed_bid", True) if t else True
            price = t.get("price") if t else None
            r = self.rit.accept_tender(tid) if is_fixed else \
                self.rit.accept_tender(tid, price=price)
            def ui():
                ok = r and not r.get("_error")
                self.lbl_ttype.config(text="ACCEPTED" if ok else "FAILED",
                                       fg=GREEN if ok else RED)
                self.btn_accept.config(state="normal", text="ACCEPT")
            self.after(0, ui)

        threading.Thread(target=do, daemon=True).start()

    def _submit_wta(self):
        tid = self._tid()
        if tid is None:
            return
        price = self.wta_price.get_float()
        if price <= 0:
            return
        self.btn_wta_submit.config(state="disabled", text="…")

        def do():
            r = self.rit.accept_tender(tid, price=price)
            def ui():
                ok = r and not r.get("_error")
                self.lbl_ttype.config(
                    text=f"SUBMITTED @ ${price:.2f}" if ok else "FAILED",
                    fg=GREEN if ok else RED)
                # Restore button
                action = str(self.tender.get("action", "") or "").upper() if self.tender else ""
                lbl = "Submit Bid" if action == "BUY" else "Submit Offer"
                self.btn_wta_submit.config(state="normal", text=lbl)
            self.after(0, ui)

        threading.Thread(target=do, daemon=True).start()

    def _reject(self):
        tid = self._tid()
        if tid is None:
            return
        def do():
            ok = self.rit.decline_tender(tid)
            self.after(0, lambda: self.lbl_ttype.config(
                text="DECLINED" if ok else "REJECT FAILED",
                fg=FG_DIM if ok else RED))
        threading.Thread(target=do, daemon=True).start()

    def _manual_trade(self):
        qty = self.qty_entry.get_int()
        if qty <= 0:
            return
        action = self.btn_trade.cget("text")
        if action not in ("BUY", "SELL"):
            return
        self.btn_trade.config(state="disabled")
        def do():
            self.rit.market_order(self.ticker, qty, action)
            self.after(0, lambda: self.btn_trade.config(state="normal"))
        threading.Thread(target=do, daemon=True).start()

    def _exit_position(self):
        pos = self.position
        if pos == 0:
            return
        action = "SELL" if pos > 0 else "BUY"
        qty = abs(int(pos))
        self.btn_exit.config(state="disabled", text="EXITING…")
        def do():
            self.rit.market_order(self.ticker, qty, action)
            self.after(0, lambda: self.btn_exit.config(
                state="normal", text="EXIT POSITION"))
        threading.Thread(target=do, daemon=True).start()


# ─── APP ─────────────────────────────────────────────────────────────────
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("RITC 2026 — LT3 Tender Tool")
        self.root.geometry("1400x800")
        self.root.minsize(700, 400)
        self.root.configure(bg=BG)

        self.rit = RIT()
        self.panels = {}
        self.tickers = []
        self.running = True
        self.zoom = 100
        self._last_book = 0

        self._build_topbar()
        self.grid_fr = tk.Frame(self.root, bg=BG)
        self.grid_fr.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        for key, d in [("<Control-plus>", 10), ("<Control-equal>", 10),
                        ("<Control-minus>", -10), ("<Control-KP_Add>", 10),
                        ("<Control-KP_Subtract>", -10)]:
            self.root.bind(key, lambda e, _d=d: self._do_zoom(_d))
        self._poll()

    def _build_topbar(self):
        bar = tk.Frame(self.root, bg=BG, height=36)
        bar.pack(fill="x", padx=8, pady=(8, 4))
        bar.pack_propagate(False)

        tk.Label(bar, text="LT3 TENDER TOOL", font=(FONT, 11, "bold"),
                 fg=FG_DIM, bg=BG).pack(side="left")

        self.lbl_status = tk.Label(bar, text="●", font=(FONT, 10),
                                    fg=RED, bg=BG)
        self.lbl_status.pack(side="right", padx=(8, 0))

        self.lbl_case = tk.Label(bar, text="", font=(MONO, 8),
                                  fg=FG_DIM, bg=BG)
        self.lbl_case.pack(side="right", padx=8)

        self.lbl_pnl = tk.Label(bar, text="—", font=(MONO, 12, "bold"),
                                 fg=CYAN, bg=BG)
        self.lbl_pnl.pack(side="right", padx=(16, 8))
        tk.Label(bar, text="P&L", font=(FONT, 9), fg=FG_DIM,
                 bg=BG).pack(side="right")

    def _do_zoom(self, delta):
        self.zoom = max(60, min(200, self.zoom + delta))
        s = self.zoom / 100
        for p in self.panels.values():
            p.lbl_tick.config(font=(MONO, int(16 * s), "bold"))
            p.lbl_price.config(font=(MONO, int(13 * s)))
            p.lbl_pos.config(font=(MONO, int(11 * s)))
            p.fig.set_dpi(max(int(90 * s), 50))
            p.canvas.draw_idle()

    def _layout(self):
        n = len(self.panels)
        if n == 0:
            return
        w = self.grid_fr.winfo_width()
        if w < 1:
            w = self.root.winfo_width()
        cols = 1 if w < 600 else (min(n, 2) if w < 1100 or n <= 2 else min(n, 3))

        for p in self.panels.values():
            p.grid_forget()
        for i, (_, p) in enumerate(self.panels.items()):
            p.grid(row=i // cols, column=i % cols, sticky="nsew", padx=4, pady=4)
        for c in range(cols):
            self.grid_fr.columnconfigure(c, weight=1)
        for r in range(math.ceil(n / cols)):
            self.grid_fr.rowconfigure(r, weight=1)

    def _poll(self):
        if not self.running:
            return

        def fetch():
            try:
                case = self.rit.case()
                if not case:
                    self.root.after(0, lambda: self.lbl_status.config(fg=RED))
                    return
                self.root.after(0, lambda: self.lbl_status.config(fg=GREEN))

                tick = case.get("tick", 0) or 0
                period = case.get("period", 0)
                status = case.get("status", "")
                name = case.get("name", "")
                self.root.after(0, lambda: self.lbl_case.config(
                    text=f"{name}  P{period}  t={tick}  {status}"))

                trader = self.rit.trader()
                if trader:
                    nlv = trader.get("nlv", 0) or 0
                    fg = CYAN if nlv >= 0 else RED
                    self.root.after(0, lambda: self.lbl_pnl.config(
                        text=f"${nlv:+,.0f}", fg=fg))

                secs = self.rit.securities()
                if not secs:
                    return

                cur = [s["ticker"] for s in secs if s.get("ticker")]
                if set(cur) != set(self.tickers):
                    self.tickers = cur
                    self.root.after(0, lambda: self._rebuild(secs))

                for sec in secs:
                    t = sec.get("ticker", "")
                    if t not in self.panels:
                        continue
                    price = sec.get("last") or sec.get("bid") or sec.get("ask")
                    bid = sec.get("bid")
                    ask = sec.get("ask")
                    pos = sec.get("position", 0) or 0
                    self.root.after(0, lambda _t=t, _p=price, _b=bid, _a=ask, _pos=pos:
                                    self.panels[_t].update_market(
                                        float(_p) if _p else None,
                                        float(_b) if _b else None,
                                        float(_a) if _a else None,
                                        _pos))

                now = time.time()
                if now - self._last_book > 0.5:
                    self._last_book = now
                    for t in self.tickers:
                        if t not in self.panels:
                            continue
                        book = self.rit.book(t)
                        if book:
                            bv = sum(b.get("quantity", 0) or 0
                                     for b in book.get("bids", []))
                            av = sum(a.get("quantity", 0) or 0
                                     for a in book.get("asks", []))
                            self.root.after(0, lambda _t=t, _bv=bv, _av=av:
                                            self.panels[_t].update_volume(_bv, _av))

                # Tenders — count per ticker for multi-tender warning
                tenders = self.rit.tenders()
                by_ticker = {}       # ticker -> first tender
                count_by_ticker = {} # ticker -> count
                if tenders:
                    for td in tenders:
                        tt = td.get("ticker", "")
                        if tt:
                            if tt not in by_ticker:
                                by_ticker[tt] = td
                            count_by_ticker[tt] = count_by_ticker.get(tt, 0) + 1

                for t in self.tickers:
                    if t in self.panels:
                        td = by_ticker.get(t)
                        cnt = count_by_ticker.get(t, 0)
                        self.root.after(0, lambda _t=t, _td=td, _tick=tick, _cnt=cnt:
                                        self.panels[_t].update_tender(_td, _tick, _cnt))

            except Exception:
                traceback.print_exc()

        threading.Thread(target=fetch, daemon=True).start()
        self.root.after(POLL_MS, self._poll)

    def _rebuild(self, secs):
        for p in self.panels.values():
            p.destroy()
        self.panels.clear()
        for w in self.grid_fr.winfo_children():
            w.destroy()
        for sec in secs:
            t = sec.get("ticker", "")
            if not t:
                continue
            self.panels[t] = SecurityPanel(self.grid_fr, t, self.rit)
        self._layout()
        self.root.bind("<Configure>", lambda e: self._layout())

    def close(self):
        self.running = False
        self.root.destroy()


def main():
    root = tk.Tk()
    app = App(root)
    root.protocol("WM_DELETE_WINDOW", app.close)
    root.mainloop()

if __name__ == "__main__":
    main()