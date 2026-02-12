import tkinter as tk
from tkinter import ttk, font as tkfont
import threading
import time
import argparse
import re
import concurrent.futures

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

# ═══════════════════════════════════════════════
#  CONFIGURATION & CONSTANTS
# ═══════════════════════════════════════════════
API_BASE_URL = "http://localhost:9999/v1"
MAX_ORDER_SIZE = 5000       # RIT Server limit per order
TARGET_LIMIT = 50000        # Case Limit (+/- 50,000)
MAX_WORKERS = 20            # Parallel execution threads

# --- COLORS (Dark Theme) ---
COL_BG       = "#0d1117"
COL_CARD     = "#161b22"
COL_BORDER   = "#30363d"
COL_TEXT     = "#c9d1d9"
COL_DIM      = "#8b949e"
COL_WHITE    = "#f0f6fc"
COL_GREEN    = "#238636"
COL_RED      = "#da3633"
COL_YELLOW   = "#d29922"
COL_BTN_BLUE = "#2196F3"    # Buy Max
COL_BTN_RED  = "#D32F2F"    # Sell Max
COL_BTN_FLAT = "#2ea043"    # Flatten (Green)

# ═══════════════════════════════════════════════
#  DATA: HEADLINES & DEALS (From Your Logic)
# ═══════════════════════════════════════════════
HEADLINE_LOOKUP = {
    "Activist Fund Builds ByteLayer Stake": -0.14, "Activist Hedge Fund Opposes Deal": -0.05,
    "All Financing Conditions Met": -0.06, "All-Stock Structure Eliminates Financing": 0.1033,
    "Antitrust Experts Raise Competition Concerns": 0.0925, "Asset Quality Deterioration Noted": -0.045,
    "Asset Sale Proceeds Exceed Target": 0.01, "Asset Sales to Fund Transaction": -0.05,
    "Atlas Bank Increases Offer Price": -0.01, "Atlas Bank Management Provides Updated Market Outlook": -0.045,
    "Atlas Shareholder Concerns": 0.045, "Bank Credit Spreads Widen Sharply": -0.0367,
    "Bank Secrecy Act Compliance Review": -0.045, "Bank Syndicate Expands": 0.0425,
    "Banks Propose Enhanced CRA Plan": -0.06, "Board Issues Supplemental Disclosure": 0.2433,
    "Board Releases Detailed Fairness Analysis": -0.015, "Bond Market Volatility Concerns": -0.054,
    "Bondholder Litigation Filed": -0.0325, "Bondholder Litigation Resolved": 0.05,
    "Branch Manager Retention Program": -0.0625, "Breach of Transaction Covenant Alleged": -0.1267,
    "Breakup Fee Structure Detailed": -0.565, "Bridge Financing Commitment Signed": 0.01,
    "Bridge Loan Replaced with Term Facility": -0.02, "ByteLayer Engages Advisors for Strategic Review": 0.065,
    "ByteLayer Sets Shareholder Meeting Date": 0.035, "CFIUS National Security Review Initiated": -1.66,
    "Canadian Regulators Provide Clearance": -0.05, "Capital Plan Approved by Fed": -0.0043,
    "Chinese SAMR Extends Review Period": -0.03, "Class Action Filed in Delaware": -0.08,
    "Closed-Door Meeting Between Transaction Principals Reported": 0.0121, "Closing Conditions Status": 0.0,
    "Closing Conditions Status Update": 0.0025, "Closing Conditions Tracker Updated": -0.0229,
    "Closing Timeline Extended": 0.0533, "CloudSys Confirms No MAE": 0.0, "CloudSys Improves Exchange Ratio": 0.0,
    "CloudSys Management Provides Updated Market Outlook": -0.1367, "CloudSys Reports Strong Revenue Growth": 0.0,
    "CloudSys Shareholder Questions Deal Logic": -0.02, "Collar Mechanism Speculation": 0.0071,
    "Commodity Price Hedge Program": 0.0233, "Community Groups Raise CRA Concerns": -0.0225,
    "Companies Commit to Capacity Expansion": -0.16, "Companies Commit to Renewable Investment": 0.03,
    "Companies Propose Behavioral Remedies": 0.0, "Company Offers Divestiture Package": -0.04,
    "Competing Bank Identified": -0.03, "Conditions Checklist Published": -0.04, "Confidential Approach Disclosed": 0.02,
    "Congressional Banking Committee Inquiry": -0.035, "Congressional Hearing Announced": 0.11,
    "Congressional Tech Oversight Hearing": -0.1275, "Consumer Advocacy Groups File Opposition": 0.0325,
    "Convertible Debt Refinanced": 0.0, "Convertible Note Holders Seek Clarity": 0.065,
    "Counsel Provides Updated Risk Factor Disclosure": 0.0388, "Credit Analysts Review CloudSys Debt Profile": -0.0233,
    "Credit Facility Amendment Negotiated": 0.022, "Credit Rating Agencies Place PHR on Watch": 0.05,
    "Credit Rating Downgrade": -0.0275, "DOJ Antitrust Clearance Granted": 0.06,
    "DOJ Antitrust Division Signals Concerns": -0.08, "DOJ Provides Conditional Clearance": 0.0,
    "DOJ Requests Extended Review": 0.05, "Debt Financing Commitments Secured": 0.0333,
    "Debt Financing Fully Committed": 0.04, "Definitive Agreement Announced": -0.098,
    "Definitive Agreement Executed": 0.3133, "Deposit Outflows Accelerate": -0.055,
    "Derivative Litigation Dismissed": 0.11, "Derivative Litigation Settled": 0.17,
    "Developer Retention Incentives": -0.105, "Dividend Policy Maintained": -0.0283,
    "Dividend Suspension Announced": 0.06, "EPA Issues Supportive Statement": 0.0,
    "ESG-Focused Fund Opposes Terms": -0.0857, "EU Commission Requests Detailed Market Share Data": -0.06,
    "Early Indications Point to Strong Shareholder Support": 0.0, "Early Vote Results Show Strong Support": 0.005,
    "Earnings Shortfall Raises MAC Questions": 0.04, "EastEnergy Improves Exchange Ratio": 0.0,
    "EastEnergy Management Provides Updated Market Outlook": 0.0, "EastEnergy Share Price Volatility": -0.08,
    "EastEnergy Shareholder Questions": -0.37, "EastEnergy Waives MAC": 0.08,
    "Employee Retention Packages Offered": 0.0375, "Energy Sector Credit Spread Widening": -0.0271,
    "Environmental Groups Challenge Merger": 0.0567, "Environmental Groups Express Support": 0.06,
    "Equity Co-Investment Secured": 0.0, "Equity Research Downgrades CloudSys": 0.27,
    "European Commission Opens Phase II Investigation": 0.04, "Exchange Ratio Collar Discussed": 0.0,
    "Expected Closing Date Pushed Back": -0.035, "Expected Timeline Revised": 0.1833,
    "FDIC Extends Comment Period": 0.005, "FERC Filing for Renewable Merger": 0.0,
    "FERC Grants Expedited Approval": -0.17, "FERC Issues Order Approving Transaction": 0.0367,
    "FERC Notification Filed": -0.034, "FERC Review Initiated for Energy Deal": -0.0517,
    "FERC Staff Holds Technical Working Session": -0.0211, "FTC Clears Transaction - Early Termination": 0.0,
    "FTC Commissioner Dissent Published": -0.355, "FTC Commissioner Issues Support Statement": 0.0,
    "FTC Opens Preliminary Review of Tech Merger": 0.0, "FTC Requests Additional Information": -0.0075,
    "FTC Staff Holds Technical Working Session": 0.1825, "Fed Announces Enhanced Review Process": 0.0233,
    "Fed Governor Issues Dissenting Opinion": -0.09, "Federal Reserve Approves Transaction": -0.1667,
    "Federal Reserve Staff Holds Technical Working Session": 0.0614, "Federal Tax Credit Clarification Sought": -0.0275,
    "FinSure Board Defends Transaction": -0.02, "FinSure Engages Advisors for Strategic Review": -0.07,
    "FinSure Sets Meeting for Shareholder Vote": -0.0383, "Final Closing Conditions Checklist Published": 0.0433,
    "Financing Conditions Satisfied": -0.0333, "Force Majeure Event Raises Transaction Concerns": -0.27,
    "Glass Lewis Recommends Approval": -0.295, "GreenGrid Engages Advisors for Strategic Review": -0.025,
    "GreenGrid Schedules Shareholder Vote": 0.102, "Growing Market Speculation Surrounds Deal Viability": -0.77,
    "HSR Filing Made": -0.0525, "Hart-Scott-Rodino Filing Submitted": 0.38, "Hedge Fund Activist Opposes Deal": 0.195,
    "ISO-NE Issues Favorable Opinion": 0.0, "ISS Issues Qualified Support": -0.02,
    "ISS Strongly Recommends Approval": 0.0, "Improved Financial Disclosures": 0.086,
    "Independent Fairness Review Generates Board Discussion": 0.0427, "Index Rebalancing Implications": -0.1825,
    "Industry Conference Generates Transaction Speculation": -0.0944, "Industry Lobbyists Support Transaction": 0.15,
    "Infrastructure Consortium Identified": -0.26, "Infrastructure Fund Opposes Terms": 0.0367,
    "Infrastructure Modernization Requirements": 0.06, "Institutional Investors Signal Support": 0.062,
    "Integration Planning Commences": -0.038, "Integration Planning Progresses": -0.055,
    "Integration Planning Underway": -0.025, "Integration Teams Appointed": -0.2533,
    "Integration Teams Formed": 0.03, "Interconnection Rights Confirmed": -0.1475,
    "Interest Rate Hedge Executed": 0.015, "Investment Grade Bond Offering Completed": 0.068,
    "Investment Grade Rating Affirmed": -0.1375, "Key Employee Retention Agreements": -0.022,
    "Key Engineer Retention Plan Announced": 0.0033, "Legal Challenge Dismissed": -0.07,
    "Market Cap Concerns Emerge": 0.0, "Material Adverse Change Clause Concern": -0.375,
    "Material Adverse Effect Question Raised": -0.32, "Material Breach of Merger Covenant Alleged": 0.02,
    "Merger Agreement Executed": 0.005, "Merger Agreement Signed": 0.0,
    "Merger Agreement Signed and Announced": -0.1167, "Minority Shareholder Litigation": -0.115,
    "Moody's Places Ratings Under Review": -0.005, "Non-Binding Indication Received": -0.1733,
    "OCC Raises Operational Risk Concerns": 0.045, "Oil Price Collapse Raises Questions": 0.04,
    "Outside Date Extended": 0.0043, "Outside Date Extended to Year-End": -0.13,
    "Permanent Financing Syndication Successful": -0.045, "PetroNorth Investor Relations Concerns": 0.05,
    "PetroNorth Management Provides Updated Market Outlook": 0.505, "PetroNorth Waives MAC Concerns": 0.1,
    "Pharmaco Increases Bid to $52.50": -0.02, "Pharmaco Management Provides Updated Market Outlook": 0.045,
    "Pharmaco Reports Strong Quarterly Earnings": 0.035, "Pharmaco Shareholder Backlash": -0.0533,
    "Pharmaco Stock Price Decline": -0.05, "Pipeline Safety Review Announced": -0.0,
    "Political Opposition Emerges From Fossil Fuel-Dependent States": 0.0, "Political Opposition Voiced": 0.035,
    "Potential Competing Interest Emerges": -0.12, "Preferred Stock Issuance": 0.01,
    "Preliminary Approach Confirmed": 0.09, "Private Equity Interest Reported": 0.14,
    "Project Delays Raise MAC Concerns": 0.215, "Project Finance Commitment Secured": 0.0,
    "Proxy Advisory Firm ISS Recommends Approval": -0.01, "Proxy Advisory Firms Issue Divided Recommendations": -0.024,
    "Rate Case Proceedings Initiated": -0.2525, "Regulatory Applications Filed": 0.005,
    "Regulatory Filings Submitted": -0.025, "Regulatory Timeline Extended": -0.0967,
    "Regulatory Timeline Slippage": 0.0267, "Renewable Energy Credit Transfer Review": 0.0167,
    "Renewable Energy Sector Sell-Off": -0.0433, "Reverse Termination Fee Increased": 0.0425,
    "Reverse Termination Fee Triggered": -0.01, "Rumor of Competing Interest": -0.07,
    "Sector Multiple Compression": -0.0475, "Sell-Side Analysts Issue Divergent Valuation Assessments": -0.0247,
    "Senior Notes Offering Successful": 0.0136, "Settlement of Shareholder Litigation": -0.04,
    "Share Buyback Program Announced": 0.1975, "Shareholder Derivative Suit Filed": -0.03,
    "Shareholder Litigation Filed": -0.0125, "Shareholder Litigation Settled": 0.01,
    "SolarPeak Engages Advisors for Strategic Review": 0.055, "SolarPeak Schedules Shareholder Meeting": 0.044,
    "Standstill Agreement Breached": -0.0333, "State Banking Regulators Provide Clearance": 0.0614,
    "State Commission Accepts Commitments": -0.28, "State Regulators Express Concerns": 0.05,
    "State Renewable Energy Review": 0.0367, "State Utility Commissions Request Hearings": -0.0533,
    "Stock Market Volatility Affects Exchange Ratio": -0.23, "Stock Price Conditions Confirmed Met": 0.04,
    "Stock Price Conditions Satisfied": -0.3, "Strategic Investor Shows Interest": -0.43,
    "Strategic Investor Takes Stake": -0.35, "Stress Test Requirements Imposed": 0.09,
    "Strong Project Pipeline Announced": 0.0525, "Strong Quarterly Results Reported": -0.0225,
    "Successful Bond Offering": -0.026, "Superior Proposal at $43 Per Share": -0.01,
    "Supplemental Environmental Review": -0.01, "Syndication Achieves Oversubscription": 0.024,
    "TGX Receives Unsolicited Inquiry": -0.03, "TLAC Requirements Clarified": 0.09,
    "Targenix Board Unanimously Reaffirms Support": -0.0533, "Targenix Engages Advisors for Strategic Review": -0.03,
    "Targenix Shareholder Meeting Scheduled": 0.026, "Tax Credit Extension Uncertainty": 0.185,
    "Tech Industry Coalition Opposes Deal": -0.1429, "Termination Fee Set at $240M": -0.0475,
    "Termination Fee Structure": 0.0, "Termination Fee Structure Disclosed": -0.0733,
    "Third-Party Regulatory Analysis Published": -0.0914, "Timeline Adjustment": -0.05,
    "Timeline Extended for Regulatory Process": -0.1, "Transmission Access Questions Raised": 0.24,
    "UK CMA Phase I Review Completed": 0.2, "UK CMA Provisionally Clears Deal": 0.0,
    "Unsolicited Inquiry Received": 0.0, "Unusual Institutional Trading Activity Observed in BYL": 0.0,
    "Unusual Institutional Trading Activity Observed in FSR": -0.055, "Unusual Institutional Trading Activity Observed in GGD": 0.008,
    "Unusual Institutional Trading Activity Observed in SPK": 0.0, "Unusual Institutional Trading Activity Observed in TGX": -0.008,
    "Utility Company Interest Reported": 0.0, "Vote Outcome Appears Favorable": -0.05,
    "Vote Tracking Shows Majority Support": 0.0,
}

MARKET_WIDE = {
    "Broad Market Sell-Off", "Central Bank Announces", "Credit Markets Flash",
    "DOJ Antitrust Division Signals", "Equity Markets Rally", "Escalating Geopolitical Tensions",
    "Executive Order Targets", "Federal Reserve Announces", "Investment Banking Activity",
    "Major Rating Agency Revises", "Major Transaction in Adjacent", "Mixed Economic Data",
    "Quarterly Institutional", "Senate Committee Launches", "Trade Policy Revisions"
}

# Deal Data
TICKER_TO_DEAL = {
    "TGX": "D1", "PHR": "D1", "BYL": "D2", "CLD": "D2",
    "GGD": "D3", "PNR": "D3", "FSR": "D4", "ATB": "D4",
    "SPK": "D5", "EEC": "D5",
    "Targenix": "D1", "Pharmaco": "D1", "ByteLayer": "D2", "CloudSys": "D2",
    "GreenGrid": "D3", "PetroNorth": "D3", "FinSure": "D4", "Atlas": "D4",
    "SolarPeak": "D5", "EastEnergy": "D5"
}

DEAL_CONFIGS = [
    {"id": "D1", "target": "TGX", "acquirer": "PHR", "sector": "Pharma", "type": "Cash",  "cash": 50.0, "ratio": 0.0,  "iv_tgt": 43.70, "iv_acq": 47.50, "p": 0.70},
    {"id": "D2", "target": "BYL", "acquirer": "CLD", "sector": "Tech",   "type": "Stock", "cash": 0.0,  "ratio": 0.75, "iv_tgt": 43.50, "iv_acq": 79.30, "p": 0.55},
    {"id": "D3", "target": "GGD", "acquirer": "PNR", "sector": "Energy", "type": "Mixed", "cash": 33.0, "ratio": 0.20, "iv_tgt": 31.50, "iv_acq": 59.80, "p": 0.50},
    {"id": "D4", "target": "FSR", "acquirer": "ATB", "sector": "Banking","type": "Cash",  "cash": 40.0, "ratio": 0.0,  "iv_tgt": 30.50, "iv_acq": 62.20, "p": 0.38},
    {"id": "D5", "target": "SPK", "acquirer": "EEC", "sector": "Renew",  "type": "Stock", "cash": 0.0,  "ratio": 1.20, "iv_tgt": 52.80, "iv_acq": 48.00, "p": 0.45},
]

# ═══════════════════════════════════════════════
#  LOGIC: DEAL ANALYTICS
# ═══════════════════════════════════════════════
class Deal:
    def __init__(self, cfg):
        self.cfg = cfg
        self.id = cfg['id']
        self.target = cfg['target']
        self.acquirer = cfg['acquirer']
        self.prob = cfg['p']
        
        # Prices & Stats
        self.tgt_price = cfg['iv_tgt']
        self.acq_price = cfg['iv_acq']
        self.news_count = 0
        self.cum_impact = 0.0
        self.headline_dp = 0.0
        
        # Calc Initials
        self.reset_calc()

    def reset_calc(self):
        # Calculate Implied Standalone (V) based on initial prices
        K = self.deal_value()
        p0 = self.cfg['p']
        denom = 1 - p0 if p0 < 1.0 else 0.01
        self.standalone = (self.cfg['iv_tgt'] - p0 * K) / denom
        self.calc_intrinsic()

    def deal_value(self):
        return self.cfg['cash'] + (self.cfg['ratio'] * self.acq_price)

    def calc_intrinsic(self):
        K = self.deal_value()
        # Intrinsic = p * K + (1-p) * S
        # We allow probability to shift based on news flow (cum_impact converted to prob)
        
        # We treat 'headline_dp' as the unabsorbed probability shift from news
        current_p = max(0.0, min(1.0, self.prob + self.headline_dp))
        
        self.intrinsic = (current_p * K) + ((1 - current_p) * self.standalone)
        return self.intrinsic

    def mispricing(self):
        return self.intrinsic - self.tgt_price

def identify_deal(headline):
    for tok in re.findall(r'[A-Z]{2,4}', headline):
        if tok in TICKER_TO_DEAL: return TICKER_TO_DEAL[tok]
    for name in ["Targenix", "Pharmaco", "ByteLayer", "CloudSys", "GreenGrid", 
                 "PetroNorth", "FinSure", "Atlas", "SolarPeak", "EastEnergy"]:
        if name in headline: return TICKER_TO_DEAL[name]
    return None

# ═══════════════════════════════════════════════
#  GUI & APP
# ═══════════════════════════════════════════════
class MergerDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("RITC 2026 — Merger Arb Dashboard")
        self.root.geometry("1200x800")
        self.root.configure(bg=COL_BG)
        
        self.deals = {d['id']: Deal(d) for d in DEAL_CONFIGS}
        self.api_key = ""
        self.polling = False
        self.tick = 0
        self.last_nid = 0
        self.seen_nids = set()
        
        # Fonts
        self.fn_header = tkfont.Font(family="Roboto", size=14, weight="bold")
        self.fn_card_title = tkfont.Font(family="Consolas", size=11, weight="bold")
        self.fn_price = tkfont.Font(family="Consolas", size=13, weight="bold")
        self.fn_small = tkfont.Font(family="Consolas", size=9)
        self.fn_btn = tkfont.Font(family="Arial", size=10, weight="bold")

        self._build_layout()

    def _build_layout(self):
        # --- HEADER ---
        header = tk.Frame(self.root, bg=COL_BG)
        header.pack(fill="x", padx=15, pady=10)
        tk.Label(header, text="MERGER ARB", fg=COL_WHITE, bg=COL_BG, font=self.fn_header).pack(side="left")
        tk.Label(header, text=" RITC 2026", fg=COL_DIM, bg=COL_BG, font=self.fn_header).pack(side="left")
        
        self.lbl_status = tk.Label(header, text="● DISCONNECTED", fg=COL_RED, bg=COL_BG, font=self.fn_small)
        self.lbl_status.pack(side="right", padx=10)
        self.lbl_tick = tk.Label(header, text="TICK 0", fg=COL_DIM, bg=COL_BG, font=self.fn_small)
        self.lbl_tick.pack(side="right")

        # --- CONTROLS ---
        controls = tk.Frame(self.root, bg=COL_BG)
        controls.pack(fill="x", padx=15, pady=5)
        tk.Label(controls, text="API Key:", fg=COL_DIM, bg=COL_BG).pack(side="left")
        self.ent_key = tk.Entry(controls, bg=COL_CARD, fg=COL_WHITE, insertbackground="white", bd=0, width=25)
        self.ent_key.pack(side="left", padx=10, ipady=3)
        self.btn_connect = tk.Button(controls, text="CONNECT", bg=COL_GREEN, fg="white", bd=0, padx=10, command=self.toggle_connect)
        self.btn_connect.pack(side="left")

        # --- DEAL CARDS ---
        self.card_container = tk.Frame(self.root, bg=COL_BG)
        self.card_container.pack(fill="x", padx=10, pady=10)
        self.card_container.grid_columnconfigure((0,1,2,3,4), weight=1)
        
        self.card_widgets = {}
        for i, did in enumerate(["D1", "D2", "D3", "D4", "D5"]):
            self._create_card(i, did)

        # --- NEWS FEED ---
        tk.Label(self.root, text="NEWS FEED", fg=COL_DIM, bg=COL_BG, font=self.fn_small, anchor="w").pack(fill="x", padx=15, pady=(15,0))
        self.news_list = tk.Text(self.root, bg=COL_BG, fg=COL_DIM, height=12, bd=0, font=self.fn_small, state="disabled")
        self.news_list.pack(fill="both", expand=True, padx=15, pady=5)
        
        # Tags for News Colors
        self.news_list.tag_config("green", foreground=COL_GREEN)
        self.news_list.tag_config("red", foreground=COL_RED)
        self.news_list.tag_config("white", foreground=COL_WHITE)
        self.news_list.tag_config("dim", foreground=COL_DIM)

        # --- FLATTEN ALL BUTTON (GREEN - BOTTOM) ---
        btn_flat = tk.Button(self.root, text="FLATTEN ALL POSITIONS (0)", bg=COL_BTN_FLAT, fg="white", 
                             font=("Arial", 12, "bold"), bd=0, pady=10, command=self.flatten_all)
        btn_flat.pack(fill="x", side="bottom")

    def _create_card(self, col_idx, did):
        d = self.deals[did]
        card = tk.Frame(self.card_container, bg=COL_CARD, padx=2, pady=2)
        card.grid(row=0, column=col_idx, padx=5, sticky="nsew")
        
        # Header
        title_frame = tk.Frame(card, bg=COL_CARD)
        title_frame.pack(fill="x", pady=5, padx=5)
        tk.Label(title_frame, text=f"{did} {d.target}", fg=COL_WHITE, bg=COL_CARD, font=self.fn_card_title).pack(side="left")
        
        # Prices
        price_grid = tk.Frame(card, bg=COL_CARD)
        price_grid.pack(fill="x", padx=5)
        lbl_mkt = tk.Label(price_grid, text=f"${d.tgt_price:.2f}", fg=COL_WHITE, bg=COL_CARD, font=self.fn_price)
        lbl_mkt.pack(side="left")
        lbl_mis = tk.Label(price_grid, text="$+0.00", fg=COL_DIM, bg=COL_CARD, font=self.fn_small)
        lbl_mis.pack(side="right")
        
        # Stats
        stats_frame = tk.Frame(card, bg=COL_CARD)
        stats_frame.pack(fill="x", padx=5, pady=2)
        lbl_prob = tk.Label(stats_frame, text=f"P={d.prob*100:.0f}%", fg=COL_WHITE, bg=COL_CARD, font=self.fn_small)
        lbl_prob.pack(side="left")

        # --- TRADING BUTTONS ---
        btn_frame = tk.Frame(card, bg=COL_CARD, pady=5)
        btn_frame.pack(fill="x", padx=5, pady=5)
        
        # BLUE: Buy Max (+50k)
        b_buy = tk.Button(btn_frame, text="Buy Max", bg=COL_BTN_BLUE, fg="white", height=2, bd=0, font=self.fn_btn,
                          command=lambda t=d.target: self.exec_trade(t, TARGET_LIMIT))
        b_buy.pack(side="left", fill="x", expand=True, padx=(0, 2))
        
        # RED: Sell Max (-50k)
        b_sell = tk.Button(btn_frame, text="Sell Max", bg=COL_BTN_RED, fg="white", height=2, bd=0, font=self.fn_btn,
                           command=lambda t=d.target: self.exec_trade(t, -TARGET_LIMIT))
        b_sell.pack(side="right", fill="x", expand=True, padx=(2, 0))

        self.card_widgets[did] = {'mkt': lbl_mkt, 'mis': lbl_mis, 'prob': lbl_prob}

    # ==========================================
    #  TRADING ENGINE (STRICT 50K)
    # ==========================================
    def exec_trade(self, ticker, target_pos):
        threading.Thread(target=self._trade_worker, args=(ticker, target_pos)).start()

    def _trade_worker(self, ticker, target_pos):
        if not HAS_REQUESTS: return
        try:
            # 1. Get current position
            resp = requests.get(f"{API_BASE_URL}/securities", params={'ticker': ticker}, headers={'X-API-Key': self.api_key})
            if resp.status_code != 200: return
            
            data = resp.json()
            if not data: return
            curr = data[0]['position']
            
            # 2. Calculate Strict Delta
            needed = target_pos - curr
            
            if needed == 0: return
            action = "BUY" if needed > 0 else "SELL"
            qty = abs(needed)
            
            # 3. Chunk and Fire (Parallel)
            orders = []
            while qty > 0:
                chunk = min(qty, MAX_ORDER_SIZE)
                orders.append(chunk)
                qty -= chunk

            print(f"FIRING {ticker}: {action} {abs(needed)} to hit {target_pos}")
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = []
                for size in orders:
                    params = {'ticker': ticker, 'type': 'MARKET', 'quantity': size, 'action': action}
                    futures.append(executor.submit(requests.post, f"{API_BASE_URL}/orders", params=params, headers={'X-API-Key': self.api_key}))
                concurrent.futures.wait(futures)
                
        except Exception as e:
            print(f"Trade Error: {e}")

    def flatten_all(self):
        print("FLATTENING ALL...")
        tickers = []
        for d in self.deals.values():
            tickers.append(d.target)
            tickers.append(d.acquirer)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            for t in tickers:
                executor.submit(self._trade_worker, t, 0)

    # ==========================================
    #  NEWS & POLLING ENGINE
    # ==========================================
    def toggle_connect(self):
        if not self.polling:
            self.api_key = self.ent_key.get()
            self.polling = True
            self.btn_connect.config(text="STOP", bg=COL_RED)
            self.lbl_status.config(text="● CONNECTED", fg=COL_GREEN)
            threading.Thread(target=self.poll_loop, daemon=True).start()
        else:
            self.polling = False
            self.btn_connect.config(text="CONNECT", bg=COL_GREEN)
            self.lbl_status.config(text="● DISCONNECTED", fg=COL_RED)

    def poll_loop(self):
        while self.polling:
            try:
                # 1. Tick
                case = requests.get(f"{API_BASE_URL}/case", headers={'X-API-Key': self.api_key}).json()
                self.tick = case['tick']
                self.root.after(0, lambda: self.lbl_tick.config(text=f"TICK {self.tick}"))

                # 2. Prices
                secs = requests.get(f"{API_BASE_URL}/securities", headers={'X-API-Key': self.api_key}).json()
                pmap = {s['ticker']: s['last'] for s in secs}

                # 3. News Parsing
                news_items = requests.get(f"{API_BASE_URL}/news", params={'since': self.last_nid}, headers={'X-API-Key': self.api_key}).json()
                if news_items:
                    for item in news_items:
                        nid = item['news_id']
                        if nid > self.last_nid:
                            self.last_nid = nid
                            headline = item['headline']
                            # Process impact immediately
                            self.process_news_item(headline)

                # 4. Update Models
                for d in self.deals.values():
                    d.tgt_price = pmap.get(d.target, d.tgt_price)
                    d.acq_price = pmap.get(d.acquirer, d.acq_price)
                    d.calc_intrinsic()

                self.root.after(0, self.update_ui)
            except: pass
            time.sleep(0.5)

    def process_news_item(self, headline):
        impact = HEADLINE_LOOKUP.get(headline, 0.0)
        deal_id = identify_deal(headline)
        
        if deal_id and deal_id in self.deals:
            d = self.deals[deal_id]
            d.news_count += 1
            
            # Update Probability (Delta P = Impact / Spread approx)
            # This is a simplification; we accumulate raw impact for now to shift intrinsic
            # Convert dollar impact to probability shift (rough heuristic based on spread)
            K = d.deal_value()
            spread = K - d.standalone
            if abs(spread) > 0.1:
                prob_delta = impact / spread
                d.prob = max(0.0, min(1.0, d.prob + prob_delta))
            
            # Log to Feed
            self.root.after(0, self.log_news_row, self.tick, deal_id, d.target, headline, impact)
        elif headline in MARKET_WIDE:
             self.root.after(0, self.log_news_row, self.tick, "ALL", "MKT", headline, 0.0)

    def log_news_row(self, tick, did, ticker, headline, impact):
        self.news_list.config(state="normal")
        
        # Formatting
        arrow = "▲" if impact > 0 else "▼" if impact < 0 else " "
        color_tag = "green" if impact > 0 else "red" if impact < 0 else "dim"
        impact_str = f"+${impact:.2f}" if impact > 0 else f"-${abs(impact):.2f}" if impact < 0 else ""
        
        # Construct Row: t=xx  D1  TGX  ▲  Headline...  +$0.10
        row = f"t={tick:<4} {did:<3} {ticker:<8} {arrow} {headline:<45} {impact_str}\n"
        
        self.news_list.insert("1.0", row, color_tag)
        self.news_list.config(state="disabled")

    def update_ui(self):
        for did, d in self.deals.items():
            w = self.card_widgets[did]
            mp = d.mispricing()
            w['mkt'].config(text=f"${d.tgt_price:.2f}")
            w['prob'].config(text=f"P={d.prob*100:.0f}% ({d.news_count})")
            w['mis'].config(text=f"${mp:+.2f}", fg=COL_GREEN if mp > 0 else COL_RED)

if __name__ == "__main__":
    root = tk.Tk()
    app = MergerDashboard(root)
    root.mainloop()