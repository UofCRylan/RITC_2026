using HTTP, JSON, Statistics, Dates

# =========================== CONFIGURATION ====================================
const API_KEY = "RYL8"  # <--- ENTER YOUR KEY HERE
const RIT_HOST = "http://localhost:9994"
const TICKERS = ["SPNG", "SMMR", "ATMN", "WNTR"]

# --- Strategy Parameters ---
const SPREAD_BASE = 0.10
const SKEW_FACTOR = 0.001
const ORDER_SIZE = 1000
const MAX_POS_INTRADAY = 9000
const AGG_LIMIT_BUFFER = 2000

# --- Risk Parameters ---
const EOM_FLUSH_START = 56
const VOL_WINDOW = 10
const VOL_MULTIPLIER = 2.0

# =========================== RIT API WRAPPER ==================================
module RIT
    using HTTP, JSON, ..Main
    
    function get_case_time()
        try
            return JSON.parse(String(HTTP.get("$(Main.RIT_HOST)/v1/case").body))
        catch; return nothing; end
    end

    function get_securities()
        try
            return JSON.parse(String(HTTP.get("$(Main.RIT_HOST)/v1/securities").body))
        catch; return []; end
    end

    function get_positions()
        try
            data = JSON.parse(String(HTTP.get("$(Main.RIT_HOST)/v1/securities").body))
            pos_dict = Dict{String, Int}()
            for sec in data
                if sec["ticker"] in Main.TICKERS
                    pos_dict[sec["ticker"]] = sec["position"]
                end
            end
            return pos_dict
        catch; return Dict{String, Int}(); end
    end

    function post_order(ticker, type, quantity, action, price=0.0)
        url = "$(Main.RIT_HOST)/v1/orders"
        params = Dict("ticker" => ticker, "type" => type, "quantity" => quantity, "action" => action, "price" => price)
        try; HTTP.post(url, ["X-API-Key" => Main.API_KEY], JSON.json(params)); catch; end
    end

    function mass_cancel(ticker_filter=nothing)
        try
            orders = JSON.parse(String(HTTP.get("$(Main.RIT_HOST)/v1/orders?status=OPEN").body))
            for o in orders
                if isnothing(ticker_filter) || o["ticker"] == ticker_filter
                    HTTP.delete("$(Main.RIT_HOST)/v1/orders/$(o["order_id"])", ["X-API-Key" => Main.API_KEY])
                end
            end
        catch; end
    end
end

# =========================== LOGIC ====================================
mutable struct MarketState
    price_history::Dict{String, Vector{Float64}}
    agg_limit::Int64
end

global state = MarketState(Dict(t => [] for t in TICKERS), 20000) 

function calculate_volatility_multiplier(ticker, current_mid)
    history = state.price_history[ticker]
    push!(history, current_mid)
    if length(history) > VOL_WINDOW; popfirst!(history); end
    
    if length(history) < VOL_WINDOW; return 1.0; end
    
    if std(history) > 0.05
        return VOL_MULTIPLIER
    else
        return 1.0
    end
end

# =========================== MAIN LOOP ========================================
function main()
    println(">>> RITC 2026 JULIA ALGO STARTED <<<")
    
    while true
        case_info = RIT.get_case_time()
        if isnothing(case_info); sleep(1); continue; end
        
        tick = case_info["tick"]
        status = case_info["status"]
        if status != "ACTIVE"; sleep(1); continue; end

        seconds_in_minute = tick % 60
        
        # --- STRATEGY 2: EOM LIQUIDATION ---
        if seconds_in_minute >= EOM_FLUSH_START
            println("!!! EOM FLUSH TRIGGERED ($seconds_in_minute s) !!!")
            RIT.mass_cancel()
            positions = RIT.get_positions()
            for (ticker, qty) in positions
                if qty > 0
                    RIT.post_order(ticker, "MARKET", abs(qty), "SELL")
                elseif qty < 0
                    RIT.post_order(ticker, "MARKET", abs(qty), "BUY")
                end
            end
            sleep(0.5); continue
        end

        securities_data = RIT.get_securities()
        current_positions = RIT.get_positions()
        current_agg_pos = sum(abs(get(current_positions, t, 0)) for t in TICKERS)
        
        for sec in securities_data
            ticker = sec["ticker"]
            if !(ticker in TICKERS); continue; end
            
            mid_price = (sec["bid"] + sec["ask"]) / 2.0
            my_pos = get(current_positions, ticker, 0)
            
            # --- STRATEGY 3: VOLATILITY WIDENING ---
            vol_mult = calculate_volatility_multiplier(ticker, mid_price)
            target_spread = max(SPREAD_BASE * vol_mult, (sec["ask"] - sec["bid"]) * 0.9)
            
            # --- STRATEGY 1: INVENTORY SKEWING ---
            skewed_mid = mid_price - (my_pos * SKEW_FACTOR)
            my_bid = round(skewed_mid - (target_spread / 2), digits=2)
            my_ask = round(skewed_mid + (target_spread / 2), digits=2)
            
            risk_full = current_agg_pos >= (state.agg_limit - AGG_LIMIT_BUFFER)
            RIT.mass_cancel(ticker)
            
            if my_pos < MAX_POS_INTRADAY
                if !risk_full || (risk_full && my_pos < 0)
                    RIT.post_order(ticker, "LIMIT", ORDER_SIZE, "BUY", my_bid)
                end
            end
            
            if my_pos > -MAX_POS_INTRADAY
                if !risk_full || (risk_full && my_pos > 0)
                    RIT.post_order(ticker, "LIMIT", ORDER_SIZE, "SELL", my_ask)
                end
            end
        end
        sleep(0.5)
    end
end

main()