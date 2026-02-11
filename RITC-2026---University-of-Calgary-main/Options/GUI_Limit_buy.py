

import tkinter as tk
from tkinter import messagebox
from dataclasses import dataclass, field


@dataclass
class OptionInstrument:
    ticker: str
    delta_per_contract: float
    contracts_held: int = 0

    def position_delta(self) -> float:
        """Return the total delta of the current position in this option."""
        return self.contracts_held * self.delta_per_contract


class VolatilityTraderGUI:
    def __init__(self, root: tk.Tk, delta_limit: int = 10000, max_contracts: int = 100):
        self.root = root
        self.root.title("Volatility Trader GUI (Simulation)")
        self.delta_limit = delta_limit
        self.max_contracts = max_contracts
        # Underlying ETF position (shares) and delta per share is 1
        self.underlying_position: int = 0
        # Create option instruments for calls and puts
        self.instruments: dict[str, OptionInstrument] = {}
        strikes = list(range(45, 55))
        for k in strikes:
            call_ticker = f"RTM1C{k}"
            put_ticker = f"RTM1P{k}"
            # Each option contract has delta ±50 (0.5 * 100 shares)
            self.instruments[call_ticker] = OptionInstrument(call_ticker, delta_per_contract=+50.0)
            self.instruments[put_ticker] = OptionInstrument(put_ticker, delta_per_contract=-50.0)
        # Build UI
        self.build_ui()

    def build_ui(self) -> None:
        # Canvas for delta lines and marker
        self.canvas_height = 200
        self.canvas_width = 400
        self.canvas = tk.Canvas(self.root, width=self.canvas_width, height=self.canvas_height, bg="white")
        self.canvas.pack(pady=10)
        # Frame for option buttons
        button_frame = tk.Frame(self.root)
        button_frame.pack()
        # Create buttons for calls and puts in a grid
        row = 0
        col = 0
        for ticker, instrument in self.instruments.items():
            btn = tk.Button(button_frame, text=f"Buy max {ticker}", width=14,
                            command=lambda t=ticker: self.buy_max_and_hedge(t))
            btn.grid(row=row, column=col, padx=5, pady=5)
            col += 1
            if col % 5 == 0:
                row += 1
                col = 0
        # Label to show current delta and underlying position
        self.status_var = tk.StringVar()
        self.status_label = tk.Label(self.root, textvariable=self.status_var, font=("Helvetica", 12))
        self.status_label.pack(pady=5)
        self.update_display()

    def total_delta(self) -> float:
        """Calculate the aggregate delta exposure (options + underlying)."""
        option_delta = sum(instr.position_delta() for instr in self.instruments.values())
        # Underlying delta is 1 per share
        return option_delta + self.underlying_position

    def buy_max_and_hedge(self, ticker: str) -> None:
        """Simulate buying max contracts of the given option and hedge delta."""
        instrument = self.instruments[ticker]
        # Determine intended delta impact of buying max contracts
        added_delta = instrument.delta_per_contract * self.max_contracts
        # Add contracts
        instrument.contracts_held += self.max_contracts
        # Compute new total option delta (excluding hedge)
        total_option_delta = sum(instr.position_delta() for instr in self.instruments.values())
        # Determine hedge trade in underlying to bring delta to zero
        hedge_shares = -total_option_delta  # because underlying delta is 1 per share
        # Adjust underlying position
        self.underlying_position += int(round(hedge_shares))
        # Update display
        self.update_display()
        # Show message summarizing trade
        msg = (
            f"Bought {self.max_contracts} contracts of {ticker} (delta per contract: {instrument.delta_per_contract}).\n"
            f"New option delta: {total_option_delta:.0f}.\n"
            f"Executed hedge trade: {int(round(hedge_shares))} shares of RTM to bring net delta to zero."
        )
        messagebox.showinfo("Trade executed (simulated)", msg)

    def update_display(self) -> None:
        """Redraw the delta limit lines and marker to reflect current exposure."""
        self.canvas.delete("all")
        # Map delta value to y-coordinate: center (0) is middle of canvas
        # Upper limit corresponds to top red line; lower limit corresponds to bottom red line
        def delta_to_y(delta: float) -> float:
            # Clamp delta to ±delta_limit for display
            clamped = max(-self.delta_limit, min(self.delta_limit, delta))
            # Normalized position between -delta_limit and +delta_limit
            norm = (clamped + self.delta_limit) / (2 * self.delta_limit)
            return (1 - norm) * self.canvas_height
        # Draw red limit lines
        upper_y = delta_to_y(+self.delta_limit)
        lower_y = delta_to_y(-self.delta_limit)
        self.canvas.create_line(0, upper_y, self.canvas_width, upper_y, fill="red", dash=(4, 2))
        self.canvas.create_line(0, lower_y, self.canvas_width, lower_y, fill="red", dash=(4, 2))
        # Draw green zero line
        zero_y = delta_to_y(0)
        self.canvas.create_line(0, zero_y, self.canvas_width, zero_y, fill="green")
        # Draw current delta marker as a blue circle
        curr_delta = self.total_delta()
        marker_y = delta_to_y(curr_delta)
        self.canvas.create_oval(self.canvas_width//2 - 5, marker_y - 5,
                                self.canvas_width//2 + 5, marker_y + 5,
                                fill="blue", outline="black")
        # Update status label
        self.status_var.set(
            f"Total delta: {curr_delta:.0f}\n"
            f"Underlying position (RTM shares): {self.underlying_position:.0f}"
        )


def main() -> None:
    root = tk.Tk()
    app = VolatilityTraderGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()