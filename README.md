# Cashier Wrist Tracker
##  For detecting unscanned items

A common way to steal by cashiers: move unscanned items straight to the basket.

CashierWristTracker uses a **cashier's left wrist** movement to understand when it enters a predefined "basket area", then generates an event.

Example use case: when an item is not scanned, but the system detects that the cashier’s wrist moved into the basket area, CashierWristTracker can trigger an alert for potential theft.

Watch the video (39sec): https://youtube.com/shorts/uhsP46S5K4k

![My image](Thumb.png)

## Features

- **Video Analysis:** Process video footage of cashiers at checkout counters.
- **Left-Wrist Tracking:** Monitors the movement of the cashier's left wrist.
- **Basket-Area Detection:** Detects when the left wrist enters a predefined basket region.
- **Event Generation:** Logs an event each time an item is placed into the basket.
- **Pose Estimation:** Uses pose detection to track wrist and hand movements.


Technical details:
- Neural network - YOLOv8n-pose (pre-trained)
- Frame sampling - 0.2 sec
