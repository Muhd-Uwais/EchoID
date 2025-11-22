# Voice Speaker Recognition - Real-Time Inference Entry Point
"""
This module serves as the main entry point for the EchoID real-time inference system.
It initializes the graphical user interface (GUI) application, allowing users to
perform live speaker recognition using the trained CNN model.

Usage:
------
    python inference.py

    (This launches the GUI window for recording and prediction)

Name: EchoID
Author: Muhd Uwais
Project: Deep Voice Speaker Recognition CNN
Purpose: Inference Execution (GUI Launcher)
License: MIT
"""

from src.inference.listener import InferenceApp


# =============================================================
# Main Execution Block
# =============================================================

if __name__ == "__main__":
    """
    Initialize and launch the Inference GUI Application.

    This script instantiates the main application class and starts
    the event loop, bringing up the user interface.
    """

    print("🚀 Starting EchoID Inference System...")

    # ------------------ Application Launch ------------------
    app = InferenceApp()
    app.run()
