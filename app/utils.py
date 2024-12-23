# Utility functions
import logging

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def handle_error(e):
    logging.error(f"An error occurred: {e}")