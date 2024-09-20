# image_loader.py

from PIL import Image, ImageTk
import os

CARD_IMAGES = {}
GREY_CARD_IMAGES = {}
JOKER_IMAGE = None

def load_card_images():
    cards_dir = os.path.join(os.getcwd(), 'assets', 'cards')
    for filename in os.listdir(cards_dir):
        if filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.jpeg'):
            card_name = os.path.splitext(filename)[0]  # Remove extension
            image_path = os.path.join('static', 'cards', filename)
            # Load color image
            CARD_IMAGES[card_name] = image_path
            # Create greyed-out image path
            GREY_CARD_IMAGES[card_name] = image_path.replace('.png', '_grey.png')  # Assume grey images exist
    # Optionally, handle joker if present
