# Oh Hell Web Game

This project provides a small web interface for playing the classic card game **Oh Hell** against basic AI opponents.

## Objectives

- Illustrate the rules and flow of the game Oh Hell.
- Allow a user to start rounds, place bids and view hands within a web browser.
- Serve as a simple example of using Flask and Dash for lightweight game UIs.

## Setup

1. Ensure you have Python 3 installed.
2. Install the required packages:

```bash
pip install -r requirements.txt
```

## Running the Website

Two interfaces are provided:

### Flask version

Run the simple Flask website:

```bash
python website.py
```

The site will start on <http://localhost:5000/> where you can begin a game, submit bids and view the current state.

### Dash version

Alternatively run the Dash UI:

```bash
python app.py
```

This launches a richer dashboard style interface also on port 8050 by default.

## Assets

Card images are stored under `assets/cards`. These are served as static files when the website runs.

