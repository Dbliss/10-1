import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
from game import Game
import random
import math

app = dash.Dash(__name__, suppress_callback_exceptions=True)

app.layout = html.Div([
    html.H1("Oh Hell Card Game"),

    # Dropdown to select the starting round
    dcc.Dropdown(
        id='starting-round',
        options=[{'label': str(i), 'value': i} for i in range(11)],
        value=0,
        clearable=False,
        style={'width': '50%'}
    ),
    html.Br(),

    # Button to start the game
    html.Button('Start Game', id='start-game-button'),
    html.Button('Reset Game', id='reset-game-button', style={'margin-left': '10px'}),  # Reset button

    # Table layout for players around a circular table
    html.Div(id='table-layout', style={'position': 'relative', 'width': '800px', 'height': '800px', 'margin': '0 auto', 'border': '1px solid black', 'border-radius': '50%'}),

    # Sidebar for game information
    html.Div([
        html.H4("Game Information"),
        html.Div(id='round-number'),
        html.Div(id='trump-card-info'),
        html.Label("Enter your bid:"),
        dcc.Input(id='player-bid', type='number'),
        html.Button('Submit Bid', id='submit-bid'),
    ], style={'border': '1px solid black', 'padding': '20px', 'width': '30%', 'float': 'right'}),

    # Bidding log display
    html.Div(id='bidding-log', style={'position': 'absolute', 'top': '10px', 'right': '10px', 'font-size': '16px'}),

    # Dealer chip
    html.Div(id='dealer-chip', style={'width': '20px', 'height': '20px', 'border-radius': '50%', 'background-color': 'blue', 'position': 'absolute'}),

    # Bidding message
    html.Div(id='bidding-message', style={'position': 'absolute', 'top': '40px', 'right': '10px', 'font-size': '16px', 'color': 'red'})
])

# Dummy placeholder for your game instance
game = Game()
game_started = False

@app.callback(
    [
        Output('table-layout', 'children'),
        Output('round-number', 'children'),
        Output('trump-card-info', 'children'),
        Output('dealer-chip', 'style'),
        Output('submit-bid', 'disabled'),
        Output('bidding-log', 'children'),
        Output('bidding-message', 'children'),  # Output for bidding message
    ],
    [Input('start-game-button', 'n_clicks'),
     Input('reset-game-button', 'n_clicks'),
     Input('submit-bid', 'n_clicks')],
    [State('starting-round', 'value'),
     State('player-bid', 'value')]
)
def update_game_layout(start_clicks, reset_clicks, bid_clicks, starting_round, player_bid):
    global game, game_started

    if reset_clicks:
        game = Game()
        game_started = False
        return ([], "", "", {}, True, "", "")

    if not start_clicks:
        return ([], "", "", {}, True, "", "")

    if start_clicks > 0 and not game_started:
        game.start_round()
        game_started = True

    if bid_clicks and player_bid is not None:
        game.receive_bid("You", player_bid)

    bidding_message, is_player_turn = game.manage_turns()

    trump_card = game.get_card_image(game.trump_card) if game.trump_card else None
    player_hands = {p.name: [game.get_card_image(c) for c in p.hand] for p in game.players}

    player_boxes = []
    radius_names = 430
    radius_cards = 170

    for i, player_name in enumerate(['AI 1', 'AI 2', 'You', 'AI 3', 'AI 4', 'AI 5', 'AI 6', 'AI 7']):
        angle = (i / 8) * (2 * math.pi)

        x_name = 400 + radius_names * math.cos(angle)
        y_name = 400 + radius_names * math.sin(angle)

        name_box = html.Div(player_name, style={
            'position': 'absolute',
            'top': f"{y_name}px",
            'left': f"{x_name}px",
            'transform': 'translate(-50%, -50%)',
            'font-weight': 'bold',
            'color': 'blue' if i == game.dealer_index else 'black'
        })

        player_boxes.append(name_box)

        x_card = 180 + radius_cards * math.cos(angle)
        y_card = 180 + radius_cards * math.sin(angle)

        player_hand = player_hands.get(player_name, [])
        card_images = []

        for idx, card_image in enumerate(player_hand):
            card_images.append(html.Img(src=card_image, style={
                'width': '60px',
                'height': '80px',
                'margin': '0',
                'position': 'absolute',
                'top': f"{y_card}px",
                'left': f"{x_card + idx * 10}px"
            }))

        card_box = html.Div(style={
            'position': 'absolute',
            'top': f"{y_card}px",
            'left': f"{x_card}px",
            'transform': 'translate(-50%, -50%)'
        }, children=card_images)

        player_boxes.append(card_box)

    if trump_card:
        trump_card_display = html.Img(src=trump_card, style={
            'width': '70px',
            'height': '100px',
            'position': 'absolute',
            'top': '400px',
            'left': '400px',
            'transform': 'translate(-50%, -50%)'
        })
    else:
        trump_card_display = html.H4('No Trump', style={
            'position': 'absolute',
            'top': '400px',
            'left': '400px',
            'transform': 'translate(-50%, -50%)'
        })

    round_info = f"Round: {game.round_number}"

    bidding_log = []
    for player in game.players:
        bid_display = f"{player.name}: {player.bid}" if player.bid is not None else f"{player.name}:"
        if player.name == "You" or player.name == game.players[game.dealer_index].name:
            bid_display = f"{bid_display} (d)"
        bidding_log.append(html.Div(bid_display, style={'color': 'blue' if player.name == game.players[game.dealer_index].name else 'black'}))

    return (
        player_boxes + [trump_card_display],
        round_info,
        "",
        {},
        not is_player_turn,
        bidding_log,
        bidding_message,
    )

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
