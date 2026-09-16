from player_quoridor import PlayerQuoridor
from seahorse.game.action import Action
from game_state_quoridor import GameStateQuoridor
from seahorse.utils.custom_exceptions import MethodNotImplementedError


DEPTH = 2


class MyPlayer(PlayerQuoridor):
    """
    Player class for Quoridor game

    Attributes:
        piece_type (str): piece type of the player
    """

    def __init__(self, piece_type: str, goal_row: int=0, name: str = "bob", *args, **kwargs) -> None:
        """
        Initialize the PlayerQuoridor instance.

        Args:
            piece_type (str): Type of the player's game piece
            goal_row (int): The row the player wants to reach
            name (str, optional): Name of the player (default is "bob")
        """
        super().__init__(piece_type, goal_row, name)

    def compute_action(self, current_state: GameStateQuoridor, remaining_time: float = 15*60, **kwargs) -> Action:
        """
        Use the minimax algorithm to choose the best action based on the heuristic evaluation of game states.

        Args:
            current_state (GameStateQuoridor): The current game state.

        Returns:
            Action: The best action as determined by minimax.
        """

        actions = tuple(current_state.generate_possible_stateless_actions())
        if not actions:
            raise RuntimeError("No legal action available.")

        best_action = None
        best_value = float("-inf")

        for action in actions:
            child = current_state.apply_action(action)
            value = self._search(child, DEPTH - 1, maximizing=False)
            if value > best_value:
                best_value = value
                best_action = action

        return best_action

    def _search(self, state: GameStateQuoridor, depth: int, maximizing: bool) -> float:
        if depth == 0 or state.is_done():
            return self._evaluate(state)

        actions = tuple(state.generate_possible_stateless_actions())
        if not actions:
            return self._evaluate(state)

        if maximizing:
            value = float("-inf")
            for action in actions:
                child = state.apply_action(action)
                value = max(value, self._search(child, depth - 1, False))
            return value
        else:
            value = float("inf")
            for action in actions:
                child = state.apply_action(action)
                value = min(value, self._search(child, depth - 1, True))
            return value

    def _evaluate(self, state: GameStateQuoridor) -> float:
        me = next(p for p in state.players if p.get_id() == self.get_id())
        opp = next(p for p in state.players if p.get_id() != self.get_id())

        my_dist = state._shortest_path(me)
        opp_dist = state._shortest_path(opp)

        if my_dist is None:
            my_dist = 100
        if opp_dist is None:
            opp_dist = 100

        return opp_dist - my_dist