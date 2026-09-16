from player_quoridor import PlayerQuoridor
from seahorse.game.action import Action
from game_state_quoridor import GameStateQuoridor
from seahorse.utils.custom_exceptions import MethodNotImplementedError

class MyPlayer(PlayerQuoridor):
    """
    Player class for Quoridor game

    V1: DFS 

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

        possible_actions = list(current_state.generate_possible_stateless_actions())
        if not possible_actions:
            raise RuntimeError("No actions available.")

        best_action = possible_actions[0]
        best_value = float("-inf")

        for action in possible_actions:
            next_state = current_state.apply_action(action)
            value = self._dfs(next_state)

            if value > best_value:
                best_value = value
                best_action = action

        return best_action
    
    def _dfs(self, initial_state: GameStateQuoridor) -> float:
      
        s = initial_state

        list = [(s, 0)]
        max_depth = 2

        while list:
   
            current_state, depth = list.pop()

            if current_state.is_done() or depth >= max_depth:
                return self.evaluate(current_state)

           
            for action in current_state.generate_possible_stateless_actions():
                child = current_state.apply_action(action)
                list.append((child, depth + 1))

        return self.evaluate(s)
        
    def evaluate(self, state: GameStateQuoridor) -> float:
        """
        Heuristic: Difference between opponent's shortest path and my shortest path[cite: 1].
        """
        me = None
        opponent = None

        if state.players[0] == self.id:
            me = state.players[0]
            opponent = state.players[1]
        else:
            me = state.players[1]
            opponent = state.players[0]

        my_dist = state._shortest_path(me)
        opp_dist = state._shortest_path(opponent)


        return float(opp_dist - my_dist)
    