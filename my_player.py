from player_quoridor import PlayerQuoridor
from seahorse.game.action import Action
from game_state_quoridor import GameStateQuoridor
from seahorse.utils.custom_exceptions import MethodNotImplementedError
import time

MIN_DEPTH = 2
MAX_DEPTH = 5
CACHE_LIMIT = 100_000

TIME_SAFETY = 0.15


# Transposition table flags: what a cached value means relative to the true minimax value.
EXACT = 0         # the cached value is the true value
LOWER_BOUND = 1   # true value >= cached value (search failed high: beta cutoff)
UPPER_BOUND = 2   # true value <= cached value (search failed low: no move beat alpha)


class SearchTimeout(Exception):
    pass

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
        self._transposition_table = {}
        self._preferred_actions = {}


    def compute_action(self, current_state: GameStateQuoridor, remaining_time: float = 15*60, **kwargs) -> Action:
        """
        Use the minimax algorithm to choose the best action based on the heuristic evaluation of game states.

        Args:
            current_state (GameStateQuoridor): The current game state.

        Returns:
            Action: The best action as determined by minimax.
        """

        start_time = time.monotonic()

        actions = self._order_actions(
            current_state,
            tuple(current_state.generate_possible_stateless_actions()),
            maximizing=True
        )

        if not actions:
            raise RuntimeError("No legal action available.")

        # Always keep a legal fallback action
        best_action = actions[0]

        # Allocate a small portion of remaining time to this turn
        time_budget = min(
            3.0,
            max(0.01, remaining_time * 0.05)
        )

        deadline = start_time + max(
            0.0,
            time_budget - TIME_SAFETY
        )

        target_depth = self._get_dynamic_depth(
            current_state,
            remaining_time,
            len(actions)
        )

        # Iterative deepening
        for depth in range(1, target_depth + 1):

            try:
                best_value = float("-inf")
                iteration_best = None

                alpha = float("-inf")
                beta = float("inf")

                # Reorder actions using the best move from previous searches
                ordered_actions = self._order_actions(
                    current_state,
                    actions,
                    maximizing=True
                )

                for action in ordered_actions:

                    if time.monotonic() >= deadline:
                        raise SearchTimeout()

                    child = current_state.apply_action(action)

                    value = self._search(
                        child,
                        depth - 1,
                        False,
                        alpha,
                        beta,
                        deadline
                    )

                    if value > best_value:
                        best_value = value
                        iteration_best = action

                    alpha = max(alpha, best_value)

                # Only save the result if the entire iteration completed
                if iteration_best is not None:
                    best_action = iteration_best

                    self._preferred_actions[
                        self._state_key(current_state)
                    ] = self._action_key(best_action)

            except SearchTimeout:
                break

        return best_action
    
    def _get_dynamic_depth(self, state: GameStateQuoridor, remaining_time: float, nb_actions: int) -> int:

        # Adjust depth based on branching factor
        if nb_actions <= 5:
            depth = 5

        elif nb_actions <= 15:
            depth = 4

        elif nb_actions <= 40:
            depth = 3

        else:
            depth = 2

        # Reduce depth when time is running low
        if remaining_time < 10:
            depth = min(depth, 2)

        elif remaining_time < 30:
            depth = min(depth, 3)

        elif remaining_time < 60:
            depth = min(depth, 4)

        return max(1, min(depth, MAX_DEPTH))

    def _search(self, state: GameStateQuoridor, depth: int, maximizing: bool, alpha: float, beta: float,  deadline: float) -> float:

        # Stop searching if time budget is exceeded
        if time.monotonic() >= deadline:
            raise SearchTimeout() 
        
        if depth == 0 or state.is_done():
            return self._evaluate(state)

        # The state key is computed once and reused (cache, move ordering, best move).
        state_key = self._state_key(state)
        cache_key = (state_key, depth, maximizing)

        # A cached bound may answer directly, or at least narrow the alpha-beta window.
        cached = self._transposition_table.get(cache_key)
        if cached is not None:
            cached_value, flag = cached
            if flag == EXACT:
                return cached_value
            if flag == LOWER_BOUND:
                alpha = max(alpha, cached_value)
            else:
                beta = min(beta, cached_value)
            if alpha >= beta:
                return cached_value

        # Window actually used for this search, needed to classify the result below
        # (alpha/beta are modified inside the loop).
        search_alpha, search_beta = alpha, beta

        actions = self._order_actions(
            state,
            tuple(state.generate_possible_stateless_actions()),
            maximizing=maximizing,
            state_key=state_key,
        )
        if not actions:
            return self._evaluate(state)

        best_action = None
        if maximizing:
            value = float("-inf")
            for action in actions:

                if time.monotonic() >= deadline:
                    raise SearchTimeout()
                
                child = state.apply_action(action)
                child_value = self._search(child, depth - 1, False, alpha, beta, deadline)
                if child_value > value:
                    value = child_value
                    best_action = action
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
        else:
            value = float("inf")
            for action in actions:

                if time.monotonic() >= deadline:
                    raise SearchTimeout()
                
                child = state.apply_action(action)
                child_value = self._search(child, depth - 1, True, alpha, beta, deadline)
                if child_value < value:
                    value = child_value
                    best_action = action
                beta = min(beta, value)
                if alpha >= beta:
                    break

        if best_action is not None:
            self._preferred_actions[state_key] = self._action_key(best_action)

        # A value outside the search window is only a bound, not the true value.
        if value <= search_alpha:
            flag = UPPER_BOUND
        elif value >= search_beta:
            flag = LOWER_BOUND
        else:
            flag = EXACT
        self._store_cache(cache_key, (value, flag))
        return value

    def _order_actions(self, state: GameStateQuoridor, actions: tuple,
                       maximizing: bool, state_key: tuple = None) -> tuple:
        """Order likely useful actions before less promising actions."""
        active_player = state.active_player
        goal_row = active_player.get_goal_row()
        pawn_moves = []
        wall_actions = []

        for action in actions:
            if action.data["type"] != "move":
                wall_actions.append(action)
                continue

            destination = action.data["destination"]
            distance_to_goal = abs(destination[0] - goal_row)
            pawn_moves.append((distance_to_goal, action))

        pawn_moves.sort(key=lambda item: item[0])
        ordered_moves = [action for _, action in pawn_moves]
        ordered_actions = ordered_moves + wall_actions

        if state_key is None:
            state_key = self._state_key(state)
        preferred_key = self._preferred_actions.get(state_key)
        if preferred_key is not None:
            for index, action in enumerate(ordered_actions):
                if self._action_key(action) == preferred_key:
                    ordered_actions.insert(0, ordered_actions.pop(index))
                    break

        return tuple(ordered_actions)

    def _state_key(self, state: GameStateQuoridor) -> tuple:
        return (
            state.active_player.get_id(),
            frozenset(state.rep.pawn_positions.items()),
            frozenset(state.rep.remaining_walls.items()),
            state.rep.walls,
        )

    def _action_key(self, action) -> tuple:
        return action.data["type"], tuple(action.data["destination"])

    def _store_cache(self, key: tuple, value: tuple) -> None:
        """Store (value, flag) for key, evicting the oldest entry when the table is full."""
        if len(self._transposition_table) >= CACHE_LIMIT:
            self._transposition_table.pop(next(iter(self._transposition_table)))
        self._transposition_table[key] = value

    def _evaluate(self, state: GameStateQuoridor) -> float:
        me = next(p for p in state.players if p.get_id() == self.get_id())
        opp = next(p for p in state.players if p.get_id() != self.get_id())

        my_dist = state._shortest_path(me)
        opp_dist = state._shortest_path(opp)

        if my_dist is None:
            my_dist = 100
        if opp_dist is None:
            opp_dist = 100
    # Terminal states
        if my_dist == 0:
            return 100_000

        if opp_dist == 0:
            return -100_000

        # Distance difference (main factor)
        distance_score = 10 * (opp_dist - my_dist)

        # Remaining walls
        my_walls = state.rep.remaining_walls[me.get_id()]
        opp_walls = state.rep.remaining_walls[opp.get_id()]

        wall_score = 2 * (my_walls - opp_walls)

        # Walls are less important near the end of the game
        if my_dist <= 2 or opp_dist <= 2:
            wall_score *= 0.25

        # Victory proximity
        urgency_score = 0

        if my_dist == 1:
            urgency_score += 40

        elif my_dist == 2:
            urgency_score += 15

        if opp_dist == 1:
            urgency_score -= 50

        elif opp_dist == 2:
            urgency_score -= 20

        return distance_score + wall_score + urgency_score
