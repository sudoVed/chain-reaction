"""
game.py  -  Pure game logic, no pygame drawing.

Cascade model  (wave-based)
---------------------------
All currently queued explosions fire as one WAVE simultaneously.
Two kinds of explosion in a wave:

  1. NORMAL  - added to queue by place() or by a previous wave where the
               cell was already >= critical_mass.  Processed in Step 1 by
               subtracting critical_mass from count.

  2. COMMITTED - detected in Step 2 when an incoming orb first pushes a cell
                 to critical_mass.  The cell is CAPTURED (owner flipped) and
                 count is RESET to 0 (post-explosion remaining orbs start here).
                 Marked in self.committed_explosions so Step 1 of the next wave
                 scatters critical_mass orbs unconditionally.

Double-hit example: cell C (interior cm=4) has 3 orbs, two sources hit at once.
  - orb 1 arrives: count 3->4 >= cm -> commit (reset to 0, queue, capture)
  - orb 2 arrives: count 0->1 < cm -> no new explosion
  - next wave: C is committed -> scatter 4 orbs, C keeps count=1
  Result: one explosion, C ends with 1 orb.  No runaway chain.
"""

from collections import deque

DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


class Cell:
    __slots__ = ("owner", "count")

    def __init__(self):
        """Initialise a cell with no owner and zero orbs."""
        self.owner = -1
        self.count = 0

    def is_empty(self):
        """Return True if no player owns this cell (owner == -1)."""
        return self.owner == -1


class Game:
    def __init__(self, num_players, rows, cols):
        """
        Create a new game on a rows x cols grid for num_players players.

        All cells start empty.  Player 0 goes first.
        first_turn_done tracks whether each player has placed at least once;
        elimination checks only begin after every player has had their first turn.
        """
        self.num_players = num_players
        self.rows = rows
        self.cols = cols

        self.grid = [[Cell() for _ in range(cols)] for _ in range(rows)]

        self.current_player = 0
        self.turn_number = 0
        self.first_turn_done = [False] * num_players
        self.alive = [True] * num_players

        self.explosion_queue = deque()
        self.committed_explosions = set()

        self.state = "placing"
        self.winner = -1

        self.history: list = []   # snapshots for undo

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    def in_bounds(self, r, c):
        """Return True if (r, c) is a valid grid coordinate."""
        return 0 <= r < self.rows and 0 <= c < self.cols

    def neighbours(self, r, c):
        """
        Return a list of (row, col) tuples for all valid orthogonal neighbours
        of cell (r, c).  Corner cells have 2, edge cells 3, interior cells 4.
        """
        return [(r + dr, c + dc) for dr, dc in DIRECTIONS
                if self.in_bounds(r + dr, c + dc)]

    def critical_mass(self, r, c):
        """
        Return the number of orbs needed to trigger an explosion at (r, c).
        Equals the neighbour count: 2 for corners, 3 for edges, 4 for interior.
        """
        return len(self.neighbours(r, c))

    def is_primed(self, r, c):
        """
        Return True if cell (r, c) is one orb away from exploding.
        A primed cell is owned and has count == critical_mass - 1.
        The renderer uses this to spin the orb cluster as a visual warning.
        """
        cell = self.grid[r][c]
        return (not cell.is_empty()) and cell.count == self.critical_mass(r, c) - 1

    def is_overloaded(self, r, c):
        """Return True if cell (r, c) currently holds enough orbs to explode."""
        return self.grid[r][c].count >= self.critical_mass(r, c)

    # ------------------------------------------------------------------
    # Placement
    # ------------------------------------------------------------------

    def can_place(self, r, c):
        """
        Return True if the current player may legally place an orb at (r, c).

        Placement is only allowed when the game is in placing state and the
        target cell is either empty or already owned by the current player.
        """
        if self.state != "placing":
            return False
        cell = self.grid[r][c]
        return cell.is_empty() or cell.owner == self.current_player

    def place(self, r, c):
        """
        Place one orb for the current player at (r, c).

        Returns False if the move is illegal (see can_place).
        On success:
          - increments the cell count and marks the player's first turn done.
          - if the cell becomes overloaded, transitions to animating state and
            queues (r, c) for the first explosion wave.
          - otherwise ends the turn immediately.
        Returns True on success.
        """
        if not self.can_place(r, c):
            return False
        self.history.append(self._snapshot())   # save for undo
        cell = self.grid[r][c]
        cell.owner = self.current_player
        cell.count += 1
        self.first_turn_done[self.current_player] = True
        self.turn_number += 1
        if self.is_overloaded(r, c):
            self.state = "animating"
            self.explosion_queue.append((r, c))
        else:
            self._end_turn()
        return True

    # ------------------------------------------------------------------
    # Wave-based cascade  (called by renderer)
    # ------------------------------------------------------------------

    def get_wave(self):
        """
        Drain the explosion queue into a plain list and return it.

        Called by the renderer at the start of each animation cycle.  The
        renderer stores the list, animates it (burst + flying phases), then
        passes it back to apply_wave() once the animation completes.
        The queue is cleared here so subsequent calls return an empty list
        until apply_wave() populates it with the next wave.
        """
        wave = list(self.explosion_queue)
        self.explosion_queue.clear()
        return wave

    def apply_wave(self, wave):
        """
        Apply one complete explosion wave.

        Step 1 - scatter orbs from every cell in the wave:
          - committed cells: scatter critical_mass orbs unconditionally,
            keep their current count.
          - normal cells: count >= cm -> subtract cm, scatter cm orbs.

        Step 2 - apply incoming orbs one by one:
          - if a cell first reaches critical_mass, COMMIT it:
            capture (flip owner), reset count to 0, add to next wave.
          - additional orbs to already-committed cells add to their
            post-explosion count (starting from 0).

        Post-wave: check for eliminations and winner after every wave.
        Winner detection runs mid-cascade (not only when the queue is empty)
        so that a single-colour runaway cascade is stopped the moment only
        one player remains alive, preventing an infinite loop.
        """
        # --- Step 1: scatter ---
        outgoing = []

        for r, c in wave:
            cell = self.grid[r][c]
            cm = self.critical_mass(r, c)
            owner = cell.owner

            if (r, c) in self.committed_explosions:
                self.committed_explosions.discard((r, c))
                if cell.count == 0:
                    cell.owner = -1
                for nr, nc in self.neighbours(r, c):
                    outgoing.append((nr, nc, owner))

            elif cell.count >= cm:
                cell.count -= cm
                if cell.count <= 0:
                    cell.count = 0
                    cell.owner = -1
                for nr, nc in self.neighbours(r, c):
                    outgoing.append((nr, nc, owner))

        # --- Step 2: apply orbs sequentially ---
        next_wave_set = set()

        for nr, nc, inc_owner in outgoing:
            ncell = self.grid[nr][nc]
            cm = self.critical_mass(nr, nc)

            # Always capture on hit -- the incoming orb converts the cell
            # immediately regardless of whether it then explodes.
            ncell.owner = inc_owner
            ncell.count += 1

            if ncell.count >= cm and (nr, nc) not in next_wave_set:
                ncell.count -= cm
                if ncell.count < 0:
                    ncell.count = 0
                next_wave_set.add((nr, nc))
                self.explosion_queue.append((nr, nc))
                self.committed_explosions.add((nr, nc))

        # --- Post-wave ---
        # Check winner every wave, not just when the queue drains.
        # A densely packed single-colour board can sustain perpetual cascades;
        # detecting the winner mid-cascade stops it immediately.
        self._check_eliminations()
        winner = self._check_winner()
        if winner is not None:
            self.state = "won"
            self.winner = winner
            self.explosion_queue.clear()
            self.committed_explosions.clear()
        elif not self.explosion_queue:
            self._end_turn()

    # ------------------------------------------------------------------
    # Bookkeeping
    # ------------------------------------------------------------------


    def _snapshot(self):
        """
        Capture the full mutable game state into a plain dict.

        Saves the grid (as a flat list of (owner, count) tuples), player
        tracking lists, and state flags.  The explosion queue and committed
        set are always empty at snapshot time because snapshots are taken
        only when state == "placing".
        """
        return {
            "grid": [(cell.owner, cell.count)
                     for row in self.grid for cell in row],
            "current_player":  self.current_player,
            "turn_number":     self.turn_number,
            "first_turn_done": list(self.first_turn_done),
            "alive":           list(self.alive),
            "state":           self.state,
            "winner":          self.winner,
        }

    def _restore(self, snap: dict):
        """
        Restore game state from a snapshot produced by _snapshot().

        Rebuilds the grid cell-by-cell, copies all scalar fields, and clears
        the explosion queue and committed set (they are always empty when a
        snapshot was taken and therefore not stored).
        """
        flat = snap["grid"]
        for i, row in enumerate(self.grid):
            for j, cell in enumerate(row):
                cell.owner, cell.count = flat[i * self.cols + j]
        self.current_player  = snap["current_player"]
        self.turn_number     = snap["turn_number"]
        self.first_turn_done = list(snap["first_turn_done"])
        self.alive           = list(snap["alive"])
        self.state           = snap["state"]
        self.winner          = snap["winner"]
        self.explosion_queue.clear()
        self.committed_explosions.clear()

    def undo(self) -> bool:
        """
        Revert to the state before the most recent placement.

        Can only be called while state == "placing" (not mid-animation).
        Returns True if an undo was performed, False if the history is empty
        or the game is currently animating/won.
        """
        if self.state != "placing" or not self.history:
            return False
        self._restore(self.history.pop())
        return True

    def _check_eliminations(self):
        """
        Mark any player as eliminated who owns zero cells after this wave.

        Only triggers after a player has completed their first turn; this
        prevents players being immediately eliminated before they've placed.
        Once alive[p] is set to False the player is skipped in _end_turn()
        and excluded from the winner check in _check_winner().
        """
        counts = [0] * self.num_players
        for row in self.grid:
            for cell in row:
                if cell.owner >= 0:
                    counts[cell.owner] += 1
        for p in range(self.num_players):
            if self.first_turn_done[p] and counts[p] == 0:
                self.alive[p] = False

    def _check_winner(self):
        """
        Return the winning player index if exactly one player remains alive,
        otherwise return None.

        Victory is only possible once every player has had at least one turn
        (to prevent the first player winning on an empty board).
        """
        if not all(self.first_turn_done):
            return None
        alive = [p for p in range(self.num_players) if self.alive[p]]
        return alive[0] if len(alive) == 1 else None

    def _end_turn(self):
        """
        Advance to the next living player and reset the game state to placing.

        Skips eliminated players in round-robin order so the turn sequence
        contracts naturally as players are knocked out.
        """
        self.state = "placing"
        nxt = (self.current_player + 1) % self.num_players
        for _ in range(self.num_players):
            if self.alive[nxt]:
                break
            nxt = (nxt + 1) % self.num_players
        self.current_player = nxt

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def orb_counts(self):
        """
        Return a list of orb counts, one entry per player.

        counts[p] is the total number of orbs currently owned by player p
        across all cells (sum of cell.count for each cell owned by p).
        Used by the renderer to display live scores in the UI bar.
        """
        counts = [0] * self.num_players
        for row in self.grid:
            for cell in row:
                if cell.owner >= 0:
                    counts[cell.owner] += cell.count
        return counts
