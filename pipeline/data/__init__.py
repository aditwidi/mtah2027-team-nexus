"""Reading the provided assets and turning them into a per-frame feature matrix.

``io`` is the only module that touches the match files; ``features`` turns what it returns into
rows indexed by video frame. Neither makes a decision or reads a label.
"""
