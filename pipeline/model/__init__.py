"""Evidence in, labels out.

``emissions`` scores each frame for H, A and D; ``decode`` turns a score sequence into a possession
sequence; ``learn`` fits the emission weights on our own pseudo-labels.
"""
