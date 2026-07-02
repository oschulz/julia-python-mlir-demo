"""
    negsumsq(x)

A trivial scalar "loss": negative sum of squares. Gradient is -2x.
"""
#negsumsq(x) = - reduce(+, x.^2)
# equivalent to
negsumsq(x) = - sum(x.^2)
