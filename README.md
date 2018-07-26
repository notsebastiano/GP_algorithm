# Alert
While this implementation was successfully tested with Lorentz and Henon attractors, with High dimensional attractors coming from a stochastic-deterministic hybrid dynamics (with additive or even multiplicative noise) the Correlation sum in log-log shows two different scaling regions. Thus results can not be trusted.

# GP_algorithm
Implementation of the Grassberger-Procaccia algorithm to estimate the correlation dimension of a set.

Correlation dimension is a fractal dimension (such as Box-counting dimension or Hausdorff dimension) and it is characteristic of the set of points. If an attractor in phase space is being studied, once the attractor is completely unfolded in m dimensions, correlation dimension becomes an invariant and further embeddings in more dimensions should not influence its value.

Also, correlation dimension is easy to calculate (but computationally expensive).
more on the Grassberger-Procaccia algorithm: http://www.scholarpedia.org/article/Grassberger-Procaccia_algorithm

# GP.py

This code takes in input a timeseries of scalar values and the embedding dimension + time delay necessary to perform a time-delay embedding in phase space to reconstruct the attractor.

GP algorithm and utility functions (such as delay embedding) to calculate the Correlation dimension from points reconstructed in phase space
