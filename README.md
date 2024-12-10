# Nested Normalizing Flow (NF2NF)

Nested Normalizing Flow to increase the expressiveness of the flow model, taking note that the base flow of a normal RealNVP is just a linear transformation w.r.t. the coefficient output by NN. 
To increase the expressiveness, we introduce the use of conditional normalizing flow to replace the linear transformation used in most flow architecture (RealNVP, Glow, etc.).
Higher level of nested should give us better expressiveness with minimal parameters increase.

**Note:** This is temporarily pending because of some problem in the numerical stability of the flow model. This is due to highly complex function approximator (a NN) which make the inverse pass unstable.
