# WFC_classical_ESTM
## requirements:-
Pygame, Colorama
(Both of these are for visualization purposes only, you can comment that part and algorithm still runs perfectly).
This is a classical implementation of super simple wave function collapse algorithm using a ESTM (Even simpler Tiled Model) in python.
THe algorithm is an example of constraint programming and functions by first initializing superposition of input state in output matrix and then finding and collapsing least entropy state. The algorithm then iterates over neighbours and propogate the changes. the steps are repeated untill we get all states in ouput matrix collapsed or a contradiction( which is rare considering large solution space and available options for output).
 I have added tile output visualization using pygame
### This is an adaption and modification of WFC algorithm by [Robert heaton](https://robertheaton.com/2018/12/17/wavefunction-collapse-algorithm/)
### The original WFC algorithm was proposed by [Maxim Gumin](https://github.com/mxgmn/WaveFunctionCollapse)
