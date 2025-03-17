# Tiny Matrix Multiplication

This repository solely focusses on matrix multiplications of a particular form (`C = A × B`) where operator matrices `A` have 10s to 100s of rows/columns, and operand matrices `B` range from 10⁶–10⁷ rows. 
Performance of these matrices are benchmarked with vendor-provided libraries on GPUs. 
The structure of these matrices allows various optimisation techniques to be applied, such as unrolling the operator matrices and cache-blocking etc. 
Our goal is to strive for maximum possible performance of these specific matrix operations on the GPUs.
