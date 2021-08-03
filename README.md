# `frft`: Multi-dimensional fractional Fourier transform in Python
<img src="WavePropagation.gif">
<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript" ></script>

## Introduction

The fractional Fourier transform (FRFT) is a valuable signal processing tool used in optics, physics, and radar engineering. 
It generalizes the familiar Fourier transform into real/reciprocal phase space as a partial rotation between these two spaces. 
Signal information represented in the fractional Fourier space can sometimes be far more illuminating than the Fourier transform. 

## Numerical convention
The definition of the one-dimensional FRFT for a parameter $\alpha \in \mathbb{R}$ can be [found in Wikipedia](https://en.wikipedia.org/wiki/Fractional_Fourier_transform#Definition). 
In our convention, $\alpha = 1$ indicates a quarter of the cycle of repetition, corresponding to a phase space 'rotation' of $\pi/2$ or $90^\circ$. 
Thus, the periodicity in $\alpha$ is $4$. 

## Implementation

The code uses Numpy's native `fftn` routine and therefore can be used for transforms of any dimensionality.
This implementation follows the theory described in Ozaktas _et al._ \cite{Ozaktas1996}. 
It computes the FRFT as the chirp-modulated signal convolved with a chirp function, followed by a final chirp modulation. 
The chirp function (_i.e._, a parabolic phase) resides as an array within the namespace of the imported module. 
It is recalculated whenever the dimensions of the input array change. 
1-D, 2-D and 3-D example data is provided as an HDF5 file `data.hdf5`. 

## Caveats
   1. Currently does not preserve signal norm. This has to be done manually. Will soon add a normalization parameter, similar to `norm="ortho"` in `numpy.fftn`. 
   1. Currently selective transforms along user-defined dimensions are not implemented; transforms along all dimensions of the input array by default. 

## Tutorial
[Here](https://github.com/siddharth-maddali/frft/blob/main/tutorial.ipynb) is a basic tutorial.

In these tutorials, the FRFT will be simulated in the range $\alpha \in [0, 2]$. 
This corresponds to the original signal ($\alpha = 0$) through to its Fourier transform ($\alpha = 1$), and ending with the inverted signal ($\alpha = 2$). 


# Acknowledgements

   * Dr. Ishwor Poudyal (Argonne National Laboratory) for generating the example data.

# References

(<a id="cit-Ozaktas1996" href="#call-Ozaktas1996">Ozaktas, Arikan <em>et al.</em>, 1996</a>) Ozaktas Haldun M, Arikan Orhan, Kutay M Alper and Bozdagt Gozde, <em>Digital computation of the fractional Fourier transform</em>, **<em>IEEE Transactions on signal processing</em>**, vol. 44, number 9, pp. 2141--2150,  1996.


