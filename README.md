## Reduced Instruction Set Machine Learning (RISML) [![BuildStatus](https://travis-ci.com/njkrichardson/picard.svg?branch=master)](https://travis-ci.com/njkrichardson/picard)

RISML is a Python package containing education-oriented reference implementations of a variety of commonly used
algorithms in machine learning and scientific computing. RISML is educational software in the sense that the
implementation is designed to be clear, and easy to understand, even for inexperienced programmers. This is in contrast
to hardened, optimized implementations for production use. Writing powerful and optimized implementations of numerical
software for a particular computer architecture is often in tension with writing simple code with clear correspondence
to the kinds of psuedocode descriptions often seen in textbooks. That is to say, many of RISML's implementations are
less performant (in terms of ops/s, memory utilization, numerical stability) than counterparts found in mature libraries
like [BLAS](http://www.netlib.org/blas/), [SciPy](https://scipy.org/), or Intel's [MKL](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html#gs.h1gi8i). 

### Intended Audience 

RISML is primarily intended for use by students in computing, engineering, and mathematics aiming to learn more about
the application of machine learning and scientific computing. The only requisite skills needed to interact with the
package are some experience programming in a high-level language, and an interest in looking under the covers at how
machine learning applications are developed and implemented. 

### What is a reduced instruction set?  In computer architecture, **reduced** and **complex** are terms used to loosely
distinguish between the instruction set architectures of different machines. A reduced instruction set computer (RISC)
architecture like ARM or MIPS exposes a relatively simple and limited instruction set (e.g., operations like ADD, MOV,
SUB, and BRANCH) to the programmer; in practice this often means that each instruction requires fewer clock cycles to
execute, though any given program will tend to compile to more instructions. In contrast, complex instruction set
computer (CISC) architectures like x86 or IA-32 expose much more complex instructions (e.g., polynomial multiplication
or string manipulation) which results in programs with fewer instructions, though each instruction requires more cycles
to execute. 

Drawing inspiration from this taxonomy, RISML provides Python implementations of many of the procedural building blocks
required to implement software for machine learning and scientific computing. One could consider a package like SciPy as
providing a more CISC-like API in that many of the component procedures required to implement an algorithm, e.g.,
principal components analysis (PCA) are abstracted from the user. In fact, RISML contains a single module `isa.py`
containing almost all of the components needed to develop full from-scratch implementations of the higher level
algorithms. Many of these building blocks are numerical routines for linear algebraic procedures: much like would be
found in a mature library like [BLAS](http://www.netlib.org/blas/),[MKL](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html#gs.h1gi8i), or
[LAPACK](http://www.netlib.org/lapack/). An example is shown below of the fundamental operation utilized in executing a
neural network; matrix-matrix multiplication. 

```python 
def matrix_matrix_multiplication(A: np.ndarray, B: np.ndarray, C: np.ndarray, alpha: float, beta: float) -> None: 
    m, n, k = A.shape, B.shape[1] 
    
    for i in range(m): 
        for j in range(n): 
            C[i, j] *= beta 
            for r in range(k): 
                C[i, j] += alpha * A[i][k] * B[k][j] 
    
    return C 
```

For production applications, using mature libraries with higher-level APIs is generally best practice. At the time I'm
writing this (November of 2021), Python programmers interested in scientific computing and machine learning must gain
some familiarity with a standard set of libraries (e.g., NumPy, SciPy, an automatic differentiation package). With this
in mind, although many reference implementations are provided using RISML alone, the comprehensive set is implemented
using NumPy for an array abstraction/linear alebra package and Jax for automatic differentiation. I avoid non idiomatic
uses of both NumPy and Jax where possible, and introductory tutorials for these two libraries are forthcoming. 


### How to get started RISML contains a variety of short example use-case scripts with accompanying visuals rendered
with [Matplotlib](https://matplotlib.org/). Perusing these examples and the figures they produce are a great way to get
started.  

Below, the figure displayed contains a 2D scatter plot, whose constituent points correspond to linear projections of a
collection of one thousand 784-vectors representing images of handwritten numeric digits. The images are grayscale with
28 x 28 pixels, so each element of a 784-vector representing the image corresponds to a 4 byte pixel value. The method
used to produce the projection matrix is termed principal components analysis (PCA) and is implemented in
`src/dimensionality_reduction.py`. The figure can be recreated by executing `python3 examples/dimension_reduction.py
--save_figs`; see below: 

```python 
from dimensionality reduction import pca 

images: np.ndarray = np.load(...) 
labels: np.ndarray = np.load(...) 

num_images: int = 1000 
image_size: int = images[0].size 
reducing_dimension: int = 2 

X = images[:num_images].reshape(num_images, image_size) 
Z = pca(X, rank=reducing_dimension) 
```

Although we've compressed the representation of each image to a simple 2-vector (i.e., compressing a 3MB image to 8B),
the method we've used preserves the linear correlational structure of the _collection_ of images, so coloring each
compressed representation (each point) by its actual numeric value (whether the image contains a '7', for example)
illustrates interesting semantic _locality_ in the vector space of compressed representations; often called the "latent"
space. 

![mnistpca](https://github.com/njkrichardson/risml/blob/master/mnist_pca_encodings.pca?raw=true)

The implementations provided in RISML are in correspondence with the associated text; Ruye Wang's _Machine Learning:
From Theory to Code_, a forthcoming book for senior undergraduates in mathematics, engineering, and computing. The book
is scheduled to be available through Cambridge University Press in 2021. 

--- ### Installation 

**HTTPS** ```bash $ git clone https://github.com/njkrichardson/risml.git ```

---
