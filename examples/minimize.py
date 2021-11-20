from argparse import ArgumentParser 
import pathing 

import jax.numpy as np 
import matplotlib.pyplot as plt

from minimization import newton

parser = ArgumentParser(description="Newton's method example.")
parser.add_argument("-v", "--verbose", action="store_true", help="verbosity")
parser.add_argument("--plot", action="store_true", help="render a plot of the function and optimization path")
args = parser.parse_args()

if __name__=="__main__": 
    f = lambda x: np.power(x, 2)
    x_initial = np.array((3.), dtype=float)

    newton(f, x_initial, verbose=args.verbose)

    if args.plot: 
        domain = np.linspace(-3., 3., 20)
        plt.figure(1) 
        plt.plot(domain, f(domain))
        plt.show()
