"""
==============================
Comparison Trapezoidal and Simpson's.

* Compares Trapezoidal and Simpson's method for numerical integration.
==============================
"""

import numpy as np
from projects.nnum.integration import TrapezoidalIntegrator, SimpsonIntegrator

params = {
    'f': lambda x: np.log(x),
    'n': 100,
    'a': 1, 'b': 4
}


print(__doc__)

result = TrapezoidalIntegrator(**params).integrate()
print('Int(sin)|[0,pi]=%f (Trapezoidal rule)' % result)

result = SimpsonIntegrator(**params).integrate()
print('Int(sin)|[0,pi]=%f (Simpson\'s rule)' % result)
