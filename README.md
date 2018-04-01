# smith-normal-form
Simple module for computing Smith Normal Form of a numpy array over Z_2

# Usage

```
import numpy as np
from snf import SmithNormalForm
bm =  np.array([[1, 1, 0, 0, 0, 0, 0],
                [1, 0, 1, 1, 1, 0, 0],
                [0, 1, 1, 0, 0, 1, 0],
                [0, 0, 0, 1, 0, 1, 1],
                [0, 0, 0, 0, 1, 0, 1]])


smithers = SmithNormalForm()
bm_snf = smithers.smithify(bm)

>>> array([[1, 0, 0, 0, 0, 0, 0],
           [0, 1, 0, 0, 0, 0, 0],
           [0, 0, 1, 0, 0, 0, 0],
           [0, 0, 0, 1, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0]])
```

