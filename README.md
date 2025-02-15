# LqSolver

**LqSolver** is a Python class designed to solve the $\(\ell_q\)$-norm regression problem of the form:

$$\min_{\mathbf{x}} \|\mathbf{A}\mathbf{x} - \mathbf{b}\|_{q}$$,

where $\(\|\cdot\|_q)$ denotes the $\(\ell_q\)$-norm. The solver can handle different values of $\(q\)$ (e.g., 1.1, 1.4, 1.7, etc.) to perform robust regression with respect to outliers and noise.

## Table of Contents

1. [Overview](#overview)  
2. [Installation](#installation)  
3. [Usage](#usage)  
4. [API Reference](#api-reference)  
5. [Examples](#examples)  
6. [Implementation Details](#implementation-details)  
7. [Contributing](#contributing)  
8. [License](#license)

---

## Overview

- **What is $\(\ell_q\)$ regression?**  
  $\(\ell_q\)$ regression attempts to minimize the $\(q\)$-norm of residuals $\(\|\mathbf{A}\mathbf{x} - \mathbf{b}\|_q\)$. This can be used to achieve robustness (for \(1 < q < 2\)) and to mitigate the effect of outliers in data.

- **Why use `LqSolver`?**  
  `LqSolver` provides a straightforward API for setting up and solving $\(\ell_q\)$-norm regression problems. It allows you to experiment with different $\(q\)$ values to observe how robust regression might behave compared to, say, standard least-squares ($\(\ell_2\)$-norm) or $\(\ell_1\)$-norm solutions.

---

## Installation

1. **Clone or download this repository**
2. **Install required dependencies** (e.g., NumPy). For example:
   ```bash
   pip install -r requirements.txt
   ```
3. (Optional) If you want to integrate with your own environment, place the `LqSolver.py` file in your project directory or install it as a module using a local `setup.py`.

---

## Usage

### Importing the class

```python
from LqSolver import LqSolver
```

### Constructing the Solver

```python
import numpy as np

# Example matrix A (shape: n x p)
A = np.random.randn(100, 5)

# Example vector b (shape: n x 1)
b = np.random.randn(100, 1)

# Instantiate the solver
solver = LqSolver(A, b)
```

### Solving the Problem

```python
# Solve for a chosen q (e.g., 1.4) with a maximum number of iterations
x, r, info = solver.Solve(q=1.4, max_iters=80)
```

Here, the solver returns:
- `x` : the solution vector
- `r` : residuals or any relevant info (optional)
- `info` : additional convergence info (optional)

---

## API Reference

### `class LqSolver(A, b)`

**Constructor**  
```python
LqSolver(A, b)
```
- **Parameters**:
  - `A` *(np.ndarray)*: The coefficient matrix of shape `(n, p)`.
  - `b` *(np.ndarray)*: The target/observation vector of shape `(n, 1)` or `(n,)`.

- **Description**:  
  Stores the matrix $\(\mathbf{A}\)$ and vector $\(\mathbf{b}\)$ internally for the $\(\ell_q\)$ regression problem.

---

### `LqSolver.Solve(q, max_iters=100)`

**Parameters**:
- `q` *(float)*: The norm parameter $\(q\)$. Typical values are in the range \((1, 2]\) or similar.  
- `max_iters` *(int, optional)*: Maximum number of iterations (default is 100).  
- Additional parameters are included for the specific case of line fitting

**Returns**:
- `x` *(np.ndarray)*: The estimated $\(\mathbf{x}\)$ that minimizes $\(\|\mathbf{A}\mathbf{x} - \mathbf{b}\|_q\)$.  
- `xL2` *(np.ndarray)*: The least squares solution 
- `info` *(dict, optional)*: A dictionary containing additional convergence information (iterations count, errors history, etc.).

**Description**:
Solves the regression problem:

$$
\min_{\mathbf{x}} \|\mathbf{A}\mathbf{x} - \mathbf{b}\|_q,
$$

**Notes**:
- If `q=2`, it reduces to a standard least-squares problem.
- If `q=1`, it approximates an $\(\ell_1\)$ (robust) regression.
- For values of `q` between 1 and 2, the solution can offer a trade-off between the robustness of $\(\ell_1\)$ and the computational efficiency and low variance of $\(\ell_2\)$.

---

## Examples

Below is a condensed version of typical usage:

```python
import numpy as np
import matplotlib.pyplot as plt
from LqSolver import LqSolver

# Generate synthetic data
n, p = 160, 5
A = np.random.randn(n, p) * 0.1
x_gt = np.random.randn(p, 1)
b_gt = A @ x_gt

# Add noise
noise_level = 20
noise = noise_level * np.linalg.norm(b_gt) / n / 10
b = b_gt + np.random.randn(n, 1) * noise

# Introduce outliers
outlier_ratio = 0.15
num_outliers = int(np.round(n * outlier_ratio))
indices = np.random.permutation(n)
b[indices[:num_outliers]] = np.random.rand(num_outliers, 1) * 2

# Solve with LqSolver for q=1.4
solver = LqSolver(A, b)
x_est, _, _ = solver.Solve(q=1.4, max_iters=80)

# Compute RMSE
rmse = np.sqrt(np.mean((x_gt - x_est)**2))

print("Estimated x:\n", x_est)
print("RMSE:", rmse)
```

---

## Implementation Details

`LqSolver` uses a constant number of iterations (based on $\Delta q$) to approximate the $\(\ell_q\)$-norm solution:

1. **Initialization**: Start from an initial guess $\(\mathbf{x}^{(0)}\)$ (typically least square solution).
2. **Residuals**: Calculated residuals $A\mathbf{x}^{(0)}-b$
3. **Iterate**: (Repeat 4+5 for a preset number of iterations) 
4. **Update**: Update the solution using $\overrightarrow{\Delta x}=\frac{\Delta q}{q_0-1}(A^TA)^{-1}A^T\log(|{\overrightarrow{r}}|)\overrightarrow{r}$.
5. **Residuals**: Calculated residuals $A\mathbf{x}^{(i)}-b$

**Complexity**: We can calculate $(A^TA)^{-1}A^T$ once at a computational cost of roughly $\(O(n^3)\)$,
Each iteration is therefore of complexity $\(O(n^2)\)$ giving a total of $\(O(n^3)\)$.
We may change the solver to use a simple linear solver for each step instead of precalculating $(A^TA)^{-1}A^T$,
for better numerical stability.

---

## Contributing

Contributions are welcome! If you want to add features, fix bugs, or propose enhancements:

1. Fork the repository.  
2. Create a new branch for your feature or bug fix.  
3. Commit your changes.  
4. Submit a Pull Request describing your work.

---

## License

This project is licensed under the terms of the [MIT License](LICENSE). See the [LICENSE](#license-text) file for details.

### License Text

```
MIT License

Copyright (c) [Year] [Author]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights  
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is  
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included  
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR  
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,  
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE  
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER  
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,  
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN  
THE SOFTWARE.
```
