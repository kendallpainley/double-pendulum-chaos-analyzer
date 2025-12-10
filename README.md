# Double Pendulum Chaos Analyzer 

A numerical simulation analyzing the chaotic dynamics and phase space of the double pendulum system using Python and SciPy.

## Key Skills Demonstrated 
* **Mathematical Modeling:** Application of **Lagrangian Mechanics** to derive the coupled, non-linear Ordinary Differential Equations (ODEs).
* **Numerical Methods:** Implementation of numerical integration (via `scipy.integrate.solve_ivp`) for system stability.
* **Chaos Theory:** Quantitative analysis of **Sensitivity to Initial Conditions**.
* **Visualization:** Creating complex **Phase Space** plots and real-time **Animations** of the chaotic trajectory. 

## Project Files

* `src/double_pendulum_solver.py`: Contains the core Python function that defines the system's differential equations and calls the SciPy ODE solver.
* `analysis_notebook.ipynb`: **Primary Showcase**—A detailed Jupyter Notebook explaining the model, validating energy conservation, performing chaotic analysis (Poincaré section, divergence), and visualizing results.
* `requirements.txt`: Lists all necessary dependencies (`numpy`, `scipy`, `matplotlib`, `jupyter`).

## How to Run Locally

1.  **Clone the repository:** `git clone https://github.com/your-username/double-pendulum-chaos.git`
2.  **Install dependencies:** `pip install -r requirements.txt`
3.  **Launch the notebook:** `jupyter notebook analysis_notebook.ipynb`

---
*Built with Python, SciPy, and Matplotlib.*
