import numpy as np
from scipy.integrate import solve_ivp

# Define global constants (or pass them in a 'p' array if you prefer OOP)
G = 9.81  # Acceleration due to gravity (m/s^2)

def pendulum_derivs(t, y, L1, L2, M1, M2):
    """
    Calculates the derivatives for the double pendulum system.

    Parameters:
    - t: Current time (not used, as the system is autonomous)
    - y: State vector [theta1, omega1, theta2, omega2]
    - L1, L2, M1, M2: System parameters (lengths and masses)

    Returns:
    - dy: Derivative vector [d(theta1)/dt, d(omega1)/dt, d(theta2)/dt, d(omega2)/dt]
    """
    theta1, omega1, theta2, omega2 = y

    # Pre-calculate common terms for efficiency and readability
    d_theta = theta1 - theta2
    cos_d = np.cos(d_theta)
    sin_d = np.sin(d_theta)

    # Denominators for the second derivatives (accelerations)
    # Den = L1 * (2*M1 + M2 - M2 * cos_d**2)
    # Den2 = L2 * (2*M1 + M2 - M2 * cos_d**2)
    
    # Common factor for the denominators
    Den_Factor = (M1 + M2 * sin_d**2) 
    
    # Numerator for d(omega1)/dt (angular acceleration 1)
    num1 = -G * (2 * M1 + M2) * np.sin(theta1)
    num1 -= M2 * G * np.sin(theta1 - 2 * theta2)
    num1 -= 2 * sin_d * M2 * (omega2**2 * L2 + omega1**2 * L1 * cos_d)
    
    # Numerator for d(omega2)/dt (angular acceleration 2)
    num2 = 2 * sin_d * (omega1**2 * L1 * (M1 + M2) + G * M2 * np.cos(theta1) + omega2**2 * L2 * M2 * cos_d)
    num2 -= M2 * L2 * omega2**2 * sin_d * cos_d
    num2 += G * np.sin(theta2) * (M1 + M2)
    num2 *= 2
    num2 -= 2 * G * M2 * np.sin(theta2) * cos_d
    
    # Corrected full equations for angular accelerations (simplified denominator for clarity)
    # These derived equations are notoriously complex and can vary slightly based on convention.
    # We will use the standard derived form for a double pendulum:
    
    # Angular acceleration for mass 1
    den1 = L1 * (2 * M1 + M2 - M2 * cos_d**2)
    domega1_dt = (M2 * L1 * omega1**2 * sin_d * cos_d + M2 * G * np.sin(theta2) * cos_d - M2 * L2 * omega2**2 * sin_d * cos_d - (M1 + M2) * G * np.sin(theta1)) / den1
    
    # Angular acceleration for mass 2
    den2 = L2 * (2 * M1 + M2 - M2 * cos_d**2)
    domega2_dt = (-M2 * L2 * omega2**2 * sin_d * cos_d + (M1 + M2) * G * np.sin(theta1) * cos_d - (M1 + M2) * L1 * omega1**2 * sin_d + M2 * G * np.sin(theta2)) / den2
    
    # Final derivative vector
    return [omega1, domega1_dt, omega2, domega2_dt]



def run_simulation(initial_conditions, L1=1.0, L2=1.0, M1=1.0, M2=1.0, t_end=30, t_steps=1001):
    """
    Runs the double pendulum simulation using scipy's solve_ivp.

    Parameters:
    - initial_conditions: State vector [theta1_0, omega1_0, theta2_0, omega2_0]
    - L1, L2, M1, M2: System parameters
    - t_end: Total simulation time in seconds
    - t_steps: Number of time points to return

    Returns:
    - results: An array containing the time and the solution (y(t))
    """
    # Time points for which to store the solution
    t_span = [0, t_end]
    t_points = np.linspace(0, t_end, t_steps)

    # Use 'RK45' (Runge-Kutta 5th order) for stability in complex ODEs
    # The 'args' tuple passes the constant parameters to pendulum_derivs
    solution = solve_ivp(
        pendulum_derivs, 
        t_span, 
        initial_conditions, 
        t_eval=t_points, 
        args=(L1, L2, M1, M2),
        method='RK45' 
    )

    # Transpose the solution array for easier plotting (t, theta1, omega1, theta2, omega2)
    results = np.vstack([solution.t, solution.y])
    return results




def calculate_energy(y, L1, L2, M1, M2):
    """
    Calculates the total mechanical energy (Potential + Kinetic) of the system.
    
    Parameters:
    - y: Full solution array (5 rows: t, th1, om1, th2, om2)
    - L1, L2, M1, M2: System parameters

    Returns:
    - Total energy array
    """
    # Extract states from the solution array
    theta1, omega1, theta2, omega2 = y[1], y[2], y[3], y[4]
    
    # Calculate Potential Energy (V)
    # V = -G * [(M1 + M2) * L1 * cos(theta1) + M2 * L2 * cos(theta2)]
    V = -(M1 + M2) * G * L1 * np.cos(theta1) - M2 * G * L2 * np.cos(theta2)

    # Calculate Kinetic Energy (T)
    # T = 0.5 * (M1 + M2) * (L1 * omega1)**2 + 0.5 * M2 * (L2 * omega2)**2 + M2 * L1 * L2 * omega1 * omega2 * cos(theta1 - theta2)
    T = 0.5 * M1 * (L1 * omega1)**2
    T += 0.5 * M2 * ((L1 * omega1)**2 + (L2 * omega2)**2 + 2 * L1 * L2 * omega1 * omega2 * np.cos(theta1 - theta2))
    
    # The formula used is for masses M1 and M2 at the end of the rods, with masses M_rod = 0.
    
    return T + V
