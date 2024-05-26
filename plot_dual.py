import matplotlib.pyplot as plt
import numpy as np

def plot_feasible_region(ax, constraints, label, color):
    x = np.linspace(0, 5, 400)
    y = np.linspace(0, 5, 400)
    X, Y = np.meshgrid(x, y)
    feasible = np.ones_like(X, dtype=bool)

    for constraint in constraints:
        feasible &= constraint
    
    ax.contourf(X, Y, feasible, levels=1, colors=[color], alpha=0.5)
    ax.plot([], [], color=color, alpha=0.5, label=label)
    
def primal_constraints(X, Y):
    return [
        X + Y <= 4,
        X <= 2,
        Y <= 3,
        X >= 0,
        Y >= 0
    ]

def dual_constraints(U, V):
    return [
        U + V >= 3,
        U >= 0,
        V >= 0,
        U + (3 - V) >= 2
    ]

fig, ax = plt.subplots()

# Define meshgrid for primal variables
X, Y = np.meshgrid(np.linspace(0, 5, 400), np.linspace(0, 5, 400))
# Define meshgrid for dual variables
U, V = X, Y

# Plot primal feasible region
plot_feasible_region(ax, primal_constraints(X, Y), 'Primal Feasible Region', 'blue')

# Plot dual feasible region
plot_feasible_region(ax, dual_constraints(U, V), 'Dual Feasible Region', 'green')

# Plot settings
ax.set_xlim(0, 5)
ax.set_ylim(0, 5)
ax.set_xlabel('x (Primal) / u (Dual)')
ax.set_ylabel('y (Primal) / v (Dual)')
ax.legend()
ax.grid(True)
plt.title('Primal and Dual Feasible Regions')
plt.show()
