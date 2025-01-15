import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

'''
Attention,
lorsqu'on regarde la courbe au niveau d'un point d'équilibre proche de epsilon, il y a du bruit numérique
qui crée une courbe fictive. Il faut se fier aux grandeurs indiqués dans les axes pour déterminer s'il y a 
stabilité ou instabilité.
'''
# Paramètres globaux
N = 5000  # Nombre de points

b = float(input("Donne ta valeur de b > 0 : "))
c = float(input("Donne ta valeur de c > 0 : "))

print("Il va y avoir un point d'équilibre stable en x0 si a est strictement supèrieur de : ", max(1/b - 1/c, 1/b - c))
print("Il va y avoir trois points d'équilibres si a est strictement infèrieur de :",1/b - 1/c)
print("Donc soit il y a trois point d'équilibre sinon x0 est stable.")
a = float(input("Choisis ta valeur de a > 0 : "))
t = np.linspace(0, 100, N)

# Fonction Euler
def Euler(initialCondition):
    x0, y0, z0 = initialCondition
    delta = t[1] - t[0]
    x = [x0] + [0] * (N - 1)
    y = [y0] + [0] * (N - 1)
    z = [z0] + [0] * (N - 1)
    for n in range(N - 1):
        x[n+1] = x[n] + delta * (z[n] + (y[n] - a) * x[n])
        y[n+1] = y[n] + delta * (1 - b * y[n] - x[n]**2)
        z[n+1] = z[n] + delta * (-x[n] - c * z[n])
    return np.array(x), np.array(y), np.array(z)

# Points d'équilibre
epsilon = 1e-3
X0 = (0+epsilon, 1 / b + epsilon, 0 + epsilon)
x0, y0, z0 = Euler(X0)
if  a < 1/b - 1/c:
    print("Le système a 3 points d'équilibres")
    X1 = (np.sqrt(1 - b * a - b / c) + epsilon, a + 1 / c + epsilon, -1 / c * np.sqrt(1 - b * a - b / c) + epsilon)
    X2 = (-np.sqrt(1 - b * a - b / c)+ epsilon, a + 1 / c + epsilon , 1 / c * np.sqrt(1 - b * a - b / c) + epsilon)
    # Calcul des trajectoires
    x1, y1, z1 = Euler(X1)
    x2, y2, z2 = Euler(X2)
else:
    print("Le système a exactement 1 point d'équilibre")


# Fonction pour créer une animation pour chaque figure
def create_animation(fig, x, y, z, title):
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(title)
    margin = 1
    ax.set_xlim(np.min(x) - margin, np.max(x) + margin)
    ax.set_ylim(np.min(y) - margin, np.max(y) + margin)
    ax.set_zlim(np.min(z) - margin, np.max(z) + margin)
    line, = ax.plot([], [], [], lw=2)

    def init():
        line.set_data([], [])
        line.set_3d_properties([])
        return line,

    def update(frame):
        line.set_data(x[:frame], y[:frame])
        line.set_3d_properties(z[:frame])
        return line,

    ani = FuncAnimation(fig, update, frames=len(t), init_func=init, blit=False, interval=50)
    return ani


# Calcul algébrique des valeurs propres de la Jacobienne aux différents points d'équilibres

def J(X):
    x, y, z = X
    return np.array([[y-a, x, 1], [-2*x, -b, 0], [-1, 0, -c]])


print("Les valeurs propres de X0 : ", np.linalg.eig(J(X0))[0])
if a < 1/b - 1/c:
    print("Les valeurs propres de X1 : ", np.linalg.eig(J(X1))[0])
    print("Les valeurs propres de X2 : ", np.linalg.eig(J(X0))[0])


# Création des trois figures et animation



fig0 = plt.figure()
ani0 = create_animation(fig0, x0, y0, z0, "Trajectoire X0")

if 1 - b * a - b / c >= 0:
    fig1 = plt.figure()
    ani1 = create_animation(fig1, x1, y1, z1, "Trajectoire X1")

    fig2 = plt.figure()
    ani2 = create_animation(fig2, x2, y2, z2, "Trajectoire X2")

# Affichage
plt.show()

