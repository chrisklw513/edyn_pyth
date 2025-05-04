import numpy as np 
import matplotlib.pyplot as plt 

"""
1. Gebiet definieren
"""

nx = 500
ny = 500

Lx = 1 
Ly = 1 

x = np.linspace(-Lx, Lx, nx)
y = np.linspace(-Ly, Ly, ny) 

dx = x[1] - x[0]
dy = y[1] - y[0]

X, Y = np.meshgrid(x,y, indexing = 'ij')

"""
2. rho(x,y) und eps 
"""

rho = -25*(X+Y) 
eps = 1 

"""
3. u initialisieren fuer A*u = b  
"""

u = np.zeros((nx,ny)) 

u_bot = u_left = -2 

u_top = u_right = 2

u[0,:] = u_left 
u[-1,:] = u_right

u[:,0] = u_bot
u[:,-1] = u_top


"""
Sparse Matrix erstellen um A * u = b nach u aufloesen zu koennen (5-Stern Laplace mit Finite Differenzen)
"""
from scipy.sparse import kron, diags, identity
from scipy.sparse.linalg import spsolve

nx_in = nx-2 
ny_in = ny-2 
 
'''diags -> [1,-2,1] und [-1,0,1] machen auf der Hauptdiagonalen -2 mit Nebendiag 1 und 1 (dafuer [-1,0,1] als Verschiebung der Werte 1)'''

Dx = 1/dx**2 *diags([1, -2, 1], [-1, 0, 1], shape=(nx_in, nx_in))
Dy = 1/dy**2 *diags([1, -2, 1], [-1, 0, 1], shape=(nx_in, nx_in))

ix = identity(nx_in) # Erzeugt eine Einheitsmatrix der dimention nx_in 
iy = identity(ny_in) # Erzeugt eine Einheitsmatrix der dimention ny_in 

A_fd = kron(ix,Dy) + kron(Dx,iy)  # Führt Kornecker Mult. durch -> Liefert Matrix in der Form die wir brauchen 

""" 
rechte Seite vorbereiten 
"""
b = -rho[1:-1, 1:-1]/ eps  


b[0,:]  -= u_left  / dx**2  #Randwerte beruecksichtigen nach Finite Differenzen Gleichung 
b[-1,:] -= u_right / dx**2  

b[:,0]  -= u_bot  / dy**2 
b[:,-1] -= u_top  / dy**2 

b_sol = b.flatten()
"""
A*u=b >>> u = spsolve(u_in,b) 
"""
u_sol_in = spsolve(A_fd,b_sol)

"""
das innere von u wieder hinzufuegen 
"""
u[1:-1, 1:-1] = u_sol_in.reshape((nx_in,ny_in))

"""
Potential darstellen 
"""
plt.figure(figsize=(10,5))
plt.contourf(X, Y, u, levels=50, cmap="jet")
plt.colorbar(label=r'\\phi(x,y)')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
