import numpy as np 
import pygame
from pygame.locals import *
import numpy as np
from numpy.linalg import inv
import sys
    
def kf_predict(X, P, Q, U):
    Fx = np.array([[1, 0, 0]
                  ,[0, 1, 0]
                  ,[0, 0, 1]])
    Fu = np.array([[dt*np.cos(X[2,0]), 0]
                  ,[dt*np.sin(X[2,0]), 0]
                  ,[0, dt]])
    X = X + np.array([ [dt*U[0][0]*np.cos(X[2,0])]
                      ,[dt*U[0][0]*np.sin(X[2,0])]
                      ,[dt*U[1][0]] ])
    P = Fx.dot(P.dot(Fx.T)) + Fu.dot(Q.dot(Fu.T)) 
    return(X,P)

def kf_update(X, P, Y, R):
    C = np.array([ [(X[0,0]-L)/np.sqrt((X[0,0]-L)**2+(X[1,0]-L)**2),
                    (X[1,0]-L)/np.sqrt((X[0,0]-L)**2+(X[1,0]-L)**2),
                    0]
               ,[-1*(X[1,0]-L)/np.sqrt((X[0,0]-L)**2+(X[1,0]-L)**2),
                    (X[0,0]-L)/np.sqrt((X[0,0]-L)**2+(X[1,0]-L)**2),
                    -1] ])
    IM = np.array([[np.sqrt( (X[0,0]-L)**2 
                          +  (X[1,0]-L)**2)]
                            ,[np.arctan2( X[1,0]-L , X[0,0]-L ) - X[2,0] ] ]) 
    IS = R + C.dot(P.dot(C.T))
    
    K = P.dot(C.T.dot(np.linalg.inv(IS)))
    X = X + K.dot(Y-IM)
    P = P - K.dot(C.dot(P)) 
    return (X,P,K)

# Initialize
dt = 1/8; r = 0.1; L = 10.0
ur = 20; ul = 0
uw = 1/2*(ur + ul); us = ur - ul 
if us ==0:
    N_iter = 100
else:
    N_iter = 2*np.pi*L/(dt*r*us)
rr = 0.1; rb = 0.01
ww = 0.10; ws = 0.01

trace = [(350,250),(350,250)]
GT = [(0,0),(0,0)]
G = [(0,0),(0,0)]
# Initialization of state matrices
X = np.array([[0.0], [0.0], [0.0]])
P = np.diag((0.1, 0.1, 0.1))
Q = np.array([[ww**2,0],[0,ws**2]])
Xv = np.array([[0.0],[0.0],[0.0]])
# Measurement matrices
R = np.array([[rr**2,0],[0,rb**2]])
# Applying the Kalman Filter - PyGame
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 50, 50)
GREEN = (0, 255, 0)
BLUE = (50, 50, 255)
pygame.init()
SCREEN_SIZE = (700, 700)
screen = pygame.display.set_mode(SCREEN_SIZE)
pygame.display.set_caption('Extended Kalman Filter - 2D Robot')
pygame.display.flip()
screen.fill(WHITE)
scale = 20
sf = 300 #250
fps = pygame.time.Clock()
paused = False
i = 0
while True:
    i += 1
    pygame.time.delay(5)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit(); sys.exit();
    pygame.draw.circle(screen, RED, (350,350), 5)
    if i<N_iter:
        W = np.array([[np.random.normal(0,ww)],[np.random.normal(0,ws)]])
        U = np.array([[r*uw],[r/L*us]]) + W
        (X, P) = kf_predict(X, P, Q, U)
        real_x = X[0][0]*scale + 350
        real_y = X[1][0]*scale + 250
        Xv = np.append(Xv,X,1)
        Px = np.sqrt(P[0][0])*sf
        Py = np.sqrt(P[1][1])*sf
        
        trace[1] = (int(real_x), int(real_y))
        pygame.draw.lines(screen, BLUE, False, trace, 3)
        trace[0] = trace[1]
        
        GT[1] = (GT[0][0] + dt*r*uw*np.cos(i*dt*r/L*us), GT[0][1] + dt*r*uw*np.sin(i*dt*r/L*us))
        G = [(GT[0][0]*scale + 350, GT[0][1]*scale + 250),(GT[1][0]*scale + 350 , GT[1][1]*scale + 250)]
        pygame.draw.lines(screen, GREEN, False, G, 2)
        GT[0] = GT[1]
        if i%20 == 0:
            Y =  np.array([[np.sqrt( (X[0,0]-L)**2 + (X[1,0]-L)**2) 
                                                   + np.random.normal(0,rr)]
                          ,[np.arctan2(X[1,0]-L,X[0,0]-L)-X[2,0] 
                                                   + np.random.normal(0,rb)]]) 
            (X, P, K)= kf_update (X, P, Y, R)
            pygame.draw.circle(screen, RED, (int(real_x), int(real_y)), 3)
            pygame.draw.ellipse(screen, BLACK, (real_x - Px, real_y - Py, 2*Px, 2*Py), 1)
            pygame.draw.line(screen, BLACK, (350,350), (int(real_x), int(real_y)), 2)
        pygame.display.update()