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

def kf_update(X, P, Y, C, R):
    IM = C.dot(X)
    IS = R + C.dot(P.dot(C.T))
    
    K = P.dot(C.T.dot(np.linalg.inv(IS)))
    X = X + K.dot(Y-IM)
    P = P - K.dot(C.dot(P)) 
    return (X,P,K)

# Initialize
dt = 1/8
r = 0.1; L = 10.0
ur = 20; ul = 5
uw = 1/2*(ur + ul); us = ur - ul 
N_iter = 2*np.pi*L/(dt*r*us)
rx = 0.1; ry = 0.1; rt = 0.001
ww = 0.10; ws = 0.01;
sf = 250
trace = [(350,250),(350,250)]
GT = [(0,0),(0,0)]
G = [(0,0),(0,0)]

# Initialization of state matrices
X = np.array([[0.0], [0.0], [0.0]])
P = np.diag((0.0, 0.0, 0.0))
Q = np.array([[ww**2,0],[0,ws**2]])
Xv = np.array([[0.0],[0.0],[0.0]])
# Measurement matrices
C = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
R = np.array([[rx**2,0,0],[0,ry**2,0],[0,0,rt**2]])
# Applying the Kalman Filter - PyGame
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 50, 50)
GREEN = (0, 255, 0)
BLUE = (50, 50, 255)
pygame.init()
SCREEN_SIZE = (700, 700)
screen = pygame.display.set_mode(SCREEN_SIZE)
pygame.display.set_caption('Kalman Filter - 2D Robot')
pygame.display.flip()
screen.fill(WHITE)
scale = 20
fps = pygame.time.Clock()
paused = False
i = 0
while True:
    i += 1
    pygame.time.delay(5)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit(); sys.exit();
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
        
        if i%8 == 0:
            rxyt = np.array([[np.random.normal(0,rx)],[np.random.normal(0,ry)],[np.random.normal(0,rt)]]);
            Y = C.dot(X) + rxyt
            (X, P , K)= kf_update (X, P, Y, C, R)
            pygame.draw.circle(screen, RED, (int(real_x), int(real_y)), 3)
            pygame.draw.ellipse(screen, BLACK, (real_x-Px, real_y-Py, 2*Px, 2*Py), 1)

    pygame.display.update()