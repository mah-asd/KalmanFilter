import numpy as np 
import pygame
from pygame.locals import *
import numpy as np
from numpy.linalg import inv
import sys
    
def kf_predict(X, P, A, Q, B, U):
    X = A.dot(X) + B.dot(U)
    P = A.dot(P.dot(A.T)) + Q
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
r = 0.1
ur = 10; ul = 10
rx = 0.05; ry = 0.075;
wx = 0.10; wy = 0.15;
sf = 80
trace = [(0,0),(0,0)]
GT = [(0,0),(0,0)]

# Initialization of state matrices
X = np.array([[0.0], [0.0]])
P = np.diag((0, 0))
A = np.eye(2)
Q = np.array([[wx**2,0],[0,wy**2]])
B = dt*A
Xv = np.array([[0.0],[0.0]])
# Measurement matrices
C = np.array([[1, 0], [0, 2]])
R = np.array([[rx**2,0],[0,ry**2]])
# Applying the Kalman Filter - PyGame
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 50, 50)
GREEN = (0, 255, 0)
BLUE = (50, 50, 255)
pygame.init()
SCREEN_SIZE = (800, 800)
screen = pygame.display.set_mode(SCREEN_SIZE)
pygame.display.set_caption('Kalman Filter - 2D Robot')
pygame.display.flip()
screen.fill(WHITE)
scale = 100
fps = pygame.time.Clock()
paused = False
i = 0
while True:
    i += 1
    pygame.time.delay(200)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit(); sys.exit();
    if i<99:
        W = np.array([[np.random.normal(0,wx)],[np.random.normal(0,wy)]])
        U = np.array([[r/2*(ur+ul)],[r/2*(ur+ul)]]) + W
        (X, P) = kf_predict(X, P, A, Q, B, U)
        real_x = X[0][0]*scale
        real_y = X[1][0]*scale
        
        Px = np.sqrt(P[0][0])*sf
        Py = np.sqrt(P[1][1])*sf
        
        trace[1] = (int(real_x), int(real_y))
        pygame.draw.lines(screen, BLUE, False, trace, 3)
        trace[0] = trace[1]
        
        GT[1] = (dt*i*r/2*(ur+ul)*scale, dt*i*r/2*(ur+ul)*scale)
        pygame.draw.lines(screen, GREEN, False, GT, 2)
        GT[0] = GT[1]
        
        if i%4 == 0:
            rxy = np.array([[np.random.normal(0,rx)],[np.random.normal(0,ry)]]);
            Y = C.dot(X) + rxy
            X, P , K = kf_update (X, P, Y, C, R)
            Xv = np.append(Xv,X,1)
        pygame.draw.circle(screen, RED, (int(real_x), int(real_y)), 3)
        pygame.draw.ellipse(screen, BLACK, (real_x-Px, real_y-Py, 2*Px, 2*Py), 2)

    pygame.display.update()