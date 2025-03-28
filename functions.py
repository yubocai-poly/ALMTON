#Supported Functions: Beale, Bohachevsky, Goldstein, Himmelblau, McCormick, Styblinski

import numpy as np
import math

def init_func(func_name):
    
    if func_name == 'Beale':
            
        def fX(X,Y):
            return (1.5 - X + X*Y)**2 + (2.25 - X + X*Y**2)**2 + (2.625 - X + X*Y**3)**2
        
        def fx(X):
            x = X[0][0]
            y = X[1][0]
            return (1.5 - x + x*y)**2 + (2.25 - x + x*y**2)**2 + (2.625 - x + x*y**3)**2
        
        def dx(X):
            x = X[0][0]
            y = X[1][0]
            return np.array([[2*x*(y**6 + y**4 - 2*y**3 - y**2 - 2*y + 3) + 5.25*y**3 + 4.5*y**2 + 3*y - 12.75],\
                             [6*x*(x*(y**5 + (2/3)*y**3 - y**2 - (1/3)*y - 1/3) + 2.625*y**2 + 1.5*y + .5)]])
        
        def d2x(X):
            x = X[0][0]
            y = X[1][0]
            ret = np.zeros((2,2))
            ret[0,0] = 2*y**6 + 2*y**4 - 4*y**3 - 2*y**2 - 4*y + 6
            ret[0,1] = ret[1,0] = 12*x*y**5 + 8*x*y**3 - 12*x*y**2 - 4*x*y - 4*x + 15.75*y**2 + 9*y + 3
            ret[1,1] = 30*x**2*y**4 + 12*x**2*y**2 - 12*x**2*y - 2*x**2 + 31.5*x*y + 9*x
            return ret
        
        def d3x(X):
            x = X[0][0]
            y = X[1][0]
            ret = np.zeros((2,2,2))
            ret[0,0,1] = ret[0,1,0] = ret[1,0,0] = 12*y**5 + 8*y**3 - 12*y**2 - 4*y - 4
            ret[1,1,0] = ret[1,0,1] = ret[0,1,1] = 60*x*y**4 + 24*x*y**2 - 24*x*y - 4*x + 31.5*y + 9
            ret[1,1,1] = 120*x**2*y**3 + 24*x**2*y - 12*x**2 + 31.5*x
            return ret;
        
    elif func_name == 'Bohachevsky':
        
        def fX(X,Y):
            return X**2 + 2*Y**2 - 0.3*np.cos(3*np.pi*X) - 0.4*np.cos(4*np.pi*Y)+0.7

        def fx(X):
            x = X[0][0]
            y = X[1][0]
            return x**2 + 2*y**2 - 0.3*np.cos(3*np.pi*x) - 0.4*np.cos(4*np.pi*y)+0.7

        def dx(X):
            x = X[0][0]
            y = X[1][0]
            return np.array([[2*x+0.9*np.pi*np.sin(3*np.pi*x)],\
                             [4*y+1.6*np.pi*np.sin(4*np.pi*y)]])

        def d2x(X):
            x = X[0][0]
            y = X[1][0]
            ret = np.zeros((2,2))
            ret[0,0] = 2+2.7*np.pi**2*np.cos(3*np.pi*x)
            ret[0,1] = ret[1,0] = 0
            ret[1,1] = 4+6.4*np.pi**2*np.cos(4*np.pi*y)
            return ret

        def d3x(X):
            x = X[0][0]
            y = X[1][0]
            ret = np.zeros((2,2,2))
            ret[0,0,0] = -8.1*np.pi**3*np.sin(3*np.pi*x)
            ret[0,0,1] = ret[0,1,0] = ret[1,0,0] = 0
            ret[1,1,0] = ret[1,0,1] = ret[0,1,1] = 0
            ret[1,1,1] = -25.6*np.pi**3*np.sin(4*np.pi*y)
            return ret;
    
    elif func_name ==  'Goldstein':
        
        def fX(X,Y):
            return (1 + (X+Y+1)**2 *(19 - 4*X + 3*X**2 - 14*Y +6*X*Y +3*Y**2))*(30 + (2*X-3*Y)**2 * (18 - 32*X + 12*X**2 + 48*Y -36*X*Y + 27*Y**2))

        def fx(X):
            x = X[0][0]
            y = X[1][0]
            return (1 + (x + y + 1)**2*(19 - 14*x + 3*x**2 - 14*y + 6*x*y + 3*y**2))*(30 + (2*x - 3*y)**2*(18 - 32*x + 12*x**2 + 48*y - 36*x*y + 27*y**2))

        def dx(X):
            x = X[0][0]
            y = X[1][0]
            return np.array([[24* (8* x**3 - 4* x**2* (9* y + 4) + 6* x* (9* y**2 + 8* y + 1) - 9* y* (3* y**2 + 4* y + 1))* ((3* x**2 + 2* x* (3* y - 7) + 3* y**2 - 14* y + 19)* (x + y + 1)**2 + 1) + 12* (x**3 + x**2* (3* y - 2) + x* (3* y**2 - 4* y - 1) + y**3 - 2* y**2 - y + 2)* ((12* x**2 - 4* x* (9* y + 8) + 3* (9* y**2 + 16* y + 6))* (2* x - 3* y)**2 + 30)],\
                             [12* (x**3 + x**2* (3* y - 2) + x* (3* y**2 - 4* y - 1) + y**3 - 2* y**2 - y + 2)* ((12* x**2 - 4* x* (9* y + 8) + 3* (9* y**2 + 16* y + 6))* (2* x - 3* y)**2 + 30) - 36* (8* x**3 - 4* x**2* (9* y + 4) + 6* x* (9* y**2 + 8* y + 1) - 9* y* (3* y**2 + 4* y + 1))* ((3* x**2 + 2* x* (3* y - 7) + 3* y**2 - 14* y + 19)* (x + y + 1)**2 + 1)]])

        def d2x(X):
            x = X[0][0]
            y = X[1][0]
            ret = np.zeros((2,2))
            ret[0,0] = 12* (672* x**6 - 336* x**5* (3* y + 8) - 20* x**4* (81* y**2 - 168* y - 119) + 40* x**3* (51* y**3 + 108* y**2 - 7* y + 56) + 3* x**2* (435* y**4 - 1360* y**3 - 1790* y**2 - 2560* y - 818) - 2* x* (459* y**5 + 870* y**4 - 310* y**3 - 2460* y**2 - 1446* y + 268) - 3* (81* y**6 - 204* y**5 - 485* y**4 - 280* y**3 - 432* y**2 - 408* y - 70))
            ret[0,1] = ret[1,0] = -12* (168* x**6 + 24* x**5* (27* y - 28) - 10* x**4* (153* y*2 + 216* y - 7) - 20* x**3* (87* y**3 - 204* y**2 - 179* y - 128) + 3* x**2* (765* y**4 + 1160* y**3 - 310* y**2 - 1640* y - 482) + 6* x* (243* y**5 - 510 *y**4 - 970* y**3 - 420* y**2 - 432* y - 204) - 567* y**6 - 972* y**5 + 495* y**4 + 3960* y**3 + 5904* y**2 + 3216* y + 390)
            ret[1,1] = -12* (108* x**6 - 36* x**5* (17* y + 12) + x**4 *(-1305* y**2 + 2040* y + 895) + 20* x**3* (153* y**3 + 174* y**2 - 31* y - 82) + 9* x**2* (405* y**4 - 680* y**3 - 970* y**2 - 280* y - 144) - 6* x* (567* y**5 + 810* y**4 - 330* y**3 - 1980* y**2 - 1968* y - 536) - 6* (567* y**6 - 378* y**5 - 1845* y**4 + 540* y**3 + 2391* y**2 + 1024* y + 85))
            return ret

        def d3x(X):
            x = X[0][0]
            y = X[1][0]
            ret = np.zeros((2,2,2))
            ret[0,0,0] = 24* (2016* x**5 - 840* x**4* (3* y + 8) - 40* x**3* (81* y**2 - 168* y - 119) + 60* x**2* (51* y**3 + 108* y**2 - 7* y + 56) + 3* x* (435* y**4 - 1360* y**3 - 1790* y**2 - 2560* y - 818) - 459* y**5 - 870* y**4 + 310* y**3 + 2460* y**2 + 1446* y - 268)
            ret[0,0,1] = ret[0,1,0] = ret[1,0,0] = -24* (504* x**5 + 60* x**4* (27* y - 28) - 20* x**3* (153* y**2 + 216* y - 7) - 30* x**2* (87* y**3 - 204* y**2 - 179* y - 128) + 3* x* (765* y**4 + 1160* y**3 - 310* y**2 - 1640* y - 482) + 3* (243* y**5 - 510* y**4 - 970* y**3 - 420* y**2 - 432* y - 204))
            ret[1,1,0] = ret[1,0,1] = ret[0,1,1] = -24* (324* x**5 - 90* x**4* (17* y + 12) - 10* x**3* (261* y**2 - 408* y - 179) + 30* x**2* (153* y**3 + 174* y**2 - 31* y - 82) + 9* x* (405* y**4 - 680* y**3 - 970* y**2 - 280* y - 144) - 3* (567* y**5 + 810* y**4 - 330* y**3 - 1980* y**2 - 1968* y - 536))
            ret[1,1,1] = 24* (306* x**5 + 15* x**4* (87* y - 68) - 10* x**3* (459* y**2 + 348* y - 31) - 90* x**2* (81* y**3 - 102* y**2 - 97* y - 14) + 9* x* (945* y**4 + 1080* y**3 - 330* y**2 - 1320* y - 656) + 6* (1701* y**5 - 945* y**4 - 3690* y**3 + 810* y**2 + 2391* y + 512))
            return ret;
        
    elif func_name ==  'Himmelblau':
        
        def fX(X,Y):
            return (X**2 + Y - 11)**2 + (X + Y**2 - 7)**2 + .001

        def fx(X):
            x = X[0][0]
            y = X[1][0]
            return (x**2 + y - 11)**2 + (x + y**2 - 7)**2

        def dx(X):
            x = X[0][0]
            y = X[1][0]
            return np.array([[4* x**3 + 4* x* y - 42* x + 2* y**2 - 14],\
                             [2* x**2 + 4* x* y + 4* y**3 - 26* y - 22]])

        def d2x(X):
            x = X[0][0]
            y = X[1][0]
            ret = np.zeros((2,2))
            ret[0,0] = 12* x**2 + 4* y - 42
            ret[0,1] = ret[1,0] = 4*x+4*y
            ret[1,1] = 4* x + 12* y**2 - 26
            return ret

        def d3x(X):
            x = X[0][0]
            y = X[1][0]
            ret = np.zeros((2,2,2))
            ret[0,0,0] = 24*x
            ret[0,0,1] = ret[0,1,0] = ret[1,0,0] = 4
            ret[1,1,0] = ret[1,0,1] = ret[0,1,1] = 4
            ret[1,1,1] = 24*y
            return ret;
    
    elif func_name ==  'McCormick':
        
        def fX(X,Y):
            return np.sin(X + Y) + (X - Y)**2 - 1.5*X + 2.5*Y + 3

        def fx(X):
            x = X[0][0]
            y = X[1][0]
            return math.sin(x + y) + (x - y)**2 - 1.5*x + 2.5*y + 1

        def dx(X):
            x = X[0][0]
            y = X[1][0]
            return np.array([[math.cos(x + y) + 2*(x - y) - 1.5],\
                             [math.cos(x + y) -2*(x - y) + 2.5]])

        def d2x(X):
            x = X[0][0]
            y = X[1][0]
            ret = np.zeros((2,2))
            ret[0,0] = - math.sin(x + y) + 2
            ret[0,1] = ret[1,0] =  - math.sin(x + y) - 2
            ret[1,1] = - math.sin(x + y) + 2
            return ret

        def d3x(X):
            x = X[0][0]
            y = X[1][0]
            ret = np.zeros((2,2,2))
            ret[0,0,0] = - math.cos(x + y)
            ret[0,0,1] = ret[0,1,0] = ret[1,0,0] = - math.cos(x + y)
            ret[1,1,0] = ret[1,0,1] = ret[0,1,1] = - math.cos(x + y)
            ret[1,1,1] = - math.cos(x + y)
            return ret;
        
    elif func_name ==  'Styblinski':
        
        def fX(X,Y):
            return 0.5*X**4 - 8* X**2 + 2.5*X + 0.5* Y**4 - 8* Y**2 + 2.5* Y + 80

        def fx(X):
            x = X[0][0]
            y = X[1][0]
            return 0.5*x**4 - 8* x**2 + 2.5*x + 0.5* y**4 - 8* y**2 + 2.5* y

        def dx(X):
            x = X[0][0]
            y = X[1][0]
            return np.array([[2*x**3 - 16* x + 2.5],\
                             [2*y**3 - 16* y + 2.5]])

        def d2x(X):
            x = X[0][0]
            y = X[1][0]
            ret = np.zeros((2,2))
            ret[0,0] = 6* x**2 - 16
            ret[0,1] = ret[1,0] = 0
            ret[1,1] = 6*y**2 - 16
            return ret

        def d3x(X):
            x = X[0][0]
            y = X[1][0]
            ret = np.zeros((2,2,2))
            ret[0,0,0] = 12*x
            ret[0,0,1] = ret[0,1,0] = ret[1,0,0] = 0
            ret[1,1,0] = ret[1,0,1] = ret[0,1,1] = 0
            ret[1,1,1] = 12*y
            return ret;
            
    return [fX, fx, dx, d2x, d3x]


def init_params(func_name):
    
    if func_name ==  'Beale':
        XMIN = 2;
        XMAX = 3.5
        YMIN = 0
        YMAX = 1
        x_min = np.array([[3,.5]])
        
    elif func_name ==  'Bohachevsky':    
        XMIN = -1;
        XMAX = 1
        YMIN = -1
        YMAX = 1
        x_min = np.array([[0,.0]])
        
    elif func_name ==  'Goldstein':
        XMIN = -2;
        XMAX = 2
        YMIN = -3
        YMAX = 1
        x_min = np.array([[0,-1]])
        
    elif func_name ==  'Himmelblau':    
        XMIN = -5
        XMAX = 5
        YMIN = -5
        YMAX = 5
        x_min = np.array([[3,2],[-2.805118,3.131312],[-3.779310,-3.283186],[3.584428,-1.848126]])

    elif func_name ==  'McCormick':    
        XMIN = -1.5;
        XMAX = 4
        YMIN = -3
        YMAX = 4
        x_min = np.array([[-0.54719, -1.54719],[2.5944, 1.5944]])
        
    elif func_name ==  'Styblinski':
        XMIN = -6;
        XMAX = 6
        YMIN = -6
        YMAX = 6
        x_min = np.array([[-2.903534, -2.903534],[-2.903534, 2.7468],[2.7468, -2.903534],[2.7468, 2.7468]])
        
            
    return [XMIN, XMAX, YMIN, YMAX, x_min]