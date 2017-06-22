'''
Created on Oct 14, 2010
@author: Stephen O'Hara
Performs tensor unfolding that is same as done by LUI in his Product Manifold MATLAB code,
which in turn uses Dahua Lin's sltensor_unfold code from Dec 17, 2005.

This code is limited to 3rd order tensors (cubes).
'''
from scipy import *

def UnfoldCube(X, k=1):
    ''' Unfold a 3rd order tensor along mode 1, 2 or 3.
    @param X: data cube to be unfolded, 3rd order tensor, (depth, row, col) in python 
    @param k: which order to unfold on. For compatibility with MATLAB 1=row, 2=col, 3=depth
    @return: 2D array of appropriate dimension
        - k=1 will be of size row x (col*depth)
        - k=2 will be of size col x (row*depth)
        - k=3 will be of size depth x (row*col)
    @raise IndexError: if k is not 1,2,or 3 
    '''
    #depth, row, col = X.shape
    if k==1:
        perm = (0,2,1)
    elif k==2:
        perm = (0,1,2)
    elif k==3:
        perm = (2,0,1)
    else:
        raise IndexError
        print "Error: Unfold cube k must be 1,2, or 3."
        
    Y = X.transpose(perm)
    Y = hstack(Y[:,])
    return(Y)
    
def gen_test_tensor():
    A = ones( (2,4,3)) #create a depth=2, row=4, col=3 tensor
    count = 1
    for k in range(2):
        for i in range(4):
            for j in range(3):
                A[k,i,j] = count
                count += 1
    return A
     
def test():
    A = gen_test_tensor();    
    print 'Testing Tensor has 2 slices of a 4x3 matrix'
    print A
    print 'Unfold mode 1 = 4x(3x2) = 4x6 matrix'
    print UnfoldCube(A,1)
    print 'Unfold mode 2 = 3x(4x2) = 3x8 matrix'
    print UnfoldCube(A,2)
    print 'Unfold mode 3 = 2x(4x3) = 2x12 matrix'
    print UnfoldCube(A,3)


if __name__ == '__main__':
    pass

