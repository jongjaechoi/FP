import numpy as np
import pandas as pd
import pickle
from scipy.stats import norm
import scipy.integrate as integrate

#############
with open('dfs.pickle','rb') as f:
    dfs = pickle.load(f)

#############
Rpt = 200  #Number of samples (Rpt stands for Repeatition.)
L = 500
I = 3
lb = 0
ub = 1

min_grid = 0.001
max_grid = 0.99
num = 30

grid = np.linspace(min_grid, max_grid, num)
ls = grid.tolist()

alt_h2 = 0.01

############# 
class Output1:   
    def __init__(self):
        pass
 
    def save(self):
        with open('self.pickle', 'wb') as f:
            pickle.dump(self, f)

############# Basic
Kernel = norm # Use Gaussian kernel
    
def K(b,bmax,h):
    return Kernel.pdf( (b-bmax) / h )

def df(x):
    return dfs[x]

def dfY(x):
    return df(x)[df(x).Winner1 == df(x).Winner2]

def dfN(x):
    return df(x)[df(x).Winner1 != df(x).Winner2]

def h1(x):
    dataset = df(x)
    return 1.06 * dataset.Max1.std() * (dataset.shape[0])**(-1/5)

def h2(x):
    dataset = df(x)
    return 1.06 * dataset.Max2.std() * (dataset.shape[0])**(-1/5)

def λℓden(b1,x):
    return sum( K(b1, df(x).Max1, h1(x)) )

def λℓ(b1,b1ℓ,x):
    return K(b1,b1ℓ,h1(x)) /  λℓden(b1,x)

###
def K2bar(b2,b2ℓ,x,crit):
    if crit == "rot":
        return Kernel.cdf((b2-b2ℓ)/h2(x))
    else:
        return Kernel.cdf((b2-b2ℓ)/alt_h2)
    
def M2w(b2,b1,x,crit):
    return np.inner( λℓ(b1,dfY(x).Max1, x), K2bar(b2, dfY(x).Max2, x, crit) )

def m2w(b2,b1,x,crit):
    if crit == "rot":
        return (1/h2(x)) * np.inner( λℓ(b1,dfY(x).Max1,x), 
                                 K(b2,dfY(x).Max2, h2(x)) ) 
    else:
        return (1/alt_h2) * np.inner( λℓ(b1,dfY(x).Max1,x), 
                                 K(b2,dfY(x).Max2, alt_h2) ) 

def M2l(b2,b1,x,crit):
    return np.inner( λℓ(b1,dfN(x).Max1,x), K2bar(b2, dfN(x).Max2, x, crit) )

def m2l(b2,b1,x,crit):
    if crit == "rot":
        return (1/h2(x)) * np.inner( λℓ(b1,dfN(x).Max1, x), 
                                 K(b2,dfN(x).Max2, h2(x)) )
    else:
        return (1/alt_h2) * np.inner( λℓ(b1,dfN(x).Max1, x), 
                                 K(b2,dfN(x).Max2, alt_h2) )

############# G2mx1mx
def G2mx1mx(b2,b1,x,crit):
    return M2w(b2,b1,x,crit) + M2l(b2,b1,x,crit)

def g2mx1mx(b2,b1,x,crit):
    return m2w(b2,b1,x,crit) + m2l(b2,b1,x,crit)

def G2mx1mx_Em(b2,b1,x):
    return sum( λℓ(b1,df(x).Max1, x) * (df(x)['Max2']<=b2) )
    
def G2mx1mx_Tr(b2,b1):
    first = b2**(2*b1) * (1/b1)**(2)  
    second = ( (b2**b1 - 1)/(np.log(b2)) )**2
    return first * second

def G2mx1mx_pds(b1):
    G2mx1mx_Em_array = [ [ G2mx1mx_Em(b2,b1,i) for b2 in ls ] for i in range(Rpt)  ]
    G2mx1mx_Em_pd = pd.DataFrame(G2mx1mx_Em_array)
    
    rotG2mx1mx_array = [ [ G2mx1mx(b2,b1,i,"rot") for b2 in ls ] for i in range(Rpt)  ]
    rotG2mx1mx_pd = pd.DataFrame(rotG2mx1mx_array)  

    altG2mx1mx_array = [ [ G2mx1mx(b2,b1,i,"alt") for b2 in ls ] for i in range(Rpt)  ]
    altG2mx1mx_pd = pd.DataFrame(altG2mx1mx_array)  

    return G2mx1mx_Em_pd, rotG2mx1mx_pd, altG2mx1mx_pd

############# m2w
def m2w_Tr(b2,b1):
    first = 2*b1*b2**(2*b1-1) * (1/b1)**(2)  
    second = ( (b2**b1 - 1)/(np.log(b2)) )**2
    return first * second

def m2w_pds(b1):
    rotm2w_array = [ [ m2w(b2,b1,i,"rot") for b2 in ls ] for i in range(Rpt)  ]
    rotm2w_pd = pd.DataFrame(rotm2w_array) 

    altm2w_array = [ [ m2w(b2,b1,i,"alt") for b2 in ls ] for i in range(Rpt)  ]
    altm2w_pd = pd.DataFrame(altm2w_array)                             
    return rotm2w_pd, altm2w_pd

############# G21w
def G21w_Em_first(b1,x):
    numerat = λℓ(b1,dfY(x).Max1,x)
    ind = df(x).Max2.copy()
    ind_numpy = ind.to_numpy(copy=True)
    ind_numpy.shape = (L,1)
    num = len(dfY(x))
    ind_matrix = np.tile(ind_numpy,num)

    comparison = dfY(x).Max2.copy()
    comparison_numpy = comparison.to_numpy(copy=True)

    boo = ind_matrix<=comparison_numpy

    denomet_pre = np.array([λℓ(b1,df(x).Max1,x)]).T * boo
    denomet = np.sum(denomet_pre, axis=0)
    # Do the following code. if all true, then you are fine
    # test = df(100).Max2<=dfY(100).Max2[0]
    # dd = test.to_numpy()
    # boo[:,0] == dd
    return np.sum( numerat / denomet )

def G21w_Em_second(b2,b1,x):
    numerat = λℓ(b1,dfY(x).Max1,x)
    ind = df(x).Max2.copy()
    ind_numpy = ind.to_numpy(copy=True)
    ind_numpy.shape = (L,1)
    num = len(dfY(x))
    ind_matrix = np.tile(ind_numpy,num)

    comparison = dfY(x).Max2.copy()
    comparison_numpy = comparison.to_numpy(copy=True)

    boo = ind_matrix<=comparison_numpy

    denomet_pre = np.array([λℓ(b1,df(x).Max1,x)]).T * boo
    denomet = np.sum(denomet_pre, axis=0)
    # Do the following code. if all true, then you are fine
    # test = df(100).Max2<=dfY(100).Max2[0]
    # dd = test.to_numpy()
    # boo[:,0] == dd
    return np.sum( (numerat / denomet) * (dfY(x).Max2<=b2) )

def G21w_Em(b2,b1,x):
    return np.exp( -G21w_Em_first(b1,x) ) * np.exp( G21w_Em_second(b2,b1,x) )
### Up to now, empirical.

def G2lwf(b,b1,x,crit):
    if crit == "rot":
        numerator = np.inner( λℓ(b1,dfY(x).Max1,x), 
                                 K(b,dfY(x).Max2,h2(x)) ) 
        denominator = G2mx1mx(b,b1,x,crit)
        return numerator / denominator
    else:
        numerator = np.inner( λℓ(b1,dfY(x).Max1,x), 
                                 K(b,dfY(x).Max2,alt_h2) ) 
        denominator = G2mx1mx(b,b1,x,crit)
        return numerator / denominator       

def G21w(b2,b1,x,crit):    
    value2, err = integrate.quad(lambda y: G2lwf(y,b1,x,crit), b2, ub)
    if crit == "rot":
        return np.exp( -(1/h2(x)) * value2 )
    else:
        return np.exp( -(1/alt_h2) * value2 )
### Up to now, smooth.

def G21w_Tr(b2,b1):    
    return b2**(2*b1)

def G21w_pds(b1):
    G21w_Em_array = [ [ G21w_Em(b2,b1,i) for b2 in ls ] for i in range(Rpt)  ]
    G21w_Em_pd = pd.DataFrame(G21w_Em_array)
    
    rotG21w_array = [ [ G21w(b2,b1,i,"rot") for b2 in ls ] for i in range(Rpt)  ]
    rotG21w_pd = pd.DataFrame(rotG21w_array)                            

    altG21w_array = [ [ G21w(b2,b1,i,"alt") for b2 in ls ] for i in range(Rpt)  ]
    altG21w_pd = pd.DataFrame(altG21w_array)

    return G21w_Em_pd, rotG21w_pd, altG21w_pd


############# G2l
def G2lf(b,b1,x,crit):
    if crit == "rot":
        numerator = np.inner( λℓ(b1,dfN(x).Max1,x), 
                                 K(b,dfN(x).Max2,h2(x)) ) 
        denominator = G2mx1mx(b,b1,x,crit)
        return numerator / denominator
    else:
        numerator = np.inner( λℓ(b1,dfN(x).Max1,x), 
                                 K(b,dfN(x).Max2,alt_h2) ) 
        denominator = G2mx1mx(b,b1,x,crit)
        return numerator / denominator

def G2l(b2,b1,x,crit):  
    # similar to G21w
    if crit == "rot":
        return np.exp( -1/(h2(x)*(I-1)) *  integrate.quad(lambda y: G2lf(y,b1,x), b2, ub)[0] )
    else:
        return np.exp( -1/(alt_h2*(I-1)) *  integrate.quad(lambda y: G2lf(y,b1,x), b2, ub)[0] )
### Up to now, smooth.

def G2l_Tr(b2,b1):
    first = 1/b1
    second = (b2**b1 - 1)/(np.log(b2))
    return first * second
### Up to now, truth.

def G2l_Em(b2,b1,x):
    # Similar to G21w_Em_second(b2,b1,x) with some minor difference 
    numerat = λℓ(b1,dfN(x).Max1,x)
        # numerator finished. Now, denomenator. 
    ind = df(x).Max2.copy()
    ind_numpy = ind.to_numpy(copy=True)
    ind_numpy.shape = (L,1)
    num = len(dfN(x))
    ind_matrix = np.tile(ind_numpy,num)
    
    comparison = dfN(x).Max2.copy()
    comparison_numpy = comparison.to_numpy(copy=True)

    boo = ind_matrix<=comparison_numpy

    denomet_pre = np.array([λℓ(b1,df(x).Max1,x)]).T * boo
    denomet = np.sum(denomet_pre, axis=0)
        # Do the following code. if all true, then you are fine
        # test = df(100).Max2<=dfY(100).Max2[0]
        # dd = test.to_numpy()
        # boo[:,0] == dd
    inside = np.sum( (numerat / denomet) * (dfN(x).Max2 > b2) )
    return np.exp( -(1/(I-1)) * inside )
### Up to now, Empirical estimator.

def G2l_pds(b1):
    G2l_Em_array = [ [ G2l_Em(b2,b1,i) for b2 in ls ] for i in range(Rpt)  ]
    G2l_Em_pd = pd.DataFrame(G2l_Em_array)
    
    rotG2l_array = [ [ G2l(b2,b1,i,"rot") for b2 in ls ] for i in range(Rpt)  ]
    rotG2l_pd = pd.DataFrame(rotG2l_array)                            

    altG2l_array = [ [ G2l(b2,b1,i,"alt") for b2 in ls ] for i in range(Rpt)  ]
    altG2l_pd = pd.DataFrame(altG2l_array)

    return G2l_Em_pd, rotG2l_pd, altG2l_pd

############# g21w, g2l, m2l
def g21w(b2,b1,x,crit):
    return ( m2w(b2,b1,x,crit) / G2mx1mx(b2,b1,x,crit) ) * G21w(b2,b1,x,crit)

def g2l(b2,b1,x,crit):
    return 1/(I-1) * ( m2l(b2,b1,x,crit) / G2mx1mx(b2,b1,x,crit) ) * G2l(b2,b1,x,crit)

def m2l_pds(b1):
    rotm2l_array = [ [ m2l(b2,b1,i,"rot") for b2 in ls ] for i in range(Rpt)  ]
    rotm2l_pd = pd.DataFrame(rotm2l_array) 

    altm2l_array = [ [ m2l(b2,b1,i,"alt") for b2 in ls ] for i in range(Rpt)  ]
    altm2l_pd = pd.DataFrame(altm2l_array)                             
    return rotm2l_pd, altm2l_pd

############# G1
def K1bar(b1,b1ℓ,x):
    return Kernel.cdf((b1-b1ℓ)/h1(x))

def G1(b1,x):
    return ( 1/L * sum(K1bar(b1,df(x).Max1,x)) )**(1/I)

def G1_Imin1(b1,x):
    return G1(b1,x)**(I-1)

mlt_G1 = (1/L)**((I-1)/I)

def G1_lmin1dr(b1,x):
    return mlt_G1 * (I-1)/(I) * 1/h1(x) * (sum(K1bar(b1,df(x).Max1,x)))**(-1/I) * sum( K(b1, df(x).Max1, h1(x)) )
### Up to now, Smooth.

def G1_Em(b1,x):
    return ( (1/L) * np.sum( df(x).Max1<=b1 ) )**(1/I)

def G1_Tr(b1):
    return b1

def G1_pds():
    G1_Em_array = [ [G1_Em(b1,i) for b1 in ls ] for i in range(Rpt)  ]
    G1_Em_pd = pd.DataFrame(G1_Em_array)
    
    G1_array = [ [ G1(b1,i) for b1 in ls ] for i in range(Rpt)  ]
    G1_pd = pd.DataFrame(G1_array)                            
    return G1_Em_pd, G1_pd

############# H2w, H2l
def H2w(b2,b1,x,crit):
    return G2l(b2,b1,x,crit)**(I-1)

def H2w_Tr(b2,b1):
    return G2l_Tr(b2,b1)**(I-1)
### Up to now, H2w

def H2l_Tr(b2,b1):
    first = 2/(1-b1**2)
    second = 1/np.log(b2)
    integ = integrate.quad(lambda x: -b2**(2*x) + b2**(3*x), b1, ub )[0]
    return first * second * integ

def H2l1(b,x,sx,crit):
    if crit=="rot":        
        first = -(I-2)/(h2(sx)*(I-1)) * np.inner( λℓ(x,dfN(sx).Max1,sx), K(b,dfN(sx).Max2,h2(sx)) ) 
        second = (-1/h2(sx)) * np.inner( λℓ(x,dfY(sx).Max1,sx), K(b,dfY(sx).Max2,h2(sx)) ) 
        denominator = G2mx1mx(b,x,sx,crit)
        return (first+second)/denominator
    else:
        first = -(I-2)/(alt_h2*(I-1)) * np.inner( λℓ(x,dfN(sx).Max1,sx), K(b,dfN(sx).Max2,alt_h2) ) 
        second = (-1/alt_h2) * np.inner( λℓ(x,dfY(sx).Max1,sx), K(b,dfY(sx).Max2,alt_h2) ) 
        denominator = G2mx1mx(b,x,sx,crit)
        return (first+second)/denominator

def H21l_int(b2,x,sx,crit):
    return np.exp(  integrate.quad(lambda b: H2l1(b,x,sx,crit), b2, ub)[0] )

def H2l(b2,b1,sx,crit):
    return 1/( 1 - G1(b1,sx)**(I-1) ) * integrate.quad(lambda x: H21l_int(b2,x,sx,crit) * G1_lmin1dr(x,sx), b1, ub )[0]


############# Useless codes, for now.
#def m2l_Tr(a,c):
#    first_1 = a**(c-1)/np.log(a)
#    first_2 = (a**(c)-1)/(a*c*np.log(a)*np.log(a))
#    second = (1/c)*( (a**(c)-1)/np.log(a) )
#    third = a**(2*c)
#    return 2 * (first_1 - first_2) * second * third
