import numpy as np
import torch
import functools as ftools

from logzero import logger

from torch.fft import fftshift, fftn, ifftn

try:
    from pyfftw.interfaces.numpy_fft import fftshift as fftshift_c
except: 
    from numpy.fft import fftshift as fftshift_c

def fftn_n( arr ):
    return fftn( arr, norm='ortho' )

def ifftn_n( arr ):
    return ifftn( arr, norm='ortho' )

chirp = np.mgrid[ 0:1, 0:1, 0:1 ]
chirp_arg = 1.j * np.pi * ftools.reduce( lambda x, y: x+y, chirp )
chip_arg = torch.from_numpy( chirp_arg ).cuda() # puts the array on the GPU

pref0 = 'chirp = tuple( fftshift_c( this )**2 / this.shape[n] for n, this in enumerate( np.mgrid[ '
suff0 = ' ] ) )'

DoNothing = lambda x: x
opdict = { 0:DoNothing, 1:fftn_n, 2:np.flip, 3:ifftn_n }

def frft( arr, alpha ):
    """
    NOTE: input array should be a fftshifted Pytorch tensor. 
    Can be created from an fftshifted numpy array ar_n by: 
   
    arr = torch.from_numpy( ar_n ).cuda()

    The last cuda() routine copies tha array to the GPU. 
    """
    if arr.shape != chirp[0].shape:
        RecalculateChirp( arr.shape )
    ops = CanonicalOps( alpha )
    return frft_base( ops[0]( arr ), ops[1] )

def frft_base( arr, alpha ):
    phi = alpha * np.pi/2. 
    cotphi = 1. / np.tan( phi )
    cscphi = np.sqrt( 1. + cotphi**2 )
    scale = np.sqrt( 1. - 1.j*cotphi ) / np.sqrt( np.prod( arr.shape ) )
    modulator = ChirpFunction( cotphi - cscphi )
    filtor = ChirpFunction( cscphi )
    arr_frft = scale * modulator * ifftn_n( fftn_n( filtor ) * fftn_n( modulator * arr ) )
    return arr_frft

def ChirpFunction( x ):
    return torch.exp( x * chirp_arg ) 

def RecalculateChirp( newshape ):
    logger.warning( 'Recalculating chirp. ' )
    global chirp_arg
    if len( newshape ) == 1:    # extra-annoying string manipulations needed with 1D data
        pref = pref0.replace( 'np.', '( np.' )
        suff = suff0.replace( ']', '], )' )
    else: 
        pref = pref0
        suff = suff0
    regrid = ','.join( tuple( '-%d:%d'%(n//2,n//2) for n in newshape ) ).join( [ pref, suff ] )
    #print( regrid )
    exec( regrid, globals() )
    chirp_arg = 1.j * np.pi * ftools.reduce( lambda x, y: x+y, chirp )
    chirp_arg = torch.from_numpy( chirp_arg ).cuda()
    return

def CanonicalOps( alpha ):
    alpha_0 = alpha % 4. 
    if alpha_0 < 0.5:
        return[ ifftn_n, 1.+alpha_0 ]
    flag = 0
    while alpha_0 > 1.5:
        alpha_0 -= 1.
        flag += 1
    return [ opdict[flag], alpha_0 ]



