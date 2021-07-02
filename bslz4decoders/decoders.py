

import struct
import bitshuffle
import numpy as np

from bslz4decoders.ccodes.decoders import read_starts, onecore_lz4
from bslz4decoders.ccodes.ompdecoders import omp_lz4, omp_lz4_blocks


"""
We are aiming to duplicate this interface from bitshuffle :

>>> help(bitshuffle.decompress_lz4)
Help on built-in function decompress_lz4 in module bitshuffle.ext:

decompress_lz4(...)
    Decompress a buffer using LZ4 then bitunshuffle it yielding an array.

    Parameters
    ----------
    arr : numpy array
        Input data to be decompressed.
    shape : tuple of integers
        Shape of the output (decompressed array). Must match the shape of the
        original data array before compression.
    dtype : numpy dtype
        Datatype of the output array. Must match the data type of the original
        data array before compression.
    block_size : positive integer
        Block size in number of elements. Must match value used for
        compression.

    Returns
    -------
    out : numpy array with shape *shape* and data type *dtype*
        Decompressed data.
"""

class BSLZ4ChunkConfig:
    """ Wrapper over a binary blob that comes from a hdf5 file """

    __slots__ = [ "shape", "dtype", "blocksize", "output_nbytes" ]

    def __init__(self, shape, dtype, blocksize=8192, output_nbytes=0 ):
        self.shape = shape
        self.dtype = dtype
        self.blocksize = blocksize
        if output_nbytes:
            self.output_nbytes = output_nbytes
        else:
            self.output_nbytes = shape[0]*shape[1]*dtype.itemsize

    def get_blocks( self, chunk, blocks=None ):
        """
        allow blocks to be pre-allocated (e.g. pinned memory)
        sets self.blocksize only if blocks is None
        """
        if blocks is None:
            # We do this in python as it doesn't seem worth making a call back
            # ... otherwise need to learn to call free on a numpy array
            total_bytes, self.blocksize = struct.unpack_from("!QL", chunk, 0)
            if self.blocksize == 0:
                self.blocksize = 8192
            nblocks =  (total_bytes + self.blocksize - 1) // self.blocksize
            assert self.output_nbytes == total_bytes, "chunk config mismatch:"+repr(self)
            blocks = np.empty( nblocks, np.uint32 )
        read_starts( chunk, self.dtype.itemsize, self.blocksize, blocks )
        return blocks
    
    def last_blocksize( self ):
        last = self.output_nbytes % self.blocksize
        tocopy = last % ( self.dtype.itemsize * 8 )
        last -= tocopy
        return last
    
    def tocopy( self ):
        return self.output_nbytes % ( self.dtype.itemsize * 8 )

    def __repr__(self):
        return "%s %s %d %d"%( repr(self.shape), repr(self.dtype),
                            self.blocksize, self.output_nbytes)

def decompress_bitshuffle( chunk, config, output = None ):
    """  Generic bitshuffle decoder depending on the
    bitshuffle library from https://github.com/kiyo-masui/bitshuffle

    input: chunk compressed data
           config, gives the shape and dtype
    returns: decompressed data
    """
    r = bitshuffle.decompress_lz4( chunk[12:],
                                   config.shape,
                                   np.dtype(config.dtype),
                                   config.blocksize // config.dtype.itemsize )
    if output is not None:
        output[:] = r
    else:
        output = r
    return output



# FIXME : make this a decorator and wrap ipp libs
def decompress_onecore( chunk, config, output = None ):
    """  One core decoding from our ccodes
    """
    if output is None:
        output = np.empty( config.shape, config.dtype )
    err = onecore_lz4( np.asarray(chunk) ,
                    config.dtype.itemsize, output.view( np.uint8 ) )
    if err:
        raise Exception("Decoding error")
    # TODO: put the bitshuffle into C !
    return bitshuffle.bitunshuffle( output.view(config.dtype) ).reshape( config.shape )


def decompress_omp( chunk, config, output = None ):
    """  Openmp decoding from our ccodes module
    """
    if output is None:
        output = np.empty( config.shape, config.dtype )
    err = omp_lz4( np.asarray(chunk) , config.dtype.itemsize, output.view( np.uint8 ) )
    if err:
        raise Exception("Decoding error")
    # TODO: put the bitshuffle into C !
    return bitshuffle.bitunshuffle( output.view(config.dtype) ).reshape( config.shape )


def decompress_omp_blocks( chunk, config,
                           offsets=None,
                           output = None ):
    """  Openmp decoding from our ccodes module
    (In the long run - we are expecting the offsets to be cached sonewhere)
    """
    achunk = np.asarray( chunk )
    if output is None:
        output = np.empty( config.shape, config.dtype )
    if offsets is None:
        offsets = config.get_blocks( achunk )
    err = omp_lz4_blocks( achunk , config.dtype.itemsize,
                          config.blocksize, offsets, output.view( np.uint8 ) )
    if err:
        raise Exception("Decoding error")
    # TODO: put the bitshuffle into C !
    return bitshuffle.bitunshuffle( output.view(config.dtype) ).reshape( config.shape )
