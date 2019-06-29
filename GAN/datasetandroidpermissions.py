"""
Author: Robert Podschwadt
data from https://www.kaggle.com/xwolf12/datasetandroidpermissions
"""
import os
import pickle
import numpy as np


possible_paths = [
'/home/robert/workspace',
'/home/poji/adversarial_workspace',
'/Users/norabest/Desktop/Texas REU/malgan-master'
]

HOME = None
for path in possible_paths:
    if os.path.exists( os.path.join( path, 'data', 'datasetandroidpermissions' ) ):
        HOME = os.path.join( path, 'data', 'datasetandroidpermissions' )
        break
if HOME is None:
    print( 'could not find dataset' )
    exit()


RAW_FILE = os.path.join( HOME, 'train.csv' )

OUT_FILE = os.path.join( HOME, 'train.data' ) # change this to match your systems

# percentage of data we want as training set
trainP = 0.8


def load_raw():
    f = open( RAW_FILE, 'r' )
    f.readline() # skip firstline
    x = []
    y = []
    for line in f.readlines():
        features = line.rstrip( '\n' ).split( ';' )
        arr = []
        for i in range( len( features ) - 1 ):
            arr.append( int( features[ i ] ) )
        x.append( arr[:] )
        y.append( [ int( features[ -1 ] ) ] )
    f.close()
    x = np.array( x )
    y = np.array( y )

    return x,y


def load_data( force_proccess=False ):

    if os.path.exists( OUT_FILE ) and not force_proccess:
        x,y = pickle.load( open( OUT_FILE, 'rb' ) )
    else:
        x,y = load_raw()
        pickle.dump( (x,y), open( OUT_FILE, 'wb' ) )
    return x,y


def training_and_test( x, y ):
    np.random.seed( 7 )
    p = np.random.permutation( len( x ) )
    print( x.shape )
    print( y.shape )
    x_train = x[ p ][ : int( len( x ) * trainP ) ]
    y_train = y[ p ][ : int( len( x ) * trainP ) ]
    x_test = x[ p ][ int( len( x ) * trainP ) : ]
    y_test = y[ p ][ int( len( x ) * trainP ) : ]
    print( x_train.shape )
    print( y_train.shape )
    print( x_test.shape )
    print( y_test.shape )
    return ( x_train, y_train ), ( x_test, y_test )

def load_data_seperatly():
    """
    Return two datasets. one for benign and one for malicious
    """
    x,y = load_data()

    i = np.argwhere( y == 1 )

    x_mal = x[ i[ :,0 ] ][:]
    y_mal = y[ i[ :,0 ] ][:]

    i = np.argwhere( y == 0 )
    x_beg = x[ i[ :,0 ] ][:]
    y_beg = y[ i[ :,0 ] ][:]

    return (x_mal, y_mal),(x_beg, y_beg)


if __name__ == '__main__':
    x,y = load_data_seperatly(  )
    print( x )
    print( y )
    print( x[ 0 ].shape )
    print( y[ 0 ].shape )
