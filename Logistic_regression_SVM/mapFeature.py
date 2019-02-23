
def mapFeature(x1, x2):
    import numpy as np
    '''
    Maps the two input features to quadratic features.
        
    Returns a new feature array with d features, comprising of
        X1, X2, X1 ** 2, X2 ** 2, X1*X2, X1*X2 ** 2, ... up to the 6th power polynomial
        
    Arguments:
        X1 is an n-by-1 column matrix
        X2 is an n-by-1 column matrix
    Returns:
        an n-by-d matrix, where each row represents the new features of the corresponding instance
    '''

    X = []

    Xlen = x1.shape[0]

    for i in range(Xlen):
        X_feat =            [x1[i],
                            x2[i],
                            x1[i]**2,
                            x1[i]*x2[i],
                            x2[i]**2,
                            x1[i]**3,
                            x1[i]**2 *x2[i],
                            x1[i] *x2[i]**2,
                            x2[i]**3,
                            x1[i]**4,
                            x1[i]**3 *x2[i],
                            x1[i]**2 *x2[i]**2,
                            x1[i] *x2[i]**3,
                            x2[i]**4,
                            x1[i]**5,
                            x1[i]**4 *x2[i],
                            x1[i]**3 *x2[i]**2,
                            x1[i]**2 *x2[i]**3,
                            x1[i] *x2[i]**4,
                            x2[i]**5,
                            x1[i]**6,
                            x1[i]**5 *x2[i],
                            x1[i]**4 *x2[i]**2,
                            x1[i]**3 *x2[i]**3,
                            x1[i]**2 *x2[i]**4,
                            x1[i] *x2[i]**5,
                            x2[i]**6]
        X.append(X_feat)

    X = np.array(X)
    print(X)
    return X
 

