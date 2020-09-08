#import libscores
#scoring_function = getattr(libscores, score_name)

import numpy as np
score_name = 'mse'

# this looks for the scoring function defined in libscores such as
def scoring_function(solution, prediction):
    ''' Mean Square Error'''
    score = np.mean((solution-prediction)**2)    
    return np.mean(score)

