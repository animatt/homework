def _nextMean(next_x, prev_mean, num_terms):
    '''
    Recursive mean: x_n = x_n-1 + 1 / n * (X_n - x_n-1)
    '''
    return prev_mean + (next_x - prev_mean) / num_terms


def _nextVar(next_x, next_mean, prev_var, prev_mean, num_terms):
    '''
    Recursive var: 
    s_n^2 = ((n - 1) * s_n^2 + (X_n - x_n+1) ** 2) / n + (x_n + x_n+1) ** 2
    '''
    next_var = prev_var + (next_mean - prev_mean) ** 2
    next_var += ((next_x - next_mean) ** 2 - prev_var) / (num_terms - 1)

    return next_var


def get_mean_var(next_x, prev_mean, prev_var, num_terms):
    '''
    returns -> (next_mean, next_var)
    '''
    next_mean = _nextMean(next_x, prev_mean, num_terms)
    next_var = _nextVar(next_x, next_mean, prev_var, prev_mean, num_terms)

    return next_mean, next_var
