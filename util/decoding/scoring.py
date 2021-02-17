import numpy as np

def cross_val_predict_proba(estimator, X, y, groups = None, cv = None, 
    n_jobs = 1, verbose = 0, fit_params = None, pre_dispatch = '2*n_jobs'):
    '''
    Gets class probability predictions for test examples 
    over cross validations runs.

    Adapted from mne.decoding.base.cross_val_multiscore(). See that func's
    documentation for details on inputs.
    '''
    import time
    import numbers
    from mne.parallel import parallel_func
    from mne.fixes import is_classifier
    from sklearn.base import clone
    from sklearn.utils import indexable
    from sklearn.model_selection._split import check_cv

    # check arguments
    X, y, groups = indexable(X, y, groups)
    cv = check_cv(cv, y, classifier = is_classifier(estimator))
    cv_iter = list(cv.split(X, y, groups))

    # We clone the estimator to make sure that all the folds are
    # independent, and that it is pickle-able.
    # Note: this parallelization is implemented using MNE Parallel
    parallel, p_func, n_jobs = parallel_func(_predict_proba, n_jobs,
                                             pre_dispatch = pre_dispatch)
    preds = parallel(p_func(clone(estimator), X, y, train, test,
                             0, None, fit_params)
                      for train, test in cv_iter)

    # flatten over parallel output
    y_hat = np.concatenate([p[0] for p in preds], axis = 0)
    is_y_true = True
    try:
        y_true = np.concatenate([p[1] for p in preds], axis = 0)
    except: # learner was unsupervised
        is_y_true = False

    # return results
    if is_y_true:
        return y_hat, y_true
    else:
        return y_hat


def _predict_proba(estimator, X, y, train, test, 
    verbose, parameters, fit_params):
    '''
    Fits an estimator to the training set and outputs probability predictions 
    (and true labels, if applicable) for test set.

    Adapted from mne.decoding.base._fit_and_score()
    '''

    from mne.fixes import _check_fit_params
    from sklearn.utils.metaestimators import _safe_split
    
    if verbose > 1:
        if parameters is None:
            msg = ''
        else:
            msg = '%s' % (', '.join('%s=%s' % (k, v)
                          for k, v in parameters.items()))
        print("[CV] %s %s" % (msg, (64 - len(msg)) * '.'))

    # Adjust length of sample weights
    fit_params = fit_params if fit_params is not None else {}
    fit_params = _check_fit_params(X, fit_params, train)

    if parameters is not None:
        estimator.set_params(**parameters)

    X_train, y_train = _safe_split(estimator, X, y, train)
    X_test, y_test = _safe_split(estimator, X, y, test, train)
    
    if y_train is None:
        estimator.fit(X_train, **fit_params)
    else:
        estimator.fit(X_train, y_train, **fit_params)
    
    y_hat = estimator.predict_proba(X_test)
    
    if y_train is None:
        return y_hat
    else:
        return y_hat, y_test

def auc_across_time(y_prob, y):
    '''
    Loops over time, computing AUC at every time point.

    I'm less confident this will work in all cases, since it may depend 
    on idiosyncracies of sklearn's auto class labelling. But in this project,
    where we explicitly assign class labels before pushing data through
    sklearn, it works fine. 
    '''

    from sklearn.metrics import roc_auc_score

    scores = []
    for time in range(y_prob.shape[1]): 
        y_hat = y_prob[:, time, 1]
        scores.append(roc_auc_score(y, y_hat))

    return np.array(scores)




