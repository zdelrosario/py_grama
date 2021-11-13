all = [
    "eval_pnd"
]

def eval_pnd_1(model, values, sign, seed=None):
    """ Evaluate a Model using a predictive model

    Evaluates a given model against a PND algorithm to determine
    "optimal points".

    Args:
        model(gr.model): predictive model to evaluate
        values(dict): predicted response and uncertainty values
        sign(dict): contains signs for minimization and maximization
        seed(int): declarble seed value for reproducibility

    Returns:
        DataFrame: Results of predictive model going through a PND algorithm.
        Conatians both values and their scores.

    Example:
    >>> import grama as gr
    >>>...
    >>>...
    >>> pred_model >>
            gr.ev_pnd_1(
                values = {
                    'x_sig':sig_values,
                    'x_pred':pred_values,
                    'x_train':train_values
                },
                sign = {
                    'm':1,
                    'n':1
                },
                seed = 101
            )

    """
    pass

def eval_pnd_2(model, df_train, df_test, sign, seed=None):
    """ Evaluate a Model using a predictive model

    Evaluates a given model against a PND algorithm to determine
    "optimal points".

    Args:
        model(gr.model): predictive model to evaluate
        x_pred: predicted response values
        x_sig: predictive uncertainties
        x_train: training response values
        sign(dict): contains signs for minimization and maximization
        seed(int): declarble seed value for reproducibility

    Returns:
        DataFrame: Results of predictive model going through a PND algorithm.
        Conatians both values and their scores.

    Example:
    >>> import grama as gr
    >>>...
    >>>...
    >>> pred_model >>
            gr.ev_pnd_2(
                x_pred = pred_values,
                x_sig = sig_values,
                x_train = train_values,
                sign = {
                    'min':1,
                    'max':1
                },
                seed = 101
            )

    """
    pass

def eval_pnd_3(data, model, values, sign, seed=None):
    """ Evaluate a Model using a predictive model

    Evaluates a given model against a PND algorithm to determine
    "optimal points".

    Args:
        data(DataFrame): DataFrame to help build predictive model
        model(gr.model): model to build prediction from
        values(dict): predicted response and uncertainty values
        sign(dict): contains signs for a minimization or maximization
        seed(int): declarble seed value for reproducibility

    Returns:
        DataFrame: Results of predictive model going through a PND algorithm.
        Conatians both values and their scores.

    Example:
    >>> import grama as gr
    >>>...
    >>>...
    >>> prediction = gr.ev_pnd_2(
            data = df_train,
            model = pred_model
            values = {
                'x_sig':sig_values,
                'x_pred':pred_values,
                'x_train':train_values
            },
            sign = {
                'min':1,
                'max':1
            },
            seed = 101
        )
    """
    pass

def pred_dict_1(x_pred, x_sig, x_train):
    """Helper function to contain values for gr.eval_pnd

    """
    pass

def pred_dict_2(x_pred, x_sig, x_train, sign_min, sign_max):
    pass



### Where values can maybe be made from a helper dictionary function that
### creates a dictionary from inputs of values, allows for predifined keys
### that we can then use to choose the values for the pred, sig, and train data
