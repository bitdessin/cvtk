try:
    import label_studio_sdk
    from .deploy import *
except ImportError:
    import warnings
    warnings.warn(
        "The environment does not have label_studio_sdk module installed. "
        "Functions in the ml module will not be available. "
        "To ensure full functionality, please install the required dependencies "
        "following the instructions at https://labelstud.io/guide/sdk. ",
        ImportWarning
    )
    