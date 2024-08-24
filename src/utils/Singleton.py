class Singleton(type):
    _instance = {}
    """
    class : Singleton
    """
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instance:
            cls._instance[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instance[cls]
