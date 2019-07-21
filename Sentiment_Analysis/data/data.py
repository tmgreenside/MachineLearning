class Data:
    def __init__(self, percent_train=0.5):
        raise NotImplementedError()

    def get_data(self):
        raise NotImplementedError()

    def vectorize_data(self):
        raise NotImplementedError()
