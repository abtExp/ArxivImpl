from tensorflow.keras.utils import plot_model

class MODEL():
    def __init__(self, config, model_name):
        self.config = config
        self.model_name = model_name
        self.config.CONFIG_INFO.MODEL_IMAGE_PATH += self.model_name+'.png'
        self.model = self.compose_model()

    def train(self):
        self.init_loaders()
        self.model.fit_generator(
                    self.train_loader,
                    validation_data=self.valid_loader,
                    epochs=self.config.TRAIN_EPOCHS,
                    callbacks=self.config.get_callbacks(self.model_name)
                )

    def save(self, path):
        self.model.save(path)

    def load_weights(self, pth):
        if pth:
            self.config.BEST_WEIGHT_PATH = pth
        self.model.load_weights(self.config.CONFIG_INFO.CHECKPOINTS_PATH)

    def init_loaders(self):
        self.train_loader = self.DATA_LOADER(self.config, 'train')
        self.valid_loader = self.DATA_LOADER(self.config, 'valid')

    def summary(self):
        self.model.summary()

    def plot(self):
        plot_model(self.model, self.config.CONFIG_INFO.MODEL_IMAGE_PATH, show_shapes=True)

    def predict(self, data):
        return self.model.predict(data)[0]

    def predict_on_batch(self):
        batch = self.config.TEST_LOADER(self.config.TEST_BATCH_SIZE, 'test')
        return self.model.predict_on_batch(batch)

    def compose_model(self):
        return

    def process(self, path, *args):
        return