import numpy as np
from keras import callbacks

class CustomCheckpoint(callbacks.Callback):
    """
    Custom Keras callback to monitor training and save the best model.

    This callback evaluates the model at the end of each epoch using a custom metric:
    JTS = |train_loss - test_loss| + test_loss.
    The model is saved when the JTS score improves.
    """
    def __init__(self, train_tfds, test_tfds, save_path):
        super().__init__()
        self.best_metric = np.inf
        self.train_tfds = train_tfds
        self.test_tfds = test_tfds
        # Ensure save_path ends with ".keras"
        if not save_path.endswith(".keras"):
            save_path += ".keras"
        self.save_path = save_path

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            return

        # Evaluate the model on train and test datasets
        train_mse = self.model.evaluate(self.train_tfds, verbose=0)
        test_mse = self.model.evaluate(self.test_tfds, verbose=0)

        custom_metric = abs(train_mse - test_mse) + test_mse

        if custom_metric < self.best_metric:
            self.best_metric = custom_metric
            self.model.save(self.save_path)
            print(f"Epoch {epoch + 1}: Model saved with JTS = {custom_metric:.6f}")
