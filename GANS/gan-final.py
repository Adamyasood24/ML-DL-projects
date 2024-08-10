import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Dropout, Reshape, UpSampling2D, ReLU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
import matplotlib.pyplot as plt

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
# Load and preprocess dataset
(ds, info) = tfds.load('stanford_dogs', split='train', with_info=True)

def scale_image(image_dict):
    image = image_dict['image']
    image = tf.image.resize(image, (128, 128))  # Ensure image size is 128x128
    return image / 255.0

AUTOTUNE = tf.data.experimental.AUTOTUNE
ds = ds.map(scale_image, num_parallel_calls=AUTOTUNE)
ds = ds.cache()
ds = ds.shuffle(8000)
ds = ds.batch(batch_size=32)  # Reduced batch size to 32
ds = ds.prefetch(buffer_size=AUTOTUNE)

# Define generator model with ReLU activation
def build_gen():
    model = Sequential()
    model.add(Dense(8*8*512, input_dim=128))
    model.add(ReLU())
    model.add(Reshape((8, 8, 512)))
    model.add(UpSampling2D())
    model.add(Conv2D(256, kernel_size=5, padding="same"))
    model.add(ReLU())
    model.add(UpSampling2D())
    model.add(Conv2D(128, kernel_size=5, padding="same"))
    model.add(ReLU())
    model.add(UpSampling2D())
    model.add(Conv2D(64, kernel_size=4, padding="same"))
    model.add(ReLU())
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(32, kernel_size=4, padding="same"))
    model.add(ReLU())
    model.add(Conv2D(3, kernel_size=4, padding="same", activation="sigmoid"))
    return model

# Define discriminator model with ReLU activation
def build_disc():
    model = Sequential()
    model.add(Conv2D(64, kernel_size=5, input_shape=(128, 128, 3), padding='same'))
    model.add(ReLU())
    model.add(Dropout(0.4))
    model.add(Conv2D(128, kernel_size=5, padding='same'))
    model.add(ReLU())
    model.add(Dropout(0.4))
    model.add(Conv2D(256, kernel_size=5, padding='same'))
    model.add(ReLU())
    model.add(Dropout(0.4))
    model.add(Conv2D(512, kernel_size=5, padding='same'))
    model.add(ReLU())
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(1, activation="sigmoid"))
    return model

generator = build_gen()
discriminator = build_disc()

# Define GAN model
class GAN(Model):
    def __init__(self, generator, discriminator, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.generator = generator
        self.discriminator = discriminator

    def compile(self, g_opt, d_opt, g_loss, d_loss, *args, **kwargs):
        super().compile(*args, **kwargs)
        self.g_opt = g_opt
        self.d_opt = d_opt
        self.g_loss = g_loss
        self.d_loss = d_loss

    def train_step(self, batch_data):
        real_images = tf.convert_to_tensor(batch_data, dtype=tf.float32)
        batch_size = tf.shape(real_images)[0]
        random_latent_vectors = tf.random.normal(shape=(batch_size, 128))

        fake_images = self.generator(random_latent_vectors, training=True)

        # Add noise to the real and fake images
        noisy_real_images = real_images + 0.15 * tf.random.uniform(tf.shape(real_images))
        noisy_fake_images = fake_images + 0.15 * tf.random.uniform(tf.shape(fake_images))

        # Train the discriminator
        with tf.GradientTape() as d_tape:
            yhat_real = self.discriminator(noisy_real_images, training=True)
            yhat_fake = self.discriminator(noisy_fake_images, training=True)
            d_loss_value = self.d_loss(tf.ones_like(yhat_real), yhat_real) + \
                           self.d_loss(tf.zeros_like(yhat_fake), yhat_fake)

        d_gradients = d_tape.gradient(d_loss_value, self.discriminator.trainable_variables)
        self.d_opt.apply_gradients(zip(d_gradients, self.discriminator.trainable_variables))

        # Train the generator
        with tf.GradientTape() as g_tape:
            gen_images = self.generator(random_latent_vectors, training=True)
            predicted_labels = self.discriminator(gen_images, training=False)
            g_loss_value = self.g_loss(tf.ones_like(predicted_labels), predicted_labels)

        g_gradients = g_tape.gradient(g_loss_value, self.generator.trainable_variables)
        self.g_opt.apply_gradients(zip(g_gradients, self.generator.trainable_variables))

        return {"d_loss": d_loss_value, "g_loss": g_loss_value}

# Optimizers
g_opt = Adam(learning_rate=0.0002, beta_1=0.5)
d_opt = Adam(learning_rate=0.00002, beta_1=0.5)
g_loss = BinaryCrossentropy(from_logits=False)
d_loss = BinaryCrossentropy(from_logits=False)

# Compile GAN
gan = GAN(generator, discriminator)
gan.compile(g_opt, d_opt, g_loss, d_loss)

# Define a custom callback to monitor generated images
class ModelMonitor(tf.keras.callbacks.Callback):
    def __init__(self, num_img=3, latent_dim=128):
        self.num_img = num_img
        self.latent_dim = latent_dim

    def on_epoch_end(self, epoch, logs=None):
        random_latent_vectors = tf.random.normal(shape=(self.num_img, self.latent_dim))
        generated_images = self.model.generator(random_latent_vectors)
        generated_images *= 255
        generated_images = generated_images.numpy().astype("uint8")

        for i in range(self.num_img):
            plt.imshow(generated_images[i])
            plt.axis('off')
            plt.show()

# Train the GAN model
gan.fit(ds, epochs=100, callbacks=[ModelMonitor()])
