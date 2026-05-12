import tensorflow as tf
from tensorflow.keras import mixed_precision
import os
import pathlib
import time
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

mixed_precision.set_global_policy('mixed_float16')

tf.config.optimizer.set_jit(True)

gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        print("GPU ENABLED")

    except RuntimeError as e:
        print(e)

print("TensorFlow Version:", tf.__version__)
print("GPU Devices:", tf.config.list_physical_devices('GPU'))

dataset_name = "facades"

_URL = f'https://efrosgans.eecs.berkeley.edu/pix2pix/datasets/{dataset_name}.tar.gz'

path_to_zip = tf.keras.utils.get_file(
    fname=f"{dataset_name}.tar.gz",
    origin=_URL,
    extract=True
)

path_to_zip = pathlib.Path(path_to_zip)

PATH = path_to_zip.parent / dataset_name


BUFFER_SIZE = 400
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256
OUTPUT_CHANNELS = 3
LAMBDA = 100
STEPS = 3000


def load(image_file):

    image = tf.io.read_file(image_file)

    image = tf.io.decode_jpeg(image)

    w = tf.shape(image)[1]

    w = w // 2

    real_image = image[:, :w, :]

    input_image = image[:, w:, :]

    input_image = tf.cast(input_image, tf.float32)

    real_image = tf.cast(real_image, tf.float32)

    return input_image, real_image

def resize(input_image, real_image, height, width):

    input_image = tf.image.resize(
        input_image,
        [height, width],
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )

    real_image = tf.image.resize(
        real_image,
        [height, width],
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )

    return input_image, real_image


def random_crop(input_image, real_image):

    stacked_image = tf.stack([input_image, real_image], axis=0)

    cropped_image = tf.image.random_crop(
        stacked_image,
        size=[2, IMG_HEIGHT, IMG_WIDTH, 3]
    )

    return cropped_image[0], cropped_image[1]


def normalize(input_image, real_image):

    input_image = (input_image / 127.5) - 1

    real_image = (real_image / 127.5) - 1

    return input_image, real_image


@tf.function
def random_jitter(input_image, real_image):

    input_image, real_image = resize(
        input_image,
        real_image,
        286,
        286
    )

    input_image, real_image = random_crop(
        input_image,
        real_image
    )

    if tf.random.uniform(()) > 0.5:

        input_image = tf.image.flip_left_right(input_image)

        real_image = tf.image.flip_left_right(real_image)

    return input_image, real_image


def load_image_train(image_file):

    input_image, real_image = load(image_file)

    input_image, real_image = random_jitter(
        input_image,
        real_image
    )

    input_image, real_image = normalize(
        input_image,
        real_image
    )

    return input_image, real_image


def load_image_test(image_file):

    input_image, real_image = load(image_file)

    input_image, real_image = resize(
        input_image,
        real_image,
        IMG_HEIGHT,
        IMG_WIDTH
    )

    input_image, real_image = normalize(
        input_image,
        real_image
    )

    return input_image, real_image

train_dataset = tf.data.Dataset.list_files(
    str(PATH / 'train/*.jpg')
)

train_dataset = train_dataset.map(
    load_image_train,
    num_parallel_calls=tf.data.AUTOTUNE
)

train_dataset = train_dataset.cache()

train_dataset = train_dataset.shuffle(BUFFER_SIZE)

train_dataset = train_dataset.batch(BATCH_SIZE)

train_dataset = train_dataset.repeat()

train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

try:
    test_dataset = tf.data.Dataset.list_files(
        str(PATH / 'test/*.jpg')
    )

except:
    test_dataset = tf.data.Dataset.list_files(
        str(PATH / 'val/*.jpg')
    )

test_dataset = test_dataset.map(load_image_test)

test_dataset = test_dataset.cache()

test_dataset = test_dataset.batch(BATCH_SIZE)

test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)


def downsample(filters, size, apply_batchnorm=True):

    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()

    result.add(
        tf.keras.layers.Conv2D(
            filters,
            size,
            strides=2,
            padding='same',
            kernel_initializer=initializer,
            use_bias=False
        )
    )

    if apply_batchnorm:

        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result


def upsample(filters, size, apply_dropout=False):

    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()

    result.add(
        tf.keras.layers.Conv2DTranspose(
            filters,
            size,
            strides=2,
            padding='same',
            kernel_initializer=initializer,
            use_bias=False
        )
    )

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:

        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result


def Generator():

    inputs = tf.keras.layers.Input(
        shape=[IMG_HEIGHT, IMG_WIDTH, 3]
    )

    down_stack = [

        downsample(64, 4, apply_batchnorm=False),

        downsample(128, 4),

        downsample(256, 4),

        downsample(512, 4),

        downsample(512, 4),

        downsample(512, 4),
    ]

    up_stack = [

        upsample(512, 4, apply_dropout=True),

        upsample(512, 4, apply_dropout=True),

        upsample(512, 4),

        upsample(256, 4),

        upsample(128, 4),

        upsample(64, 4),
    ]

    initializer = tf.random_normal_initializer(0., 0.02)

    last = tf.keras.layers.Conv2DTranspose(
        OUTPUT_CHANNELS,
        4,
        strides=2,
        padding='same',
        kernel_initializer=initializer,
        activation='tanh',
        dtype='float32'
    )

    x = inputs

    skips = []

    for down in down_stack:

        x = down(x)

        skips.append(x)

    skips = reversed(skips[:-1])

    for up, skip in zip(up_stack, skips):

        x = up(x)

        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)

generator = Generator()
generator.summary()

def Discriminator():

    initializer = tf.random_normal_initializer(0., 0.02)

    inp = tf.keras.layers.Input(
        shape=[IMG_HEIGHT, IMG_WIDTH, 3],
        name='input_image'
    )

    tar = tf.keras.layers.Input(
        shape=[IMG_HEIGHT, IMG_WIDTH, 3],
        name='target_image'
    )

    x = tf.keras.layers.concatenate([inp, tar])

    down1 = downsample(64, 4, False)(x)

    down2 = downsample(128, 4)(down1)

    down3 = downsample(256, 4)(down2)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)

    conv = tf.keras.layers.Conv2D(
        512,
        4,
        strides=1,
        kernel_initializer=initializer,
        use_bias=False
    )(zero_pad1)

    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)

    last = tf.keras.layers.Conv2D(
        1,
        4,
        strides=1,
        kernel_initializer=initializer,
        dtype='float32'
    )(zero_pad2)

    return tf.keras.Model(inputs=[inp, tar], outputs=last)

discriminator = Discriminator()


loss_object = tf.keras.losses.BinaryCrossentropy(
    from_logits=True
)

def generator_loss(disc_generated_output, gen_output, target):

    gan_loss = loss_object(
        tf.ones_like(disc_generated_output),
        disc_generated_output
    )

    l1_loss = tf.reduce_mean(
        tf.abs(target - gen_output)
    )

    total_gen_loss = gan_loss + (LAMBDA * l1_loss)

    return total_gen_loss


def discriminator_loss(disc_real_output, disc_generated_output):

    real_loss = loss_object(
        tf.ones_like(disc_real_output),
        disc_real_output
    )

    generated_loss = loss_object(
        tf.zeros_like(disc_generated_output),
        disc_generated_output
    )

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss


generator_optimizer = tf.keras.optimizers.Adam(
    1e-4,
    beta_1=0.5
)

discriminator_optimizer = tf.keras.optimizers.Adam(
    1e-4,
    beta_1=0.5
)


checkpoint_dir = './training_checkpoints'

checkpoint_prefix = os.path.join(
    checkpoint_dir,
    "ckpt"
)

checkpoint = tf.train.Checkpoint(
    generator_optimizer=generator_optimizer,
    discriminator_optimizer=discriminator_optimizer,
    generator=generator,
    discriminator=discriminator
)

def generate_images(model, test_input, tar, step):

    prediction = model(test_input, training=False)

    plt.figure(figsize=(12, 6))

    display_list = [
        test_input[0],
        tar[0],
        prediction[0]
    ]

    title = [
        'Input',
        'Ground Truth',
        'Prediction'
    ]

    for i in range(3):

        plt.subplot(1, 3, i + 1)

        plt.title(title[i])

        plt.imshow(display_list[i] * 0.5 + 0.5)

        plt.axis('off')

    os.makedirs(
        "generated_images",
        exist_ok=True
    )

    plt.savefig(
        f"generated_images/output_step_{step}.png"
    )

    plt.close()


@tf.function
def train_step(input_image, target):

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

        gen_output = generator(
            input_image,
            training=True
        )

        disc_real_output = discriminator(
            [input_image, target],
            training=True
        )

        disc_generated_output = discriminator(
            [input_image, gen_output],
            training=True
        )

        gen_loss = generator_loss(
            disc_generated_output,
            gen_output,
            target
        )

        disc_loss = discriminator_loss(
            disc_real_output,
            disc_generated_output
        )

    generator_gradients = gen_tape.gradient(
        gen_loss,
        generator.trainable_variables
    )

    discriminator_gradients = disc_tape.gradient(
        disc_loss,
        discriminator.trainable_variables
    )

    generator_optimizer.apply_gradients(
        zip(generator_gradients, generator.trainable_variables)
    )

    discriminator_optimizer.apply_gradients(
        zip(discriminator_gradients, discriminator.trainable_variables)
    )

    return gen_loss, disc_loss


def fit(train_ds, test_ds, steps):

    example_input, example_target = next(
        iter(test_ds.take(1))
    )

    start = time.time()

    for step, (input_image, target) in enumerate(train_ds.take(steps)):

        gen_loss, disc_loss = train_step(
            input_image,
            target
        )

        if (step + 1) % 100 == 0:

            print(
                f"Step: {step+1} | "
                f"Gen Loss: {gen_loss:.4f} | "
                f"Disc Loss: {disc_loss:.4f}"
            )

        if (step + 1) % 1000 == 0:

            generate_images(
                generator,
                example_input,
                example_target,
                step + 1
            )

            checkpoint.save(
                file_prefix=checkpoint_prefix
            )

            print(
                f"Checkpoint Saved at Step {step+1}"
            )

            print(
                f"Time Taken: {time.time() - start:.2f} sec"
            )

            start = time.time()

if __name__ == "__main__":

    fit(
        train_dataset,
        test_dataset,
        steps=STEPS
    )

    generator.save(
        "pix2pix_generator.keras"
    )

    print("Training Completed")

    # FINAL TEST OUTPUTS

    for inp, tar in test_dataset.take(3):

        prediction = generator(
            inp,
            training=False
        )

        plt.figure(figsize=(12, 6))

        display_list = [
            inp[0],
            tar[0],
            prediction[0]
        ]

        title = [
            'Input',
            'Ground Truth',
            'Prediction'
        ]

        for i in range(3):

            plt.subplot(1, 3, i + 1)

            plt.title(title[i])

            plt.imshow(
                display_list[i] * 0.5 + 0.5
            )

            plt.axis('off')

        plt.show()