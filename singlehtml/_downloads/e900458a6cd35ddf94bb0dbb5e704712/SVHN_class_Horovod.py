import tensorflow as tf
import horovod.tensorflow as hvd

hvd.init()

# Pinning GPUs (one GPU per process)
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
    
(mnist_images, mnist_labels), _ = tf.keras.datasets.mnist.load_data(path='mnist-%d.npz' % hvd.rank())

dataset = tf.data.Dataset.from_tensor_slices(
        (tf.cast(mnist_images[..., tf.newaxis] / 255.0, tf.float32),
         tf.cast(mnist_labels, tf.int64)))

batch_size = 64
dataset = dataset.repeat().shuffle(10000).batch(batch_size)

# Horovod: adjust learning rate based on number of GPUs.
scaled_lr = 0.001 * hvd.size()
opt = tf.optimizers.Adam(scaled_lr)
opt = hvd.DistributedOptimizer(opt, backward_passes_per_step=1, average_aggregated_gradients=True)

mnist_model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
    ])

mnist_model.compile(loss=tf.losses.SparseCategoricalCrossentropy(),
                    optimizer=opt,
                    metrics=['accuracy'],
                    experimental_run_tf_function=False
                    )

callbacks = [
    horovod.tensorflow.keras.callbacks.BroadcastGlobalVariablesCallback(0),
    horovod.tensorflow.keras.callbacks.MetricAverageCallback(),
    horovod.tensorflow.keras.callbacks.LearningRateWarmupCallback(initial_lr=scaled_lr,
    warmup_epochs=3, verbose=1),
    ]

if hvd.rank() == 0:
    callbacks.append(tf.keras.callbacks.ModelCheckpoint('./checkpoint-{epoch}.h5'))
verbose = 1 if hvd.rank() == 0 else 0

mnist_model.fit(dataset, steps_per_epoch=500 // hvd.size(), callbacks=callbacks, epochs=24, verbose=verbose)
