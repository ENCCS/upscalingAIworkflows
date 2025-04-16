.. _tf_mltgpus:

Distributed training in TensorFlow
==================================

TensorFlow provides different methods to distribute training with minial coding.
``tf.distribute.Strategy`` is a TensorFlow API to distribute training across
multiple GPUs, multiple machines, or TPUs. Using this API, you can distribute
your existing models.

The main advanges of using ``tf.distribute.Strategy`` are:

- Easy to use and support multiple user segments,
  including researchers, machine learning engineers, etc.
- Provide good performance out of the box.
- Easy switching between strategies.

You can distribute training using ``tf.distribute.Strategy`` with a high-level
API like Keras ``Model.fit``, as we are familiar with, as well as custom training
loops (and, in general, any computation using TensorFlow).
You can use tf.distribute.Strategy with very few changes to your code, because
the underlying components of TensorFlow have been changed to become strategy-aware.
This includes variables, layers, models, optimizers, metrics, summaries, and checkpoints.

Types of strategies
___________________

``tf.distribute.Strategy`` covers several use cases along different axes.
Some of these combinations are currently supported. TensorFlow promises in the website
that others will be added in the future. Some of these axes are:

- **Synchronous vs asynchronous training**: These are two common ways of distributing
  training with data parallelism. In sync training, all workers train over different
  slices of input data in sync, and aggregating gradients at each step. In async training,
  all workers are independently training over the input data and updating variables asynchronously.
  Typically sync training is supported via all-reduce and async through parameter server architecture.

- **Hardware platform**: You may want to scale your training onto multiple GPUs on
  one machine, or multiple machines in a network (with 0 or more GPUs each), or on Cloud TPUs.

MirroredStrategy
++++++++++++++++

``tf.distribute.MirroredStrategy`` supports synchronous distributed training on
multiple GPUs on one machine. It creates one replica per GPU device. Each variable
in the model is mirrored across all the replicas. Together, these variables form
a single conceptual variable called MirroredVariable. These variables are kept
in sync with each other by applying identical updates.

Efficient all-reduce algorithms are used to communicate the variable updates across
the devices. All-reduce aggregates tensors across all the devices by adding them up,
and makes them available on each device. It’s a fused algorithm that is very efficient
and can reduce the overhead of synchronization significantly. There are many all-reduce
algorithms and implementations available, depending on the type of communication available
between devices. By default, it uses the NVIDIA Collective Communication Library (`NCCL <https://developer.nvidia.com/nccl>`_)
as the all-reduce implementation.

The main features of ``tf.distribute.MirroredStrategy``:

- All the variables and the model graph is replicated on the replicas.
- Input is evenly distributed across the replicas.
- Each replica calculates the loss and gradients for the input it received.
- The gradients are synced across all the replicas by summing them.
- After the sync, the same update is made to the copies of the variables on each replica.

We can initiate the strategy Using

.. code-block :: python

  strategy = tf.distribute.MirroredStrategy()

If the list of devices is not specified in the ``tf.distribute.MirroredStrategy``
constructor, it will be auto-detected. For exmaple, if we book a node with 5 GPUs,
the result of

.. code-block :: python

  print ('Number of devices: {}'.format(strategy.num_replicas_in_sync))

will be

.. code-block :: bash

  Number of devices: 5

Let's apply the ``tf.distribute.MirroredStrategy`` to the fashion MINST dataset.
We can start by downloading, and transforming the data into proper format.

.. code-block :: python

  fashion_mnist = tf.keras.datasets.fashion_mnist
  (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

  # Adding a dimension to the array -> new shape == (28, 28, 1)
  # We are doing this because the first layer in our model is a convolutional
  # layer and it requires a 4D input (batch_size, height, width, channels).
  # batch_size dimension will be added later on.
  train_images = train_images[..., None]
  test_images = test_images[..., None]

  # Getting the images in [0, 1] range.
  train_images = train_images / np.float32(255)
  test_images = test_images / np.float32(255)

We need to change the shape of dataset in order to feed it to the model. The
global batch sizes is equal to the batch size*number of replicas because each
replica will take a batch per run.

.. code-block :: python

  BUFFER_SIZE = len(train_images)
  BATCH_SIZE_PER_REPLICA = 64
  GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
  EPOCHS = 10

Tranforming to the TensorFlow type tensor dataset and distributing among replicas

.. code-block :: python

  train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(BUFFER_SIZE).batch(GLOBAL_BATCH_SIZE)
  test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(GLOBAL_BATCH_SIZE)

  train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)
  test_dist_dataset = strategy.experimental_distribute_dataset(test_dataset)

We use ``tf.keras.callbacks`` for different purposes. Here, three callbacks are

- ``tf.keras.callbacks.TensorBoard``: writes a log for TensorBoard, which allows
  you to visualize the graphs.
- ``tf.keras.callbacks.ModelCheckpoint``: saves the model at a certain frequency,
  such as after every epoch.
- ``tf.keras.callbacks.LearningRateScheduler``: schedules the learning rate to
  change after, for example, every epoch/batch.

The setup for the saving the checkpoint callback is:

.. code-block :: python

  # Define the checkpoint directory to store the checkpoints.
  checkpoint_dir = './training_checkpoints'
  # Define the name of the checkpoint files.
  checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

For the decay learning rate is:

.. code-block:: python

  # Define a function for decaying the learning rate.
  # You can define any decay function you need.
  def decay(epoch):
  if epoch < 3:
    return 1e-3
  elif epoch >= 3 and epoch < 7:
    return 1e-4
  else:
    return 1e-5

And for printing the learning rate at the end of each epoch:

.. code-block:: python

  class PrintLR(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
      print('\nLearning rate for epoch {} is {}'.format(epoch + 1, model.optimizer.lr.numpy()))

Put all of the callbacks together.

.. code-block:: python

   callbacks = [
     tf.keras.callbacks.TensorBoard(log_dir='./logs'),
     tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix, save_weights_only=True),
     tf.keras.callbacks.LearningRateScheduler(decay),
     PrintLR()]

For illustrative purposes, a custom callback called ``PrintLR`` was added
to display the learning rate in the notebook.

Training with ``Model.fit``
+++++++++++++++++++++++++++

After defining the model with proper loss function, for example

.. code-block :: python

  with strategy.scope():
  model = tf.keras.Sequential([
      tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape = [28,28,1]),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Conv2D(64, 3, activation='relu'),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(64, activation='relu'),
      tf.keras.layers.Dense(10)])

  model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy'])

Now, we can simply call the usual ``Model.fit`` function to train the model!

.. code-block :: python

  start = time.time()
  model.fit(train_dataset, epochs=EPOCHS, callbacks=callbacks)
  endt = time.time()-start
  print("Time for {} epochs: {:0.2f}ms".format(EPOCHS,1000*endt))

Which will print

.. code-block :: python

  Epoch 1/10
  188/188 [==============================] - 6s 29ms/step - loss: 0.2341 - accuracy: 0.9160
  Epoch 2/10
  188/188 [==============================] - 2s 9ms/step - loss: 0.2243 - accuracy: 0.9188
  Epoch 3/10
  188/188 [==============================] - 2s 9ms/step - loss: 0.2174 - accuracy: 0.9220
  Epoch 4/10
  188/188 [==============================] - 2s 9ms/step - loss: 0.2111 - accuracy: 0.9232
  Epoch 5/10
  188/188 [==============================] - 2s 9ms/step - loss: 0.2045 - accuracy: 0.9260
  Epoch 6/10
  188/188 [==============================] - 2s 9ms/step - loss: 0.1954 - accuracy: 0.9291
  Epoch 7/10
  188/188 [==============================] - 2s 9ms/step - loss: 0.1878 - accuracy: 0.9327
  Epoch 8/10
  188/188 [==============================] - 2s 9ms/step - loss: 0.1856 - accuracy: 0.9326
  Epoch 9/10
  188/188 [==============================] - 2s 9ms/step - loss: 0.1737 - accuracy: 0.9372
  Epoch 10/10
  188/188 [==============================] - 2s 9ms/step - loss: 0.1676 - accuracy: 0.9390
  Time for 10 epochs: 25876.68ms

That simple!! ``tf.keras`` APIs to build the model and ``Model.fit`` for training it
made the

Custom loop training
++++++++++++++++++++

In cases where we need to customize the training procedure, we still are able to use
the ``tf.distribute.MirroredStrategy``. Here, the setup is a bit more elaborated and
needs some care. Let's create a model using ``tf.keras.Sequential``.
We can also use the Model Subclassing API to do this.

.. code-block :: python

  def create_model():
    model = tf.keras.Sequential([
      tf.keras.layers.Conv2D(32, 3, activation='relu'),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Conv2D(64, 3, activation='relu'),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(64, activation='relu'),
      tf.keras.layers.Dense(10)])

    return model

Normally, on a single machine with 1 GPU/CPU, loss is divided by the number of examples
in the batch of input. How should the loss function be calculated within ``tf.distribute.Strategy``?

It requires special care. Why?

- For an example, let's say you have 4 GPU's and a batch size of 64. One batch of input is
  distributed across the replicas (4 GPUs), each replica getting an input of size 16.

- The model on each replica does a forward pass with its respective input and calculates the loss.
  Now, instead of dividing the loss by the number of examples in its respective input
  (``BATCH_SIZE_PER_REPLICA = 16``), the loss should be divided by the ``GLOBAL_BATCH_SIZE (64)``.

**Why do this?**

- This needs to be done because after the gradients are calculated on each replica,
  they are synced across the replicas by summing them.

How to do this in TensorFlow?

- If we're writing a custom training loop, as in this tutorial, you should sum
  the per example losses and divide the sum by the GLOBAL_BATCH_SIZE:
  ``scale_loss = tf.reduce_sum(loss) * (1. / GLOBAL_BATCH_SIZE)``
  or you can use tf.nn.compute_average_loss which takes the per example loss,
  optional sample weights, and GLOBAL_BATCH_SIZE as arguments and returns the scaled loss.
- If you are using regularization losses in your model then you need to scale
  the loss value by number of replicas. You can do this by using the
  ``tf.nn.scale_regularization_loss`` function.
- Using ``tf.reduce_mean`` is not recommended. Doing so divides the loss by actual
  per replica batch size which may vary step to step. More on this below.
- This reduction and scaling is done automatically in keras ``model.compile``
  and ``model.fit`` (Why aren't we grateful then?!)
- If using ``tf.keras.losses`` classes (as in the example below),
  the loss reduction needs to be explicitly specified to be one of ``NONE`` or ``SUM``.
  ``AUTO`` and ``SUM_OVER_BATCH_SIZE`` are disallowed when used with ``tf.distribute.Strategy``.
  ``AUTO`` is disallowed because the user should explicitly think about what reduction
  they want to make sure it is correct in the distributed case. ``SUM_OVER_BATCH_SIZE``
  is disallowed because currently it would only divide by per replica batch size,
  and leave the dividing by number of replicas to the user, which might be easy to miss.
  So the user must do the reduction themselves explicitly.
- If ``labels`` is multi-dimensional, then average the ``per_example_loss`` across
  the number of elements in each sample. For example, if the shape of ``predictions``
  is ``(batch_size, H, W, n_classes)`` and labels is ``(batch_size, H, W)``,
  you will need to update ``per_example_loss`` like:
  ``per_example_loss /= tf.cast(tf.reduce_prod(tf.shape(labels)[1:]), tf.float32)``

.. callout :: Verify the shape of the loss

  Loss functions in tf.losses/tf.keras.losses typically return the average over
  the last dimension of the input. The loss classes wrap these functions. Passing
  ``reduction=Reduction.NONE`` when creating an instance of a loss class means
  "no additional reduction". For categorical losses with an example input shape of
  ``[batch, W, H, n_classes]`` the n_classes dimension is reduced. For pointwise
  losses like ``losses.mean_squared_error`` or ``losses.binary_crossentropy`` include
  a dummy axis so that ``[batch, W, H, 1]`` is reduced to [batch, W, H].
  Without the dummy axis ``[batch, W, H]`` will be incorrectly reduced to ``[batch, W]``.

.. code-block :: python

  with strategy.scope():
  # Set reduction to `none` so we can do the reduction afterwards and divide by
  # global batch size.
  loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
      from_logits=True,
      reduction=tf.keras.losses.Reduction.NONE)
  def compute_loss(labels, predictions):
      per_example_loss = loss_object(labels, predictions)
      return tf.nn.compute_average_loss(per_example_loss, global_batch_size=GLOBAL_BATCH_SIZE)

By defining the metrics, we track the test loss and training and test accuracy.
We can use .result() to get the accumulated statistics at any time.

.. code-block :: python

  with strategy.scope():
    test_loss = tf.keras.metrics.Mean(name='test_loss') # from logits

    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

Model, optimizer, and checkpoint must be created under ``strategy.scope``.

.. code-block :: python

  with strategy.scope():
    model = create_model()

    optimizer = tf.keras.optimizers.Adam()
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)

Calculations of loss, gradients and updating the gradients

.. code-block :: python

  def train_step(inputs):
    images, labels = inputs

    with tf.GradientTape() as tape:
      predictions = model(images, training=True)
      loss = compute_loss(labels, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_accuracy.update_state(labels, predictions)
    return loss

  def test_step(inputs):
    images, labels = inputs

    predictions = model(images, training=False)
    t_loss = loss_object(labels, predictions)

    test_loss.update_state(t_loss)
    test_accuracy.update_state(labels, predictions)

The ``run`` command replicates the provided computation and runs it with
the distributed input.

.. code-block :: python

  @tf.function
  def distributed_train_step(dataset_inputs):
    per_replica_losses = strategy.run(train_step, args=(dataset_inputs,))
    return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                           axis=None)

  @tf.function
  def distributed_test_step(dataset_inputs):
    return strategy.run(test_step, args=(dataset_inputs,))

  import time

  start = time.time()

  for epoch in range(EPOCHS):
    # TRAIN LOOP
    total_loss = 0.0
    num_batches = 0
    for x in train_dist_dataset:
      total_loss += distributed_train_step(x)
      num_batches += 1
    train_loss = total_loss / num_batches

    # TEST LOOP
    for x in test_dist_dataset:
      distributed_test_step(x)

    if epoch % 2 == 0:
      checkpoint.save(checkpoint_prefix)

    template = ("Epoch {}, Loss: {:0.2f}, Accuracy: {:0.2f}, Test Loss: {:0.2f}, "
                "Test Accuracy: {:0.2f}")
    print (template.format(epoch+1, train_loss,
                           train_accuracy.result()*100, test_loss.result(),
                           test_accuracy.result()*100))

    test_loss.reset_states()
    train_accuracy.reset_states()
    test_accuracy.reset_states()

  endt = time.time()
  timelp = 1000*(endt-start)

  print("Elapsed time in (ms): {:0.2f}".format(timelp))

The output will be

.. code-block :: python

  INFO:tensorflow:batch_all_reduce: 8 all-reduces with algorithm = nccl, num_packs = 1
  INFO:tensorflow:batch_all_reduce: 8 all-reduces with algorithm = nccl, num_packs = 1
  INFO:tensorflow:batch_all_reduce: 8 all-reduces with algorithm = nccl, num_packs = 1
  Epoch 1, Loss: 0.71, Accuracy: 74.71, Test Loss: 0.48, Test Accuracy: 83.05
  Epoch 2, Loss: 0.43, Accuracy: 84.76, Test Loss: 0.41, Test Accuracy: 85.70
  Epoch 3, Loss: 0.37, Accuracy: 86.96, Test Loss: 0.37, Test Accuracy: 86.63
  Epoch 4, Loss: 0.34, Accuracy: 87.95, Test Loss: 0.37, Test Accuracy: 86.86
  Epoch 5, Loss: 0.32, Accuracy: 88.60, Test Loss: 0.34, Test Accuracy: 87.69
  Epoch 6, Loss: 0.30, Accuracy: 89.36, Test Loss: 0.32, Test Accuracy: 88.93
  Epoch 7, Loss: 0.28, Accuracy: 89.61, Test Loss: 0.31, Test Accuracy: 88.64
  Epoch 8, Loss: 0.27, Accuracy: 90.05, Test Loss: 0.32, Test Accuracy: 88.64
  Epoch 9, Loss: 0.26, Accuracy: 90.50, Test Loss: 0.29, Test Accuracy: 89.60
  Epoch 10, Loss: 0.25, Accuracy: 90.98, Test Loss: 0.29, Test Accuracy: 89.33
  Elapsed time in (ms): 39034.53

Single GPU calculations
+++++++++++++++++++++++

For the sake of comparision, let's repeat the calculations on a single GPU.

.. code-block :: python

  def model_sngpu(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape = input_shape),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10)
    ])

    model.compile(optimizer = 'adam', loss = tf.keras.losses.SparseCategoricalCrossentropy(
      from_logits=True), metrics = ['accuracy'])

    return model

.. code-block :: python

  start = time.time()
  with tf.device("GPU:0"):
      model_sngp = model_sngpu([28,28,1])
      history = model_sngp.fit(train_images, train_labels, epochs = EPOCHS,
                              batch_size=GLOBAL_BATCH_SIZE, validation_split = 0.15)
  endt = time.time()-start
  print("Time for {} epochs: {:0.2f}ms".format(EPOCHS,1000*endt))

The output will be

.. code-block :: python

  Epoch 1/10
  160/160 [==============================] - 2s 9ms/step - loss: 0.7309 - accuracy: 0.7413 - val_loss: 0.4898 - val_accuracy: 0.8129
  Epoch 2/10
  160/160 [==============================] - 1s 8ms/step - loss: 0.4256 - accuracy: 0.8485 - val_loss: 0.3918 - val_accuracy: 0.8606
  Epoch 3/10
  160/160 [==============================] - 1s 8ms/step - loss: 0.3674 - accuracy: 0.8710 - val_loss: 0.3627 - val_accuracy: 0.8679
  Epoch 4/10
  160/160 [==============================] - 1s 8ms/step - loss: 0.3428 - accuracy: 0.8791 - val_loss: 0.3453 - val_accuracy: 0.8757
  Epoch 5/10
  160/160 [==============================] - 1s 8ms/step - loss: 0.3220 - accuracy: 0.8848 - val_loss: 0.3342 - val_accuracy: 0.8808
  Epoch 6/10
  160/160 [==============================] - 1s 8ms/step - loss: 0.3038 - accuracy: 0.8910 - val_loss: 0.3342 - val_accuracy: 0.8826
  Epoch 7/10
  160/160 [==============================] - 1s 8ms/step - loss: 0.2885 - accuracy: 0.8960 - val_loss: 0.3154 - val_accuracy: 0.8876
  Epoch 8/10
  160/160 [==============================] - 1s 8ms/step - loss: 0.2752 - accuracy: 0.9011 - val_loss: 0.2992 - val_accuracy: 0.8918
  Epoch 9/10
  160/160 [==============================] - 1s 8ms/step - loss: 0.2647 - accuracy: 0.9038 - val_loss: 0.3161 - val_accuracy: 0.8834
  Epoch 10/10
  160/160 [==============================] - 1s 8ms/step - loss: 0.2569 - accuracy: 0.9066 - val_loss: 0.2810 - val_accuracy: 0.9003
  Time for 10 epochs: 13603.21ms

.. callout :: Compare the results

   Now have three time elapsed using three different methods:

    1. MirroredStrategy - ``Model.fit``: 25876.68ms
    2. MirroredStrategy - custom loop  : 39034.53ms
    3. A single GPU - ``Model.fit``    : 13603.21ms

   As we can see, distributed training not only did not improve the elapsed time
   but also substantially incresed it! Can you explain why?

The ``for`` loop that marches though the input (training or test datasets) can be implemented
using other methods too. For example, one can make use of Python iterator functions
``iter`` and ``next``. Using iterator we have more control over the number of steps we wish to
execute the commands. Another way of implementing could be using ``for`` inside ``tf.function``.

ParameterServerStrategy
+++++++++++++++++++++++

Parameter server training is a common data-parallel method to scale up model training on
multiple machines. A parameter server training cluster consists of workers and parameter servers.
Variables are created on parameter servers and they are read and updated by workers in each step.
Similar to ``MirroredStrategy``, it can be implemented using Keras API ``Model.fit`` or custom
training loop.

In TensorFlow 2, parameter server training uses a central coordinator-based architecture via the
``tf.distribute.experimental.coordinator.ClusterCoordinator`` class. In this implementation,
the worker and parameter server tasks run ``tf.distribute.Servers`` that listen for tasks
from the coordinator. The coordinator creates resources, dispatches training tasks, writes
checkpoints, and deals with task failures.

In the programming running on the coordinator, one uses a ``ParameterServerStrategy`` object to
define a training step and use a ``ClusterCoordinator`` to dispatch training steps to remote workers.

MultiWorkerMirroredStrategy
+++++++++++++++++++++++++++

``tf.distribute.MultiWorkerMirroredStrategy`` is very similar to ``MirroredStrategy``. It implements
synchronous distributed training across multiple workers, each with potentially multiple GPUs.
Similar to tf.distribute.MirroredStrategy, it creates copies of all variables in the model on
each device across all workers. One of the key differences to get multi worker training going,
as compared to multi-GPU training, is the multi-worker setup. The 'TF_CONFIG' environment variable
is the standard way in TensorFlow to specify the cluster configuration to each worker that is part
of the cluster. In other words, the main difference between ``MultiWorkerMirroredStrategy`` and
``MirroredStrategy`` is While in *MultiWorkerMirroredStrategy*, the network setup is necessary,
in *MirroredStrategy* the setup is automatically topology aware meaning that we don't need
to setup the network and interconnects.

.. exercise :: Distributed training for SVHN dataset

  Use the Jupyter notebook provide in the previous session to implement MirroredStrategy
  using both ``Model.fit`` and custom training loop methods. Compare your results with
  training on a single GPU calculations. Does the conclusion we had above holds here too?

  **Advance** Load a checkpoint and evaluate the performance of the metrics on the tests
  datasets. For each of ``Model.fit`` and custom training loop, you should find proper
  set of commands.

  .. solution:: Discussion

    Similar steps applied in this section can be applied to the notebook.

    **Advance**
    ``Model.fit``

    .. code-block :: python

      model = create_model()
      model.load_weights(checkpoint_path)
      loss, acc = model.evaluate(test_images, test_labels, verbose=2)
      print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

    Custom training

    .. code-block :: python

      eval_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='eval_accuracy')
      model = create_model()
      optimizer = tf.keras.optimizers.Adam()
      test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(GLOBAL_BATCH_SIZE)

      @tf.function
      def eval_step(images, labels):
        predictions = model(images, training=False)
        eval_accuracy(labels, predictions)

      checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
      checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

      for images, labels in test_dataset:
        eval_step(images, labels)

      print ('Restored model, accuracy : {:5.2f}%'.format(eval_accuracy.result()*100))
