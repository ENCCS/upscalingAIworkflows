.. _tf_mltgpus:

Distributed training in TensorFlow
==================================

TensorFlow provides different methods to distribute training with minimal coding.
``tf.distribute.Strategy`` is a TensorFlow API to distribute training across
multiple GPUs, multiple machines, or TPUs. Using this API, you can distribute
your existing models.

The main advantages of using ``tf.distribute.Strategy``, according to TensorFlow, are:

- Easy to use and support multiple user segments,
  including researchers, machine learning engineers, etc.
- Provide good performance out of the box.
- Easy switching between strategies.

You can distribute training using ``tf.distribute.Strategy`` with a high-level
API like Keras ``Model.fit``, as we are familiar with, as well as custom training
loops (and, in general, any computation using TensorFlow).
You can use ``tf.distribute.Strategy`` with very few changes to your code, because
the underlying components of TensorFlow have been changed to become strategy-aware.
This includes variables, layers, models, optimizers, metrics, summaries, and checkpoints.

Types of strategies
-------------------

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
----------------
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
between devices. By default, it uses the NVIDIA Collective Communication Library 
(`NCCL <https://developer.nvidia.com/nccl>`_) as the all-reduce implementation.

The main features of ``tf.distribute.MirroredStrategy``:

- All the variables and the model graph is replicated on the replicas.
- Input is evenly distributed across the replicas.
- Each replica calculates the loss and gradients for the input it received.
- The gradients are synced across all the replicas by summing them.
- After the sync, the same update is made to the copies of the variables on each replica.

We can initiate the strategy Using

.. code-block:: python

  strategy = tf.distribute.MirroredStrategy()

If the list of devices is not specified in the ``tf.distribute.MirroredStrategy``
constructor, it will be auto-detected. For example, if we book a node with 5 GPUs,
the result of

.. code-block:: python

  print ('Number of devices: {}'.format(strategy.num_replicas_in_sync))

will be

.. code-block:: bash

  Number of devices: 5

Let's apply the ``tf.distribute.MirroredStrategy`` to the Quora dataset using the NNLM model.
Since we already have download the dataset and saved as a pickle library, we can simply use 
loading part from previous section.

We need to change the shape of dataset in order to feed it to the model. The
global batch sizes is equal to the batch size*number of replicas because each
replica will take a batch per run.

.. code-block:: python

   buffer_size = train_df.size
   batch_size_per_replica = 64
   global_batch_size = batch_size_per_replica * strategy.num_replicas_in_sync
   
Transforming to the TensorFlow type tensor dataset and distributing among replicas

.. code-block:: python

   train_dataset = (tf.data.Dataset.from_tensor_slices((train_df.question_text.values, train_df.target.values))
                   .shuffle(buffer_size) 
                   .batch(global_batch_size, drop_remainder=True)
                   .prefetch(tf.data.experimental.AUTOTUNE)) #.shuffle(buffer_size)
   
   valid_dataset = (tf.data.Dataset.from_tensor_slices((valid_df.question_text.values, valid_df.target.values))
                   .batch(global_batch_size, drop_remainder=True)
                   .prefetch(tf.data.experimental.AUTOTUNE))

We use ``tf.keras.callbacks`` for different purposes. Here, three callbacks are

- ``tf.keras.callbacks.TensorBoard``: writes a log for TensorBoard, which allows
  you to visualize the graphs.

- ``tfdocs.modeling.EpochDots()``:  To reduce the logging noise use the tfdocs.EpochDots 
  which simply prints a . for each epoch, and a full set of metrics every 100 epochs.

- ``tf.keras.callbacks.EarlyStopping`` : to avoid long and unnecessary training times. 
  This callback is set to monitor the val_loss.

There are other callbacks which can be of interests for specific purposes. Nonetheless, we just use
the callbacks mentioned above.

Training with ``Model.fit``
---------------------------

We define a function to instantiate the model, train it and returns the history object.

.. code-block:: python

   def train_and_evaluate_model(module_url, embed_size, name, trainable=False):
       hub_layer = hub.KerasLayer(module_url, input_shape=[], output_shape=[embed_size], dtype = tf.string, trainable=trainable)
       model = tf.keras.models.Sequential([
                                         hub_layer,
                                         tf.keras.layers.Dense(256, activation='relu'),
                                         tf.keras.layers.Dense(64, activation='relu'),
                                         tf.keras.layers.Dense(1, activation='sigmoid')
       ])
   
       model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                   loss = tf.losses.BinaryCrossentropy(),
                   metrics = [tf.metrics.BinaryAccuracy(name='accuracy')])
       history = model.fit(train_dataset, #train_df['question_text'], train_df['target'],
                         epochs = 100,
                         batch_size=32,
                         validation_data=valid_dataset, #(valid_df['question_text'], valid_df['target']),
                         callbacks=[tfdocs.modeling.EpochDots(),
                                    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, mode='min'),
                                    tf.keras.callbacks.TensorBoard(logdir/name)],
                         verbose = 0
                         )
       return history

Now, we can simply call the usual ``Model.fit`` function to train the model!

.. code-block:: python

   with strategy.scope():
      start = time.time()
      histories['nnlm-en-dim128'] = train_and_evaluate_model(module_url, embed_size=128, name='nnlm-en-dim128')
      endt = time.time()-start
      print("\n \n Time for {} ms".format(1000*endt))

Which will print

.. code-block:: python

   Epoch: 0, accuracy:0.9326,  loss:0.2864,  val_accuracy:0.9385,  val_loss:0.1761,  
   .....................
 
   Time for 85504.98509407043 ms

That simple! ``tf.keras`` APIs to build the model and ``Model.fit`` for training it
made the distributed training very easy.

Custom loop training
--------------------

In cases where we need to customize the training procedure, we still are able to use
the ``tf.distribute.MirroredStrategy``. Here, the setup is a bit more elaborated and
needs some care. Let's create a model using ``tf.keras.Sequential``.

There is a difference to create the datasets in comparison to the previous section; as will be explained
below, here we need to add a dummy dimension to our dataset_inputs.

.. code-block:: python

   buffer_size = train_df.size
   batch_size_per_replica = 64
   global_batch_size = batch_size_per_replica * strategy.num_replicas_in_sync
   
   train_dataset = (tf.data.Dataset.from_tensor_slices((train_df.question_text.values[...,None], train_df.target.values[...,None]))
                   .shuffle(buffer_size)
                   .batch(global_batch_size, drop_remainder=True)
                   .prefetch(tf.data.experimental.AUTOTUNE))
   
   valid_dataset = (tf.data.Dataset.from_tensor_slices((valid_df.question_text.values[...,None], valid_df.target.values[...,None]))
                   .batch(global_batch_size, drop_remainder=True)
                   .prefetch(tf.data.experimental.AUTOTUNE))
   
The model function can be defined using Keras Sequential API.

.. code-block:: python

   module_url = "https://tfhub.dev/google/tf2-preview/nnlm-en-dim128/1"
   embeding_size = 128
   name_of_model = 'nnlm-en-dim128/1'
   def create_model(module_url, embed_size, name, trainable=False):
       hub_layer = hub.KerasLayer(module_url, input_shape=[],
                                  output_shape=[embed_size],dtype = tf.string, trainable=trainable)
       
       model = tf.keras.models.Sequential([hub_layer,
                                           tf.keras.layers.Dense(256, activation='relu'),
                                           tf.keras.layers.Dense(64, activation='relu'),
                                           tf.keras.layers.Dense(1, activation='sigmoid')])
       return model

Calculation of loss with Mirrored Strategy:
-------------------------------------------

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

.. callout:: Verify the shape of the loss

  Loss functions in tf.losses/tf.keras.losses typically return the average over
  the last dimension of the input. The loss classes wrap these functions. Passing
  ``reduction=Reduction.NONE`` when creating an instance of a loss class means
  "no additional reduction". For categorical losses with an example input shape of
  ``[batch, W, H, n_classes]`` the n_classes dimension is reduced. For pointwise
  losses like ``losses.mean_squared_error`` or ``losses.binary_crossentropy`` include
  a dummy axis so that ``[batch, W, H, 1]`` is reduced to [batch, W, H].
  Without the dummy axis ``[batch, W, H]`` will be incorrectly reduced to ``[batch, W]``.

.. code-block:: python

   with strategy.scope():
   # Set reduction to `none` so we can do the reduction afterwards and divide by
   # global batch size.
   
       loss_object = tf.losses.BinaryCrossentropy(
           from_logits=False,
           reduction=tf.losses.Reduction.NONE)
   
       def compute_loss(labels, predictions):
           per_example_loss = loss_object(labels, predictions)
           return tf.nn.compute_average_loss(per_example_loss, global_batch_size=global_batch_size)
           
       train_accuracy = tf.metrics.BinaryAccuracy(name='train_accuracy')
       valid_accuracy = tf.metrics.BinaryAccuracy(name='valid_accuracy')
       
       model = create_model(module_url, embed_size=embeding_size, name=name_of_model, trainable=False)
       optimizer = tf.optimizers.Adam()

By defining the metrics, we track the test loss and training and test accuracy.
We can use .result() to get the accumulated statistics at any time.

The next step is the calculations of loss, gradients and updating the gradients.

.. code-block:: python

   def train_step(inputs):
       texts, labels = inputs
   
       with tf.GradientTape() as tape:
           predictions = model(texts, training=True)
           loss = compute_loss(labels, predictions)
   
       gradients = tape.gradient(loss, model.trainable_variables)
       optimizer.apply_gradients(zip(gradients, model.trainable_variables))
   
       train_accuracy.update_state(labels, predictions)
       return loss 
   
   def valid_step(inputs):
       texts, labels = inputs
   
       predictions = model(texts, training=False)
       v_loss = compute_loss(labels, predictions)
       
       valid_accuracy.update_state(labels, predictions)
   
       return v_loss

Before being able to run the training, we need to create ``replica datasets`` objects for 
distributed training using

.. code-block:: python

   train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)
   valid_dist_dataset = strategy.experimental_distribute_dataset(valid_dataset)

The ``run`` command replicates the provided computation and runs it with
the distributed input.

.. code-block:: python

   epochs = 20
   # `run` replicates the provided computation and runs it
   # with the distributed input.
   @tf.function
   def distributed_train_step(dataset_inputs):
       per_replica_losses = strategy.run(train_step, args=(dataset_inputs,))
       return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                            axis=None)
   
   @tf.function
   def distributed_valid_step(dataset_inputs):
       per_replica_losses = strategy.run(valid_step, args=(dataset_inputs,))
       return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                            axis=None)
   
   history_df = pd.DataFrame(columns=['epochs', 'train_loss', 'valid_loss', 'train_acc', 'valid_acc'])
   
   start = time.time()
   for epoch in range(epochs):
       # TRAIN LOOP
       total_loss = 0.0
       num_batches = 0
       
       for x in train_dist_dataset:
           total_loss += distributed_train_step(x)
           num_batches += 1
       train_loss = total_loss / num_batches
   
       # TEST LOOP
       v_total_loss = 0.0
       v_num_batches = 0
       for x in valid_dist_dataset:
           v_total_loss += distributed_valid_step(x)
           v_num_batches += 1
       valid_loss = v_total_loss / v_num_batches
   
       template = ("Epoch {}, Loss: {:.4f}, Accuracy: {:.4f}, Valid Loss: {:.4f}, Valid Accuracy: {:.4f}")
       print(template.format(epoch + 1, train_loss,
                            train_accuracy.result() * 100, 
                            valid_loss,
                            valid_accuracy.result() * 100))
       
       history_df = history_df.append({'epochs':epoch + 1,
                                       'train_loss':train_loss.numpy(),
                                       'valid_loss':valid_loss.numpy(),
                                       'train_acc':train_accuracy.result().numpy() * 100,
                                       'valid_acc':valid_accuracy.result().numpy() * 100},
                                     ignore_index=True)
       
       train_accuracy.reset_states()
       valid_accuracy.reset_states()
   
   endt = time.time()
   timelp = 1000*(endt-start)

The output will be

.. code-block:: python

  Epoch 1, Loss: 0.1653, Accuracy: 94.3007, Valid Loss: 0.1384, Valid Accuracy: 94.6289
  Epoch 2, Loss: 0.1416, Accuracy: 94.7266, Valid Loss: 0.1334, Valid Accuracy: 95.3125
  Epoch 3, Loss: 0.1371, Accuracy: 94.9104, Valid Loss: 0.1311, Valid Accuracy: 95.0195
  Epoch 4, Loss: 0.1322, Accuracy: 95.0705, Valid Loss: 0.1266, Valid Accuracy: 95.1172
  Epoch 5, Loss: 0.1271, Accuracy: 95.2275, Valid Loss: 0.1306, Valid Accuracy: 94.8242
  Epoch 6, Loss: 0.1225, Accuracy: 95.3569, Valid Loss: 0.1329, Valid Accuracy: 94.6289
  Epoch 7, Loss: 0.1174, Accuracy: 95.5476, Valid Loss: 0.1367, Valid Accuracy: 95.0195
  Epoch 8, Loss: 0.1124, Accuracy: 95.7629, Valid Loss: 0.1374, Valid Accuracy: 94.8242
  Epoch 9, Loss: 0.1073, Accuracy: 95.9566, Valid Loss: 0.1430, Valid Accuracy: 94.8242
  Epoch 10, Loss: 0.1024, Accuracy: 96.1298, Valid Loss: 0.1481, Valid Accuracy: 94.6289
  Epoch 11, Loss: 0.0970, Accuracy: 96.3626, Valid Loss: 0.1521, Valid Accuracy: 94.4336
  Epoch 12, Loss: 0.0918, Accuracy: 96.5173, Valid Loss: 0.1577, Valid Accuracy: 94.5312
  Epoch 13, Loss: 0.0867, Accuracy: 96.7639, Valid Loss: 0.1723, Valid Accuracy: 94.4336
  Epoch 14, Loss: 0.0814, Accuracy: 96.9439, Valid Loss: 0.1681, Valid Accuracy: 94.1406
  Epoch 15, Loss: 0.0774, Accuracy: 97.0573, Valid Loss: 0.1741, Valid Accuracy: 94.4336
  Epoch 16, Loss: 0.0719, Accuracy: 97.2963, Valid Loss: 0.1874, Valid Accuracy: 93.8477
  Epoch 17, Loss: 0.0666, Accuracy: 97.5559, Valid Loss: 0.1871, Valid Accuracy: 93.5547
  Epoch 18, Loss: 0.0622, Accuracy: 97.7168, Valid Loss: 0.1976, Valid Accuracy: 94.5312
  Epoch 19, Loss: 0.0579, Accuracy: 97.8577, Valid Loss: 0.2086, Valid Accuracy: 93.9453
  Epoch 20, Loss: 0.0533, Accuracy: 98.1013, Valid Loss: 0.2278, Valid Accuracy: 93.7500
  Elapsed time in (ms): 71537.03

The ``for`` loop that marches though the input (training or test datasets) can be implemented
using other methods too. For example, one can make use of Python iterator functions
``iter`` and ``next``. Using iterator we have more control over the number of steps we wish to
execute the commands. Another way of implementing could be using ``for`` inside ``tf.function``.

ParameterServerStrategy
-----------------------

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
---------------------------

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

.. exercise:: Which one is faster?

   Comment out the ``EarlyStopping`` callback, fix the number of epochs to ``20`` and 
   train the model using ``Model.fit`` API:
   1. On 4 GPUs using ``MirroredStrategy``
   2. On a single GPU using pinning method
   and compare the elapsed time with the number you obtained above.

   Which of these methods faster? Do you have any explanation for that?
   

.. exercise:: Evaluation for a custom training
  
   Evaluate the performance of the metrics on the tests datasets for custom training loop.

   .. solution::

     .. code-block:: python

        eval_accuracy = tf.metrics.BinaryAccuracy(name='eval_accuracy')     
        @tf.function
        def eval_step(texts, labels):
            predictions = model(texts, training=False)
            eval_accuracy(labels, predictions)
            eval_accuracy.update_state(labels, predictions)     
        
        for texts, labels in valid_dataset:
            eval_step(images, labels)
        
        print ('The model accuracy : {:5.2f}%'.format(eval_accuracy.result()*100))        

.. exercise:: (Advanced) Custom training loop for SVHN

   Use the ``SVHN_class`` code provided in :doc:`/tf_intro` and write a custom training loop using 
   ``MirroredStrategy``.
   