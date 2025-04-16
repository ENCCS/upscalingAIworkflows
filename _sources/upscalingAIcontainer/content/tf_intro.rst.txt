.. _tf_intro:

.. figure :: https://www.gstatic.com/devrel-devsite/prod/v93a6dcf50ad5e38e51034415df5b4a8345b5c8613f785e48818ae468dabf73c8/tensorflow/images/lockup.svg
  :width: 40%

  `(Image Source) <https://www.tensorflow.org>`_

TensorFlow on a single GPU
==========================

TensorFlow is a well-known library developed primarily in Google which has been
proven to be one of the most robust, reilable, and fast libraries for deep learning
among developers. I think most of us have had some form of exposure to TensorFlow
at some point in our deep learning/machin learning journery.

In this section we focus on using a single GPU for training our model. It is rather
easy to transfer/port traning of the model to the GPU with minimal coding.

TensorFlow supports running computations on a variety of types of devices, including
CPU and GPU. They are represented with string identifiers for example:

  - ``/device:CPU:0``: The CPU of your machine.
  - ``/GPU:0``: Short-hand notation for the first GPU of your machine that is
    visible to TensorFlow.
  - ``/job:localhost/replica:0/task:0/device:GPU:1``: Fully qualified name of
    the second GPU of your machine that is visible to TensorFlow.

If a TensorFlow operation has both CPU and GPU implementations, by default,
the GPU device is prioritized when the operation is assigned. For example, ``tf.matmul``
has both CPU and GPU kernels and on a system with devices ``CPU:0`` and ``GPU:0``,
the ``GPU:0`` device is selected to run ``tf.matmul`` unless you explicitly request
to run it on another device.

If a TensorFlow operation has no corresponding GPU implementation, then the operation
falls back to the CPU device. For example, since ``tf.cast`` only has a CPU kernel,
on a system with devices ``CPU:0`` and ``GPU:0``, the ``CPU:0`` device is selected
to run ``tf.cast``, even if requested to run on the ``GPU:0`` device.

Get the physical devices
________________________

After booking a node with multiple GPUs, let's check if we have TensorFlow module
loaded and if the physical GPU device is available.

.. code-block:: python

  import tensorflow as tf
  print("Num of GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
  print("TensorFlow version: ", tf.__version__)

.. code-block:: bash

  Num of GPUs Available:  6
  TensorFlow version:  2.5.0

We can see the list of all of available devices:

.. code-block:: python

  tf.config.list_physical_devices()

.. code-block :: bash

  [PhysicalDevice(name='/physical_device:CPU:0', device_type='CPU'),
  PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU'),
  PhysicalDevice(name='/physical_device:GPU:1', device_type='GPU'),
  PhysicalDevice(name='/physical_device:GPU:2', device_type='GPU'),
  PhysicalDevice(name='/physical_device:GPU:3', device_type='GPU'),
  PhysicalDevice(name='/physical_device:GPU:4', device_type='GPU'),
  PhysicalDevice(name='/physical_device:GPU:5', device_type='GPU')]

If you have GPUs, then you should see the GPU device in the above list.
We can also check specifically for the GPU or CPU devices.

.. code-block:: python

  tf.config.list_physical_devices('GPU')
  tf.config.list_physical_devices('CPU')

Placement of calculations
_________________________

TensorFlow automatically place tensor operations to physical devices which is by
default is the GPU if available. Now, let's define a random Tensor, and check where
it is placed.

.. code-block:: python

  x = tf.random.uniform([3, 3])
  x.device

.. code-block :: bash

  '/job:localhost/replica:0/task:0/device:GPU:0'

The above string will end with ``GPU:K`` if the Tensor is placed on the K-th GPU device.
We can also check if a tensor is placed on a specific device by using ``device_endswith``:

.. code-block :: python

  print("Is the Tensor on CPU #0:  "),
  print(x.device.endswith('CPU:0'))
  print('')
  print("Is the Tensor on GPU #0:  "),
  print(x.device.endswith('GPU:0'))

.. code-block :: bash

  Is the Tensor on CPU #0:
  False

  Is the Tensor on GPU #0:
  True

Determining the Placement
_________________________

It is possible to force placement on specific devices, if they are available. We can view
the benefits of GPU acceleration by running some tests and placing the operations on
the CPU or GPU respectively.

.. code-block :: python

  import time
  def time_matadd(x):
    start = time.time()
    for loop in range(10):
        tf.add(x, x)
    result = time.time()-start
    print("Matrix addition (10 loops): {:0.2f}ms".format(1000*result))

  def time_matmul(x):
    start = time.time()
    for loop in range(10):
        tf.matmul(x, x)
    result = time.time()-start
    print("Matrix multiplication (10 loops): {:0.2f}ms".format(1000*result))

We run the above tests first on a CPU using ``tf.device("CPU:0")``,
which forces the operations to be run on the CPU.

.. code-block :: python

  print("On CPU:")
  with tf.device("CPU:0"):
    x = tf.random.uniform([1000, 1000])
    assert x.device.endswith("CPU:0")
    time_matadd(x)
    time_matmul(x)

.. code-block :: bash

  On CPU:
  Matrix addition (10 loops): 3.51ms
  Matrix multiplication (10 loops): 199.40ms

And doing the same operations on the GPU:

.. code-block :: python

  if tf.config.experimental.list_physical_devices("GPU"):
    print("On GPU:")
    with tf.device("GPU:0"):
      x = tf.random.uniform([1000, 1000])
      assert x.device.endswith("GPU:0")
      time_matadd(x)
      time_matmul(x)

.. code-block :: bash

  On GPU:
  Matrix addition (10 loops): 0.89ms
  Matrix multiplication (10 loops): 22.64ms

Note the significant time difference between running these operations on different devices.

Logging device placement
________________________

We can find out which devices your operations and tensors are assigned to by putting
``tf.debugging.set_log_device_placement(True)`` as the first statement of your program.
Enabling device placement logging causes any Tensor allocations or operations to be printed.

.. exercise :: Training on CPU and GPU

  You can find two neural networks for image classifier for the `The Street View
  House Numbers (SVHN)` dataset in the github :download:`SVHN notebook  <code/SVHN_class.ipynb>`.
  Try to train the model on CPU and GPU. Compare the results.

  Can you place manually some parts on GPU and some on CPU?
