Test episode
============


.. code-block:: python


   def lr_schedule(epoch):
       
       if epoch < 15:
           return args.base_lr
       if epoch < 25:
           return 1e-1 * args.base_lr
       if epoch < 35:
           return 1e-2 * args.base_lr
       return 1e-3 * args.base_lr
   
   ##### Steps
   # Step 1: import Horovod
   import horovod.tensorflow.keras as hvd
   hvd.init()
   
   # Step 2: pin to a GPU
   gpus = tf.config.experimental.list_physical_devices('GPU')
   for gpu in gpus:
       tf.config.experimental.set_memory_growth(gpu, True)
   if gpus:
       tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
       
   # Step 3: only set `verbose` to `1` if this is the root worker.
   if hvd.rank() == 0:
       verbose = 1
   else:
       verbose = 0
   #####

