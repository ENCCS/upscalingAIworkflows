.. _setup:

Access to Vega
===============

You should have received your username from the Vega support with the help of our colleagues at
ENCCS. One important step to login to the system is to generate an SSH key and upload your public
to the server. You can read it in `the Vega documentation <https://doc.vega.izum.si/ssh/>`_.

Once the setup is completed, you can login to the system. While the ``Singularity``
module is readily available in the login node, we should book a compute node to run
our examples. You can book a CPU-node for 1 hour using this command

.. code-block:: bash

  salloc -n 1 -t 1:00:00

And for a GPU-node with *X* number of GPUs per node (there are 4 GPUs available per node in Vega)

.. code-block:: bash

   salloc -n 1 --gres=gpu:X --partition=gpu --mem-per-gpu=40GB --reservation enccs \
   --ntasks 4 --cpus-per-task 1  -t 02:00:00

Once the allocation is granted you will receive a message similar to

.. code-block:: text

  salloc: Pending job allocation 24122556
  salloc: job 24122556 queued and waiting for resources
  salloc: job 24122556 has been allocated resources
  salloc: Granted job allocation 24122556
  salloc: Waiting for resource configuration
  salloc: Nodes cn0381 are ready for job

The granted CPU compute node here is `cn0381`. The general form for the CPU compute node likes
like `cn0XXX` for GPU compute node `gnXXX`. Now you should SSH to the compute node to run interactively
our job using the command

.. code-block:: bash

  ssh cn0381

You might get a warning regarding the authenticity of the host, similar to the
output below.

.. code-block:: text

  The authenticity of host 'cn0381 (<no hostip for proxy command>)' can't be established.
  ECDSA key fingerprint is SHA256:0BOlvbjVPLytjYEium04uNTCACCQN/Rr7NMJhje30aw.
  Are you sure you want to continue connecting (yes/no/[fingerprint])? yes
  Warning: Permanently added 'cn0381' (ECDSA) to the list of known hosts.

Please enter ``yes`` to the question and ignore it as it is a self-assigned useless check
that doesn't understand what is the purpose of our login. Now, we are able to run
our jobs interactively.

To run Jupyter Notebooks, we need to load Anaconda module.

.. code-block:: bash

   module load Anaconda3/2020.11
   module load scikit-learn

Afterwards, you can run a Jupyter kernel by specifying the port number and ip address.
The ip address here is the name of compute node, in the example given above is `cn0381`.

.. code-block:: bash

   jupyter-notebook --no-browser --port=8888 --ip=cn0381

The result should look like

.. code-block:: text

  [I 13:21:26.105 NotebookApp] JupyterLab extension loaded from /cvmfs/sling.si/modules/el7/software/Anaconda3/2020.11/lib/python3.8/site-packages/jupyterlab
  [I 13:21:26.105 NotebookApp] JupyterLab application directory is /cvmfs/sling.si/modules/el7/software/Anaconda3/2020.11/share/jupyter/lab
  [I 13:21:26.107 NotebookApp] Serving notebooks from local directory: /ceph/hpc/home/euhosseine
  [I 13:21:26.107 NotebookApp] Jupyter Notebook 6.1.4 is running at:
  [I 13:21:26.107 NotebookApp] http://cn0381:8888/?token=80d695595aa333c6d97dc6f868f96b36f4812622a5008090
  [I 13:21:26.107 NotebookApp]  or http://127.0.0.1:8888/?token=80d695595aa333c6d97dc6f868f96b36f4812622a5008090
  [I 13:21:26.107 NotebookApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
  [C 13:21:26.122 NotebookApp]

    To access the notebook, open this file in a browser:
        file:///ceph/hpc/home/euhosseine/.local/share/jupyter/runtime/nbserver-339349-open.html
    Or copy and paste one of these URLs:
        http://cn0381:8888/?token=80d695595aa333c6d97dc6f868f96b36f4812622a5008090
     or http://127.0.0.1:8888/?token=80d695595aa333c6d97dc6f868f96b36f4812622a5008090

In your local machine (PC/laptop), open a terminal and use this command to tunnel to the running kernel 
on Vega.

.. code-block:: bash

   ssh -N -f -L 8888:cn0381:8888 username@vglogin0005.vega.izum.si

The first port number is for your local machine and the second port number is what
you specified above running a Jupyter Notebook. Open a browser, and enter ``http://localhost:8888``.
You should see a prompt to enter the password or the token. The token in this run is the number after 
the ``token``. Entering the token, you will be shown the tree of structure of home folder.

To use TensorFlow or Horovod in this course, we can simply load them through module system.

.. code-block:: bash

   module load TensorFlow/2.5.0-fosscuda-2020b


Or

.. code-block:: bash

   module load Horovod/0.22.1-fosscuda-2020b-TensorFlow-2.5.0
