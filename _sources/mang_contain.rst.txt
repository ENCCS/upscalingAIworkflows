.. _mang_contain:

Cleaning Up Containers
======================

Removing images
_______________

The images and their corresponding containers can start to take up a
lot of disk space if you don't clean them up occasionally, so it's a
good idea to periodically remove container images that you won't be
using anymore.

In order to remove a specific image, you need to find out details
about the image, specifically, the "image ID". For example say my
laptop contained the following image.

.. code-block:: bash

   docker image ls

.. code-block:: text

   REPOSITORY       TAG         IMAGE ID       CREATED          SIZE
   hello-world      latest      fce289e99eb9   15 months ago    1.84kB


You can remove the image with a `docker image rm` command that
includes the image ID, such as:

.. code-block:: bash

   docker image rm fce289e99eb9

or use the image name, like so:

.. code-block:: bash

   docker image rm hello-world

However, you may see this output:

.. code-block:: text

   Error response from daemon: conflict: unable to remove repository
   reference "hello-world" (must force) - container e7d3b76b00f4 is
   using its referenced image fce289e99eb9

This happens when Docker hasn't cleaned up some of the times when a
container has been actually run. So before removing the container
image, we need to be able to see what containers are currently
running, or have been run recently, and how to remove these.

What containers are running?
----------------------------

Working with containers, we are going to shift to a new docker
command: `docker container`.  Similar to `docker image`, we can list
running containers by typing:

.. code-block:: bash

   docker container ls

.. code-block:: text

   CONTAINER ID        IMAGE               COMMAND             CREATED             STATUS              PORTS               NAMES

Notice that this command didn't return any containers because our
containers all exited and thus stopped running after they completed
their work.

.. callout:: `docker ps`

   The command `docker ps` serves the same purpose as `docker container
   ls`, and comes from the Unix shell command `ps` which describes
   running processes.

What containers have run recently?
__________________________________

There is also a way to list running containers, and those that have
completed recently, which is to add the `--all`/`-a` flag to the
`docker container ls` command as shown below.

.. code-block:: bash

   docker container ls --all

.. code-block:: text

   CONTAINER ID        IMAGE               COMMAND             CREATED             STATUS                     PORTS               NAMES
   9c698655416a        hello-world         "/hello"            2 minutes ago       Exited (0) 2 minutes ago                       zen_dubinsky
   6dd822cf6ca9        hello-world         "/hello"            3 minutes ago       Exited (0) 3 minutes ago                       eager_engelbart


.. callout:: Keeping it clean

   You might be surprised at the number of containers Docker is still
   keeping track of.  One way to prevent this from happening is to add
   the `--rm` flag to `docker run`. This will completely wipe out the
   record of the run container when it exits. If you need a reference
   to the running container for any reason, **don't** use this flag.

How do I remove an exited container?
____________________________________

To delete an exited container you can run the following command,
inserting the `CONTAINER ID` for the container you wish to remove.  It
will repeat the `CONTAINER ID` back to you, if successful.

.. code-block:: bash

   docker container rm 9c698655416a

output:

.. code-block:: text

   9c698655416a

If you want to remove all exited containers at once you can use the
`docker containers prune` command.  **Be careful** with this command.
If you have containers you may want to reconnect to, you should not
use this command.  It will ask you if to confirm you want to remove
these containers, see output below.  If successful it will print the
full `CONTAINER ID` back to you.

.. code-block:: bash

   docker container prune

Output:

.. code-block:: text

   WARNING! This will remove all stopped containers.
   Are you sure you want to continue? [y/N] y
   Deleted Containers:
   9c698655416a848278d16bb1352b97e72b7ea85884bff8f106877afe0210acfc
   6dd822cf6ca92f3040eaecbd26ad2af63595f30bb7e7a20eacf4554f6ccc9b2b


Removing images
_______________

Now that we've removed any potentially running or stopped containers,
we can try again to delete the `hello-world` **image**.

.. code-block:: bash

   docker image rm hello-world

output

.. code-block:: text

   Untagged: hello-world:latest
   Untagged: hello-world@sha256:5f179596a7335398b805f036f7e8561b6f0e32cd30a32f5e19d17a3cda6cc33d
   Deleted: sha256:fce289e99eb9bca977dae136fbe2a82b6b7d4c372474c9235adc1741675f587e
   Deleted: sha256:af0b15c8625bb1938f1d7b17081031f649fd14e6b233688eea3c5483994a66a3

The reason that there are a few lines of output, is that a given image
may have been formed by merging multiple underlying layers.  Any
layers that are used by multiple Docker images will only be stored
once.  Now the result of `docker image ls` should no longer include
the `hello-world` image.
