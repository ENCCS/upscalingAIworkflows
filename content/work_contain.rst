.. _work_contain:

Working with Singularity containers
===================================

Singularity's image cache
_________________________

While Singularity doesn't have a local image repository in the same
way as Docker, it does cache downloaded image files. As we saw in the
previous episode, images are simply `.sif` files stored on your local
disk.

If you delete a local `.sif` image that you have pulled from a remote
image repository and then pull it again, if the image is unchanged
from the version you previously pulled, you will be given a copy of
the image file from your local cache rather than the image being
downloaded again from the remote source. This removes unnecessary
network transfers and is particularly useful for large images which
may take some time to transfer over the network.  To demonstrate this,
remove the `hello-world.sif` file stored in your `test` directory and
then issue the `pull` command again:

.. code-block:: bash

   rm hello-world.sif
   singularity pull hello-world.sif shub://vsoch/hello-world

Output

.. code-block:: bash

   INFO:    Use image from cache

As we can see in the above output, the image has been returned from
the cache and we don't see the output that we saw previously showing
the image being downloaded from Singularity Hub.

How do we know what is stored in the local cache? We can find out
using the `singularity cache` command:

.. code-block:: bash

   singularity cache list

.. code-block:: bash

   There are 1 container file(s) using 62.65 MB and 0 oci blob file(s)
   using 0.00 kB of space Total space used: 62.65 MB

This tells us how many container files are stored in the cache and how
much disk space the cache is using but it doesn't tell us **what** is
actually being stored. To find out more information we can add the
`-v` verbose flag to the `list` command:

.. code-block:: bash

   singularity cache list -v

Output

.. code-block:: bash

   NAME                     DATE CREATED           SIZE             TYPE
   hello-world_latest.sif   2020-04-03 13:20:44    62.65 MB         shub

   There are 1 container file(s) using 62.65 MB and 0 oci blob file(s) using 0.00 kB of space
   Total space used: 62.65 MB

This provides us with some more useful information about the actual
images stored in the cache.  In the `TYPE` column we can see that our
image type is `shub` because it's a `SIF` image that has been pulled
from Singularity Hub.

.. callout:: Cleaning the Singularity image cache

   We can remove images from the cache using the `singularity cache
   clean` command.  Running the command without any options will
   display a warning and ask you to confirm that you want to remove
   everything from your cache. You can also remove specific images or
   all images of a particular type. Look at the output of `singularity
   cache clean --help` for more information.

.. exercise:: Basic exercise: Clearing specific image types from the cache

   What command would you use to remove only images of type
   **shub** from your local Singularity image cache?  How could you
   test this safely to ensure your command is going to do the right
   thing?

   .. tip:: You can see exactly what would be deleted by using the ``--dry-run`` option or ``-n``.

   .. solution::

      .. code-block:: bash

	singularity cache clean --type=shub
	singularity cache clean -n --type=shub

      .. code-block:: bash

	 Removing /<cache_dir>/.singularity/cache/shub


Working with containers
-----------------------

Running specific commands within a container
++++++++++++++++++++++++++++++++++++++++++++

We saw earlier that we can use the `singularity inspect` command to
see the run script that a container is configured to run by
default. What if we want to run a different command within a
container, or we want to open a shell within a container that we can
interact with?

If we know the path of an executable that we want to run within a
container, we can use the `singularity exec` command. For example,
using the `hello-world.sif` container that we've already pulled from
Singularity Hub, we can run the following within the `test` directory
where the `hello-world.sif` file is located:

.. code-block:: bash

   singularity exec hello-world.sif /bin/echo Hello World!

Output

.. code-block:: bash

   Hello World!

Here we see that a container has been started from the
`hello-world.sif` image and the `/bin/echo` command has been run
within the container, passing the input `Hello World!`.  The command
has echoed the provided input to the console and the container has
terminated.

.. exercise:: Basic exercise: Running a different command within the
              "hello-world" container

   Can you run a container based on the `hello-world.sif` image that
   **prints the current date and time**?

   .. solution::

      .. code-block:: bash

	 singularity exec hello-world.sif /bin/date

      Output

      .. code-block:: bash

	 Fri Jun 26 15:17:44 BST 2020

Running a shell within a container
++++++++++++++++++++++++++++++++++

If you want to open an interactive shell within a container,
Singularity provides the `singularity shell` command. Again, using the
`hello-world.sif` image, and within our `test` directory, we can run a
shell within a container from the hello-world image:

.. code-block:: bash

  singularity shell hello-world.sif

Output

.. code-block:: bash

   Singularity> whoami
   [<your username>]
   Singularity> ls
   hello-world.sif
   Singularity>

As shown above, we have opened a shell in a new container started from
the `hello-world.sif` image.

.. callout:: Running a shell inside a Singularity container

   Q: What do you notice about the output of the above commands
   entered within the Singularity container shell?

   Q: Does this differ from what you might see within a Docker container?

Use the `exit` command to exit from the container shell.

Users, files and directories within a Singularity container
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

The first thing to note is that when you run `whoami` within the
container you should see the username that you are signed in as on the
host system when you run the container.  For example, if my username
is `jc1000`:

.. code-block:: bash

   singularity shell hello-world.sif

   Singularity> whoami
   jc1000


But hang on! I downloaded the standard, public version of the
`hello-world.sif` image from Singularity Hub. I haven't customised it
in any way. How is it configured with my own user details?!

If you have any familiarity with Linux system administration, you may
be aware that in Linux, users and their Unix groups are configured in
the `/etc/passwd` and `/etc/group` files respectively.  In order for
the shell within the container to know of my user, the relevant user
information needs to be available within these files within the
container.

Assuming this feature is enabled on your system, when the container is
started, Singularity appends the relevant user and group lines from
the host system to the `/etc/passwd` and `/etc/group` files within the
container `[1]
<https://www.intel.com/content/dam/www/public/us/en/documents/presentation/hpc-containers-singularity-advanced.pdf>`_.

Singularity also **binds** some directories from the host system where
you are running the `singularity` command into the container that
you're starting. Note that this bind process isn't copying files into
the running container, it is simply making an existing directory on
the host system visible and accessible within the container
environment. If you write files to this directory within the running
container, when the container shuts down, those changes will persist
in the relevant location on the host system.

There is a default configuration of which files and directories are
bound into the container but ultimate control of how things are set up
on the system where you're running Singularity is determined by the
system administrator.  As a result, this section provides an overview
but you may find that things are a little different on the system that
you're running on.

One directory that is likely to be accessible within a container that
you start is your **home directory**.  The mapping of file content and
directories from a host system into a Singularity container is
illustrated in the example below showing a subset of the directories
on the host Linux system and in a Singularity container:

.. code-block:: text

   Host system:                                                      Singularity container:
   -------------                                                     ----------------------
   /                                                                 /
   ├── bin                                                           ├── bin
   ├── etc                                                           ├── etc
   │   ├── ...                                                       │   ├── ...
   │   ├── group  ─> user's group added to group file in container ─>│   ├── group
   │   └── passwd ──> user info added to passwd file in container ──>│   └── passwd
   ├── home                                                          ├── usr
   │   └── jc1000 ───> user home directory made available ──> ─┐     ├── sbin
   ├── usr                 in container via bind mount         │     ├── home
   ├── sbin                                                    └────────>└── jc1000
   └── ...                                                           └── ...

.. exercise:: Questions and exercises: Files in Singularity containers

   **Q1:** What do you notice about the ownership of files in a
   container started from the hello-world image?  (e.g. take a look
   at the ownership of files in the root directory (`/`))

   **Exercise 1:** In this container, try editing (for example
   using the editor `vi` which should be avaiable in the container)
   the `/rawr.sh` file. What do you notice?

   If you're not familiar with `vi` there are many quick
   reference pages online showing  the main commands for using the editor,
   for example `this one <http://web.mit.edu/merolish/Public/vi-ref.pdf>`_

   **Exercise 2:** In your home directory within the container
   shell, try and create a simple text file. Is it possible to
   do this? If so, why? If not, why not?! If you can  successfully create
   a file, what happens to it when you exit the shell and the container
   shuts down?

   .. solution::

      **A1:** Use the `ls -l` command to see a detailed file
      listing including file ownership and permission details.  You may
      see that all the files are owned by you, alternatively, most files
      in the root (`/`) directory may be owned by the `root` user. If
      the files are owned by you, this looks good - you should be ready
      to edit something in the exercise that follows...otherwise, if the
      files are owned by root, maybe not...

      **Exercise 1:** Unfortunately, it's not so easy, depending on how
      you tried to edit `/rawr.sh` you probably saw an error similar to
      the following: `Can't open file for writing` or `Read-only file
      system`

      **Exercise 2:** Within your home directory, you **should** be able
      to successfully create a file. Since you're seeing your home
      directory on the host system which has been bound into the
      container, when you exit and the container shuts down, the file
      that you created within the container should still be present when
      you look at your home directory on the host system.

Using Docker images with Singularity
____________________________________

Singularity can also start containers from Docker images, opening up
access to a huge number of existing container images available on
`Docker Hub <https://hub.docker.com/>`_ and other registries.

While Singularity doesn't support running Docker images directly, it
can pull them from Docker Hub and convert them into a suitable format
for running via Singularity. When you pull a Docker image, Singularity
pulls the slices or **layers** that make up the Docker image and
converts them into a single-file Singularity SIF image.

For example, moving on from the simple **Hello World** examples that
we've looked at so far, let's pull one of the `official Docker Python
images <https://hub.docker.com/_/python>`_. We'll use the image with
the tag `3.8.6-slim-buster` which has Python 3.8.6 installed on
Debian's `Buster <https://www.debian.org/releases/buster/>`_ (v10)
Linux distribution:

.. code-block:: bash

   singularity pull python-3.8.6.sif docker://python:3.8.6-slim-buster

Output

.. code-block:: text

   INFO:    Converting OCI blobs to SIF format
   INFO:    Starting build...
   Getting image source signatures
   Copying blob 852e50cd189d done
   Copying blob 334ed303e4ad done
   Copying blob a687a65725ea done
   Copying blob fe607cb30fbe done
   Copying blob b8a3bc0a3645 done
   Copying config 08d8e312de done
   Writing manifest to image destination
   Storing signatures
   2020/12/07 18:36:18  info unpack layer: sha256:852e50cd189dfeb54d97680d9fa6bed21a6d7d18cfb56d6abfe2de9d7f173795
   2020/12/07 18:36:19  info unpack layer: sha256:334ed303e4ad2f8dc872f2e845d79012ad648eaced444e009ae9a397cc4b4dbb
   2020/12/07 18:36:19  info unpack layer: sha256:a687a65725ea883366a61d24db0f946ad384aea893297d9510e50fa13f565539
   2020/12/07 18:36:19  info unpack layer: sha256:fe607cb30fbe1148b5885d58c909d0c08cbf2c0848cc871845112f3ee0a0f9ba
   2020/12/07 18:36:19  info unpack layer: sha256:b8a3bc0a3645e2afcd8807830833a0df0bd243d58d518e17b2335342e2614bd3
   INFO:    Creating SIF file...
   INFO:    Build complete: python-3.8.6.sif

Note how we see singularity saying that it's "**Converting OCI blobs
to SIF format**". We then see the layers of the Docker image being
downloaded and unpacked and written into a single SIF file. Once the
process is complete, we should see the python-3.8.6.sif image file in
the current directory.

We can now run a container from this image as we would with any other
singularity image.

.. exercise:: Running the Python 3.8.6 image that we just pulled from Docker Hub

   - **E1**: Try running the Python 3.8.6 image. What happens?
   - **E2**: Try running some simple Python statements...

   .. solution::

      Running the Python 3.8.6 image

      .. code-block:: bash

        singularity run python-3.8.6.sif

      This should put you straight into a Python interactive shell
      within the running container:

      .. code-block:: bash

        Python 3.8.6 (default, Nov 25 2020, 02:47:44)
	      [GCC 8.3.0] on linux
	      Type "help", "copyright", "credits" or "license" for more information.
	      >>>

      Now try running some simple Python statements:

      .. code-block:: bash

        >>> import math
	      >>> math.pi
	      3.141592653589793
	      >>>

In addition to running a container and having it run the default run
script, you could also start a container running a shell in case you
want to undertake any configuration prior to running Python. This is
covered in the following exercise:

.. exercise:: Open a shell within a Python container

   Try to run a shell within a singularity container based on
   the `python-3.8.6.sif` image.  That is, run a container that
   opens a shell rather than the default Python interactive console
   as we saw above. Can you find more than one way to achieve this?

   Within the shell, try starting the Python interactive console
   and running some Python commands.

   .. solution::

      Recall from the earlier material that we can use the
      `singularity shell` command to open a shell within a container.
      To open a regular shell within a container based on the
      `python-3.8.6.sif` image, we can therefore simply run:

      .. code-block::

        singularity shell python-3.8.6.sif

	      Singularity> echo $SHELL
	      /bin/bash

	      Singularity> cat /etc/issue
	      Debian GNU/Linux 10 \n \l

        Singularity> exit

      It is also possible to use the `singularity exec` command to
      run an executable within a container.  We could, therefore, use
      the `exec` command to run `/bin/bash`:

      .. code-block:: bash

        singularity exec python-3.8.6.sif /bin/bash

        Singularity> echo $SHELL
	      /bin/bash

      You can run the Python console from your container shell
      simply by running the `python` command.

This concludes the second episode and Part I of the Singularity
material. Part II contains a further two episodes where we'll look
creating your own images and then more advanced use of containers for
running MPI parallel applications.
