.. _intro_docker:

Introduction to Docker
======================

There are many container runtime available out there. One of popular runtimes is Docker.
Docker has proven to an extraordinary tool for developers and researchers alike.
Most of today's workshop will be spent of Docker command line (CLI) utility.

Downloading Docker images
-------------------------

The ``docker image`` command is used to list and modify Docker images.
You can find out what container images you have on your computer by
using the following command (“ls” is short for “list”):

.. code-block:: bash

  docker image ls

If you’ve just installed Docker, you won’t see any images listed.

To get a copy of the ``hello-world`` Docker image from the internet, run
this command:

.. code-block:: bash

  docker pull hello-world

You should see output like this:

.. code-block:: text

   Using default tag: latest
   latest: Pulling from library/hello-world
   93288797bd35: Pull complete
   Digest: sha256:975f4b14f326b05db86e16de00144f9c12257553bba9484fed41f9b6f2257800
   Status: Downloaded newer image for hello-world:latest
   docker.io/library/hello-world:latest

.. callout:: DockerHub

   Where did the ``hello-world`` image come from? It came from the
   DockerHub website, which is a place to share Docker images with other
   people. More on that in a later episode.

.. exercise:: Exercise: Check on Your Images

   What command would you use to see if the ``hello-world`` Docker image
   had downloaded successfully and was on your computer? Give it a try
   before checking the solution.

   .. solution::

      To see if the ``hello-world`` image is now on your computer, run:

      .. code-block:: bash

         docker image ls

Note that the downloaded ``hello-world`` image is not in the folder
where you are in the terminal! (Run ``ls`` by itself to check.) The
image is not a file like our normal programs and files; Docker stores it
in a specific location that isn’t commonly accessed, so it’s necessary
to use the special ``docker image`` command to see what Docker images
you have on your computer.

Running the ``hello-world`` container
-------------------------------------

To create and run containers from named Docker images you use the
``docker run`` command. Try the following ``docker run`` invocation.
Note that it does not matter what your current working directory is.

.. code-block:: bash

   docker run hello-world

   Hello from Docker!
   This message shows that your installation appears to be working correctly.

To generate this message, Docker took the following steps:

1. The Docker client contacted the Docker daemon.
2. The Docker daemon pulled the "hello-world" image from the Docker Hub.
   (arm64v8)
3. The Docker daemon created a new container from that image which runs the
   executable that produces the output you are currently reading.
4. The Docker daemon streamed that output to the Docker client,
   which sent it to your terminal.

The ``hello-world`` container is set up to run an action by default -
namely to print this message.

.. callout:: Using ``docker run`` to get the image

   We could have skipped the ``docker pull`` step; if you use the
   ``docker run`` command and you don’t already have a copy of the
   Docker image, Docker will automatically pull the image first and then
   run it.

Running a container with a chosen command
-----------------------------------------

But what if we wanted to do something different with the container? The
output just gave us a suggestion of what to do – let’s use a different
Docker image to explore what else we can do with the ``docker run``
command. The suggestion above is to use ``ubuntu``, but we’re going to
run a different type of Linux, ``alpine`` instead because it’s quicker
to download.

.. callout:: Run the Alpine Docker container

   Try downloading and running the ``alpine`` Docker container. You can
   do it in two steps, or one. What are they?

What happened when you ran the Alpine Docker container?

.. code-block:: bash

   docker run alpine

If you never used the *alpine* docker image on your computer, docker
probably printed a message that it couldn’t find the image and had to
download it. If you used the alpine image before, the command will
probably show no output. That’s because this particular container is
designed for you to provide commands yourself. Try running this instead:

.. code-block:: bash

  docker run alpine cat /etc/os-release

You should see the output of the ``cat /etc/os-release`` command, which
prints out the version of Alpine Linux that this container is using and
a few additional bits of information.

.. exercise:: Exercise: Hello World, Part 2

   Can you run the container and make it print a “hello world” message?
   Give it a try before checking the solution.

   .. solution::

      Use the same command as above, but with the ``echo`` command to
      print a message.

      .. code-block:: bash

         docker run alpine echo ‘Hello World’

So here, we see another option – we can provide commands at the end of
the ``docker run`` command and they will execute inside the running
container.

Running containers interactively
--------------------------------

In all the examples above, Docker has started the container, run a
command, and then immediately shut down the container. But what if we
wanted to keep the container running so we could log into it and test
drive more commands? The way to do this is by adding the interactive
flag ``-it`` to the ``docker run`` command and by providing a shell
(usually ``bash`` or ``sh``) as our command. The alpine docker image
doesn’t include ``bash`` so we need to use ``sh``.

.. code-block:: bash

   docker run -it alpine sh

Your prompt should change significantly to look like this:

.. code-block:: bash

   / #

That’s because you’re now inside the running container! Try these
commands:

-  ``pwd``
-  ``ls``
-  ``whoami``
-  ``echo $PATH``
-  ``cat /etc/os-release``

All of these are being run from inside the running container, so you’ll
get information about the container itself, instead of your computer. To
finish using the container, just type ``exit``.

.. callout:: Reminder of terminology: images and containers

   Recall that a container “image” is the template from which particular
   instances of containers will be created.

Let’s explore our first Docker container. The Docker team provides a
simple container image online called ``hello-world``. We’ll start with
that one.
