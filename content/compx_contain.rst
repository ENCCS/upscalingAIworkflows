.. _compx_contain:

Creating More Complex Container Images
======================================

In order to create and use your own containers, you may need more
information than our previous example. You may want to use files from
outside the container, copy those files into the container, and just
generally learn a little bit about software installation. This episode
will cover these. Note that the examples will get gradually more and
more complex - most day-to-day use of containers can be accomplished
using the first 1-2 sections on this page.

Using scripts and files from outside the container
--------------------------------------------------

Let's create a file and folder called it ``foo/dummy.py`` in the root
folder.

Please copy the ``Dockerfile`` and place it in the ``foo`` directory.
Let’s say we wanted to try running the script using our recently
created ``ubuntu-python`` container.

.. callout:: Running containers

    What command would we use to run python from the ``alpine-python``
    container?

If we try running the container and Python script, what happens?

.. code-block:: bash

   docker run alice/ubuntu-python python3 dummy.py

Output

.. code-block:: text

   python3: can’t open file ‘dummy.py’: [Errno 2] No such file or directory

.. callout:: No such file or directory

   What does the error message mean? Why might the Python inside the
   container not be able to find or open our script?

The problem here is that the container and its file system is separate
from our host computer’s file system. When the container runs, it can’t
see anything outside itself, including any of the files on our computer.
In order to use Python (inside the container) and our script (outside
the container, on our computer), we need to create a link between the
directory on our computer and the container.

This link is called a “mount” and is what happens automatically when a
USB drive or other external hard drive gets connected to a computer -
you can see the contents appear as if they were on your computer.

We can create a mount between our computer and the running container by
using an additional option to ``docker run``. We’ll also use the
variable ``$PWD`` which will substitute in our current working
directory. The option will look like this

.. code-block:: bash

  -v $PWD:/temp

What this means is – link my current directory with the container, and
inside the container, name the directory ``/temp``

Let’s try running the command now:

.. code-block:: bash

   docker run -v $PWD:/temp alice/ubuntu-python python3 dummy.py

But we get the same error!

.. code-block:: text

   python3: can't open file 'dummy.py': [Errno 2] No such file or directory

This final piece is a bit tricky -- we really have to remember to put
ourselves inside the container. Where is the `dummy.py` file? It's in
the directory that's been mapped to `/temp` -- so we need to include
that in the path to the script. This command should give us what we
need:

.. code-block:: bash

   docker run -v $PWD:/temp alice/alpine-python python3 /temp/dummy.py

Note that if we create any files in the `/temp` directory while the
container is running, these files will appear on our host filesystem
in the original directory and will stay there even when the container
stops.

.. exercise:: Checking the options, Interactive jobs

   1. Can you go through each piece of the Docker command above
      the explain what it does? How would you characterize the
      key components of a Docker command?

   2. Try using the directory mount option but run the container
      interactively. Can you find the folder that's connected to
      your computer? What's inside?

   .. solution::

      1. Here's a breakdown of each piece of the command above

	 - `docker run`: use Docker to run a container
	 - `-v $PWD:/temp`: connect my current working directory
	   (`$PWD`) as a folder inside the container called `/temp`
	 - `alice/ubuntu-python`: name of the container to run
	 - `python3 /temp/dummy.py`: what commands to run in the container

	 More generally, every Docker command will have the form:
	 `docker [action] [docker options] [docker image] [command
	 to run inside]`

      2. The docker command to run the container interactively is:

	 .. code-block:: bash

	    docker run -v $PWD:/temp -it alice/ubuntu-python sh

	 Once inside, you should be able to navigate to the `/temp`
	 folder and see that's contents are the same as the files on your
	 computer:

	 .. code-block:: bash

	    /# cd /temp
	    /# ls

Mounting a folder can be very useful when you want to run the software
inside your container on many different input files. In other
situations, you may want to save or archive an authoritative version
of your data by adding it to the container permanently.  That's what
we will cover next.

Including personal scripts and data in a container
--------------------------------------------------

Our next project will be to add our own files to a container -
something you might want to do if you're sharing a finished analysis
or just want to have an archived copy of your entire analysis
including the data. Let's as some that we've finished with our
`dummy.py` script and want to add it to the container itself.

In your shell, you should still be in the `dummy` folder in the
`docker-intro` folder.

.. code-block:: bash

   pwd

Output

.. code-block:: bash

   /Users/yourname/foo


We will modify our Dockerfile again to build an image based on Alpine
Linux with Python 3 installed (just as we did perviously). This time
we will add an additional line before the `CMD` line:

.. code-block:: dockerfile

   COPY dummy.py /home

This line will cause Docker to copy the file from your computer into
the container's file system *at build time*. Modify the Dockerfile as
before (or copy the version from the `basic/` subdirectory) and add
the extra copy line. Once you have done that, build the container like
before, but give it a different name:

.. code-block::

   docker build -t alice/ubuntu-dummy .


.. exercise:: Did it work?

   Can you remember how to run a container interactively? Try
   that with this one.  Once inside, try running the Python script.

   .. solution::

      You can start the container interactively like so:

      .. code-block:: bash

         docker run -it alice/ubuntu-dummy sh

      You should be able to run the python command inside the
      container like this:

      .. code-block:: bash

         /# python3 /home/dummy.py

This `COPY` keyword can be used to place your own scripts or own data
into a container that you want to publish or use as a record. Note
that it's not necessarily a good idea to put your scripts inside the
container if you're constantly changing or editing them.  Then,
referencing the scripts from outside the container is a good idea, as
we did in the previous section. You also want to think carefully about
size -- if you run `docker image ls` you'll see the size of each image
all the way on the right of the screen. The bigger your image becomes,
the harder it will be to easily download.

.. callout:: Copying alternatives

   Another trick for getting your own files into a container is by
   using the `RUN` keyword and downloading the files from the
   internet. For example, if your code is in a GitHub repository, you
   could include this statement in your Dockerfile to download the
   latest version every time you build the container:

   .. code-block:: dockerfile

      RUN git clone https://github.com/alice/mycode

   Similarly, the `wget` command can be used to download any file
   publicly available on the internet:

   .. code-block:: dockerfile

      RUN wget ftp://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/2.10.0/ncbi-blast-2.10.0+-x64-linux.tar.gz


More fancy `Dockerfile` options
-------------------------------

We can expand on the example above to make our container even more
"automatic".  Here are some ideas:

Make the `dummy.py` script run automatically:

.. code-block:: dockerfile

   FROM ubuntu:18.04

   COPY dummy.py /home
   RUN apt update && apt install -y python3

   # Run the dummy.py script as the default command
   CMD python3 /home/dummy.py
   # OR
   # CMD ["python3", "/home/dummy.py"]

Build and test it:

.. code-block:: bash

   docker build -t ubuntu-dummy:v1 .
   docker run ubuntu-dummy:v1

Make the `dummy.py` script run automatically with arguments from the
command line:

.. code-block:: dockerfile

   FROM ubuntu:18.04

   COPY dummy.py /home
   RUN apt update && apt install -y python3

   # Run the dummy.py script as the default command and
   # allow people to enter arguments for it
   ENTRYPOINT ["python3", "/home/dummy.py"]

Build and test it:

.. code-block:: bash

   docker build -t ubuntu-dummy:v2 .
   docker run ubuntu-dummy:v2

Add the `dummy.py` script to the `PATH` so you can run it directly:

.. code-block:: dockerfile

   FROM ubuntu:18.04

   COPY dummy.py /home
   # set script permissions
   RUN chmod +x /home/dummy.py
   # add /home folder to the PATH
   ENV PATH /home:$PATH

   RUN apt update && apt install -y python3

Build and test it:

.. code-block:: bash

   docker build -t ubuntu-dummy:v3 .
   docker run alpine-dummy:v3 dummy.py
