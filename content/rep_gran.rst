.. _rep_gran:

Containers in research workflows
================================

Although this workshop is titled "Reproducible computational
environments using containers", so far we have mostly covered the
mechanics of using Docker with only passing reference to the
reproducibility aspects. In this section, we discuss these aspects in
more detail.

.. callout:: Work in progress

   Note that reproducibility aspects of software and containers are an
   active area of research, discussion and development so are subject
   to many changes. We will present some ideas and approaches here but
   best practices will likely evolve in the near future.

Reproducibility
_______________

By *reproducibility* here we mean the ability of someone else (or your
future self) being able to reproduce what you did computationally at a
particular time (be this in research, analysis or something else) as
closely as possible even if they do not have access to exactly the
same hardware resources # that you had when you did the original work.

Some examples of why containers are an attractive technology to help
with reproducibility include:

- The same computational work can be run across multiple different
  technologies seamlessly (e.g. Windows, macOS, Linux).
- You can save the exact process that you used for your computational
  work (rather than relying on potentially incomplete notes).
- You can save the exact versions of software and their dependencies
  in the image.
- You can access legacy versions of software and underlying
  dependencies which may not be generally available any more.
- Depending on their size, you can also potentially store a copy of
  key data within the image.
- You can archive and share the image as well as associating a
  persistent identifier with an image to allow other researchers to
  reproduce and build on your work.

Sharing images
______________

As we have already seen, the Docker Hub provides a platform for
sharing images publicly. Once you have uploaded an image, you can
point people to its public location and they can download and build
upon it.

This is fine for working collaboratively with images on a day-to-day
basis but the Docker Hub is not a good option for long time archive of
images in support of research and publications as:

- free accounts have a limit on how long an image will be hosted if it
  is not updated
- it does not support adding persistent identifiers to images
- it is easy to overwrite tagged images with newer versions by mistake.

Archiving images
----------------

If for any reason you decided to archive an image, you can use the
command below to take a snapshot of the image.

.. code-block:: bash

   docker save alice/alpine-python:v1 -o alpine-python.tar


.. callout:: Restoring the image from a save

   Unsurprisingly, the command `docker load alpine-python.tar.gz` would
   be used to load the saved container and make it available to be used
   on your system. Note that the command can restore the compressed
   container directly without the need to uncompress first.

Reproducibility good practice
-----------------------------

- Make use of images to capture the computational environment
  required for your work.
- Decide on the appropriate granularity for the images you will use
  for your computational work - this will be different for each
  project/area. Take note of accepted practice from contemporary work
  in the same area.  What are the right building blocks for
  individual images in your work?
- Document what you have done and why - this can be put in comments
  in the Dockerfile and the use of the image described in associated
  documentation and/or publications.  Make sure that references are
  made in both directions so that the image and the documentation are
  appropriately linked.


Container Granularity
---------------------

As mentioned above, one of the decisions you may need to make when
containerising your research workflows is what level of *granularity*
you wish to employ. The two extremes of this decision could be
characterised as:

- Create a single container image with all the tools you require for
  your research or analysis workflow
- Create many container images each running a single command (or step)
  of the workflow and use them in sequence

Of course, many real applications will sit somewhere between these two
extremes.

.. callout:: Positives and negatives

   What are the advantages and disadvantages of the two approaches to
   container granularity for research workflows described above? Think
   about this and write a few bullet points for advantages and
   disadvantages for each approach in the course Etherpad.

   **Single large container**

   .. tabs::

      .. tab:: Advantages

         - Simpler to document
         - Full set of requirements packaged in one place
         - Potentially easier to maintain (though could be opposite if
           working with large, distributed group)

      .. tab:: Disadvantages

         - Could get very large in size, making it more difficult to
           distribute
         - Could use Docker multi-stage build
           docs.docker.com/develop/develop-images/multistage-build to
           reduce size
         - Singularity also has a multistage build feature:
           sylabs.io/guides/3.2/user-guide/definition_files.html#multi-stage-builds
         - May end up with same dependency issues within the container
           from different software requirements
         - Potentially more complex to test
         - Less re-useable for different, but related, work

   **Multiple smaller containers**

   .. tabs::

      .. tab:: Advantages

         - Individual components can be re-used for different, but
           related, work
         - Individual parts are smaller in size making them easier to
           distribute
         - Avoid dependency issues between different softwares
         - Easier to test

      .. tab:: Disadvantage

          - More difficult to document
          - Potentially more difficult to maintain (though could be
            easier if working with large, distributed group)
	  - May end up with dependency issues between component
            containers if they get out of sync

Container Orchestration
_______________________

Although you can certainly manage research workflows that use multiple
containers manually, there are a number of container orchestration
tools that you may find useful when managing workflows that use
multiple containers.  We won't go in depth on using these tools in
this lesson but instead briefly describe a few options and point to
useful resources on using these tools to allow you to explore them
yourself.

- Docker Compose
- Kubernetes
- Docker Swarm

.. callout:: The Wild West

   Use of container orchestration tools for research workflows is a
   relatively new concept and so there is not a huge amount of
   documentation and experience out there at the moment. You may need
   to search around for useful information or, better still, contact
   your friendly neighbourhood to discuss what you want to do.

`Docker Compose <https://docs.docker.com/compose/>`_ provides a
way of constructing a unified workflow (or service) made up of
multiple individual Docker containers. In addition to the individual
Dockerfiles for each container, you provide a higher-level
configuration file which describes the different containers and how
they link together along with shared storage definitions between the
containers. Once this high-level configuration has been defined, you
can use single commands to start and stop the orchestrated set of
containers.


`Kubernetes <https://kubernetes.io>`_ is an open source framework
that provides similar functionality to Docker Compose. Its particular
strengths are that is platform independent and can be used with many
different container technologies and that it is widely available on
cloud platforms so once you have implemented your workflow in
Kubernetes it can be deployed in different locations as required. It
has become the de facto standard for container orchestration.

`Docker Swarm <https://docs.docker.com/engine/swarm/>`_ provides
a way to scale out to multiple copies of similar containers. This
potentially allows you to parallelise and scale out your research
workflow so that you can run multiple copies and increase
throughput. This would allow you, for example, to take advantage of
multiple cores on a local system or run your workflow in the cloud to
access more resources. Docker Swarm uses the concept of a manager
container and worker containers to implement this distribution.
