Upscaling AI Workflows
============================

Artificial Intelligence (AI) has become a foundational building block
of our modern world.  Accordingly, a vast effort has been put into
bringing AI to researchers and practitioners of a wide range of
fields.  Nonetheless, the computationally intensive task of training
an AI increasingly requires more computational power than what our
laptops and PCs can offer. Therefore, the ability to develop and train
a neural network on large clusters seems imperative. This workshop
teaches us how to scale an AI-powered application in large clusters,
i.e., supercomputers.

.. prereq::

   Working knowledge of Unix OS is required. In addition, a basic
   understanding of Neural Networks (NNs) is desirable.  Please follow
   the `link <https://docs.docker.com/get-docker/>`_ to install docker 
   locally on your laptop as we need to use it for some part of the 
   begininng of this workshop. Details of using and accessing to 
   the Vega cluster are given in :doc:`setup` section.


.. toctree::
   :hidden:
   :maxdepth: 1

   setup

.. csv-table::
   :widths: auto
   :delim: ;

   5 min ; :doc:`intro-container`
   10 min ; :doc:`intro_docker`
   10 min ; :doc:`mang_contain`
   10 min ; :doc:`create_contain`
   10 min ; :doc:`compx_contain`
   10 min ; :doc:`singlrty_start`
   10 min ; :doc:`work_contain`
   10 min ; :doc:`build_contain`
   25 min ; :doc:`tf_intro`
   25 min ; :doc:`tf_mltgpus`
   25 min ; :doc:`hvd_intro`
   25 min ; :doc:`train_contain`   


.. toctree::
   :maxdepth: 1
   :caption: The lesson

   intro-container
   intro_docker
   mang_contain
   create_contain
   compx_contain
   singlrty_start
   work_contain
   build_contain
   mpi_contain
   tf_intro
   tf_mltgpus
   hvd_intro
   train_contain


.. toctree::
   :maxdepth: 1
   :caption: Reference

   quick-reference
   guide

.. _learner-personas:

Who is the course for?
----------------------




About the course
----------------

This lesson material is developed by the `EuroCC National Competence Center
Sweden (ENCCS) <https://enccs.se/>`_ and taught in ENCCS workshops. It aims
at researchers and developers who have experience working with AI and wish to train their applications on supercomputers.
The lesson material is licensed under `CC-BY-4.0
<https://creativecommons.org/licenses/by/4.0/>`_ and can be reused in any form
(with appropriate credit) in other courses and workshops.
Instructors who wish to teach this lesson can refer to the :doc:`guide` for
practical advice.



See also
--------
Docker provides plenty of educational materials for users. Therefore, checking `Docker official website <https://docs.docker.com/get-started/>`_
is highly recommended. The same can be stated about `Singularity <https://sylabs.io/guides/latest/user-guide/>`_
, where one can find many compelling examples with relevant details.

TensorFlow and Horovorod documentation are also good sources of learning about commands
and their proper use.

Credits
-------

The lesson file structure and browsing layout is inspired by and
derived from `work <https://github.com/coderefinery/sphinx-lesson>`_
by `CodeRefinery <https://coderefinery.org/>`_ licensed under the `MIT
license <http://opensource.org/licenses/mit-license.html>`_. We have
copied and adapted most of their license text.

Materials from the below references have been used in various parts of this course.
  - The Carpentries lesson on `"Reproducible Computational Environments Using Containers: Introduction to Docker and Singularity" <https://epcced.github.io/2020-12-08-Containers-Online/>`_
  - `TensorFlow documentation <https://www.tensorflow.org/guide/distributed_training>`_
  - `Horovod documentation <https://horovod.readthedocs.io/en/stable/>`_

Instructional Material
^^^^^^^^^^^^^^^^^^^^^^

This instructional material is made available under the `Creative
Commons Attribution license (CC-BY-4.0)
<https://creativecommons.org/licenses/by/4.0/>`_.  The following is a
human-readable summary of (and not a substitute for) the `full legal
text of the CC-BY-4.0 license
<https://creativecommons.org/licenses/by/4.0/legalcode>`_.  You are
free to:

- **share** - copy and redistribute the material in any medium or format
- **adapt** - remix, transform, and build upon the material for any purpose,
  even commercially.

The licensor cannot revoke these freedoms as long as you follow these
license terms:

- **Attribution** - You must give appropriate credit (mentioning that your work
  is derived from work that is Copyright (c) Hossein Ehteshami and individual contributors and, where practical, linking
  to `<https://enccs.se>`_), provide a `link to the license
  <https://creativecommons.org/licenses/by/4.0/>`_, and indicate if changes were
  made. You may do so in any reasonable manner, but not in any way that suggests
  the licensor endorses you or your use.
- **No additional restrictions** - You may not apply legal terms or
  technological measures that legally restrict others from doing anything the
  license permits.

With the understanding that:

- You do not have to comply with the license for elements of the material in
  the public domain or where your use is permitted by an applicable exception
  or limitation.
- No warranties are given. The license may not give you all of the permissions
  necessary for your intended use. For example, other rights such as
  publicity, privacy, or moral rights may limit how you use the material.


Software
^^^^^^^^

Except where otherwise noted, the example programs and other software
provided with this repository are made available under the `OSI
<http://opensource.org/>`_-approved
`MIT license <https://opensource.org/licenses/mit-license.html>`__.
