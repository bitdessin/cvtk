Installation
############


Quick Installation
******************

The source code for **cvtk** is available on `GitHub <https://github.com/bitdessin/cvtk>`_,
and the built package is available on `PyPI <https://pypi.org/project/cvtk/>`_.
User can install **cvtk** with minimal dependencies using the following command:


.. code-block:: bash

    pip install cvtk



Installation with Full Features
*******************************


For additional functionality,
such as handling COCO format files,
generating source code for object classification tasks with `PyTorch <https://pytorch.org/>`_,
generating source code for object detection and instance segmentation tasks
with `MMDetection <https://mmdetection.readthedocs.io/en/latest/>`_,
and creating `Flask <https://flask.palletsprojects.com/>`_ applications,
the **cvtk** with full dependencies should be installed.

The **cvtk** with full dependencies requires **torch** package (`PyTorch <https://pytorch.org/>`_) version 2.0.0 or later
and **mmdet** package (`MMDetection <https://mmdetection.readthedocs.io/en/latest/>`_) version 3.0.0 or later.
It is recommended that the **torch** and **mmdet** packages
(and the related packages **openmim**, **mmengine**, and **mmcv**)
be installed manually prior to installation.
This is because these packages depend on the operating system (OS) and CUDA version
and cannot be installed automatically.
Below is an example of **torch** and **mmdet** installation in a CUDA 11.8 environment.
For detailed installation instructions, please refer to the tutorial for each package.

.. note::

    The installation of **mmdet** and **mmcv** package is prone to failure.
    It is important to ensure that the versions of **mmdet** and **mmcv** installed
    are compatible with the CUDA and **torch** versions.
    If the versions do not match, an error will occur when calling **mmcv**
    (which is called by **mmdet** and **mmdet** is called by **cvtk**).


.. code-block:: bash

    # PyTorch (CUDA 11.8)
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

    # MMDetection (mim, mmengin, mmcv, mmdet)
    pip install -U openmim
    mim install mmengine
    mim install "mmcv>=2.0.0"
    mim install "mmdet>=3.0.0"


Then, user can install **cvtk** with all dependencies using the following command:

.. code-block:: bash

    pip install cvtk[all]

