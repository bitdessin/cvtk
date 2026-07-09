Installation
############


Quick Installation
******************

**cvtk** requires `PyTorch <https://pytorch.org/get-started/locally/>`_.
While PyTorch can be installed automatically during **cvtk** installation,
it is strongly recommended to install PyTorch first to ensure proper GPU support.

Below is an example of installing PyTorch for CUDA 12.6 followed by **cvtk**:


.. code-block:: bash

    pip install -U pip

    # install PyTorch 2.12.1 with CUDA 12.6 support
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

    # install cvtk
    pip install cvtk





Installation with Full Features
*******************************

**cvtk** optionally integrates with `MMDetection <https://github.com/open-mmlab/mmdetection>`_
to provide access to a comprehensive suite of object detection and instance segmentation models.

.. note::

    The original MMDetection project by OpenMMLab is no longer actively maintained.
    **cvtk** now recommends using OneDL MMDetection, a community fork maintained by the VBTI team,
    which provides regular updates, bug fixes, and improved compatibility.

To enable full detection and segmentation functionality, install the OneDL MMDetection dependencies:


.. code-block:: bash

    pip install -U pip

    # install PyTorch 2.12.1 with CUDA 12.6 support
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

    # install OneDL MMDetection and dependencies
    pip install onedl-mim
    mim install onedl-mmengine
    mim install onedl-mmcv
    mim install onedl-mmdetection

    # install cvtk with full features
    pip install cvtk[full]



Label Studio Backend (Optional)
*******************************

If you want to deploy a Label Studio ML backend with **cvtk**,
install the Label Studio runtime dependencies as well.

You can either install everything via the ``full`` extra,
or install Label Studio packages explicitly:

.. code-block:: bash

    # includes MMDetection + Label Studio dependencies
    pip install cvtk[full]

    # OR install Label Studio stack explicitly
    pip install label-studio-sdk label-studio-ml label-studio-tools


.. note::

    Current Label Studio backend deployment in **cvtk** supports
    detection and instance segmentation model sources.
    Classification source deployment is not supported in this flow.


.. note::

    When running a generated Label Studio backend, ensure these environment
    variables are configured in your runtime environment:

    - ``LABEL_STUDIO_BASE_DATA_DIR``
    - ``LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT``


Troubleshooting Installation Issues
***********************************

If the above installation commands fail due to firewall restrictions, platform unavailability,
or errors like ``ModuleNotFoundError: No module named 'mmcv._ext'``,
you can install the packages from source:

.. note::

    The version numbers in the following examples (v2.3.6.post1 for mmcv and v3.5.1 for mmdetection)
    are provided as examples only.
    Check the official repositories for the latest releases:
    
    - `OneDL MMCv Releases <https://github.com/VBTI-development/onedl-mmcv/releases>`_
    - `OneDL MMDetection Releases <https://github.com/VBTI-development/onedl-mmdetection/releases>`_


.. code-block:: bash

    # remove any existing mmcv and mmdet installations
    pip uninstall -y mmcv mmdet

    # install core dependencies
    pip install onedl-mim
    pip install onedl-mmengine

    # build and install mmcv from source (replace v2.3.6.post1 with latest version)
    wget https://github.com/VBTI-development/onedl-mmcv/archive/refs/tags/v2.3.6.post1.tar.gz -O onedl-mmcv-v2.3.6.post1.tar.gz
    tar -zxvf onedl-mmcv-v2.3.6.post1.tar.gz
    cd onedl-mmcv-2.3.6.post1
    pip install -e . -v --no-cache-dir --no-build-isolation --force-reinstall
    cd ..

    # build and install mmdetection from source (replace v3.5.1 with latest version)
    wget https://github.com/VBTI-development/onedl-mmdetection/archive/refs/tags/v3.5.1.tar.gz -O onedl-mmdetection-v3.5.1.tar.gz
    tar -zxvf onedl-mmdetection-v3.5.1.tar.gz
    cd onedl-mmdetection-3.5.1
    pip install -e . -v --no-cache-dir --no-build-isolation --force-reinstall
    cd ..

    # install cvtk with full features
    pip install cvtk[full]

.. note::

    Source installation may take several minutes to compile native extensions.
    Ensure you have a C++ compiler and development headers installed on your system.


