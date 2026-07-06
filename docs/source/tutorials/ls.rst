Utils for Label Studio
######################

The **cvtk** package provides several command-line utilities for interacting with Label Studio,
which facilitate exporting annotations and integrating machine learning models
with Label Studio for assisted labeling.

To enable these utilities to communicate with Label Studio,
several environment variables must be set.
We recommend setting the following variables before using any of these utilities.


.. code-block:: sh

    export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
    export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT="/Users/USERNAME/path/to/ws"
    export LABEL_STUDIO_BASE_DATA_DIR="/U/path/to/labelstudio-data"
    export LABEL_STUDIO_URL="https://0.0.0.0:8080"
    export LABEL_STUDIO_API_KEY="f6dea26f0a0f81883e04681b4e649c600fe50fc"


``LABEL_STUDIO_BASE_DATA_DIR`` specifies the directory where Label Studio stores its data.
The path of this directory is differ depending on the system environment and startup options of Label Studio.
You can use the following command to find out the path of this directory,
start a Python interpreter and execute the commands below.


.. code-block:: python

    >>> import os
    >>> from label_studio.core.utils.params import get_base_data_dir
    >>> os.path.abspath(user_data_dir("label-studio"))
    ~/.local/share/label-studio


``LABEL_STUDIO_URL`` specifies the URL of the Label Studio server.
Just copies the URL displayed in the web browser when accessing Label Studio.

``LABEL_STUDIO_API_KEY`` specifies the API key for accessing Label Studio.
You can find the API key in Label Studio by navigating to the user profile page
(by clicking on the user icon in the top right corner of the screen)
and then selecting the "Account" tab.
The API key is displayed in the "API Key" section of the page.


Annotation Export
*****************

Annotations from Label Studio can be exported via the web interface.  
For convenient access from the command line, the **cvtk** package provides
the ``cvtk ls-export`` utility, which requires the ``--project`` and ``--output`` arguments.


.. code-block:: sh

    cvtk ls-export \
        --project 0 \
        --output ./output/instances.coco.json


This command exports the annotations from the project with ID 0
and saves them to the specified file (:file:`./output/instances.coco.json`).

Users can also provide the URL and API key manually instead of using environment variables.  
For example, if the Label Studio server is running on ``localhost`` at port ``8080``
and the API key is ``f6dea26f0a0f81883e04681b4e649c600fe50fc``,
the command can be executed as follows:


.. code-block:: sh

    cvtk ls-export \
        --project 0 \
        --output ./output/instances.coco.json \
        --url http://localhost:8080 \
        --token f6dea26f0a0f81883e04681b4e649c600fe50fc



ML-Assisted Labeling
********************

The **cvtk** package allows users to integrate trained machine learning models
with Label Studio for ML-Assisted Labeling.
This enables models to assist in the annotation process within Label Studio.
To do this, use the ``cvtk ls-backend`` command,
which generates the necessary source code and supporting files for backend integration.

Assume that a source code file :file:`det.py` for object detection has already been created
using **cvtk** (as described in the `Object Detection Tutorial <det.html>`_),
and the trained model weights are saved in :file:`./outputs/strawberry.pth`.

Then, run the following command,
which generates the backend source code and related files
in the directory specified by the ``--project`` argument.


.. code-block:: sh

    cvtk ls-backend \
        --project lsbackend \
        --source det.py \
        --label ./labels.txt \
        --model ./outputs/strawberry.py \
        --weights ./outputs/strawberry.pth


Source files will be generated in the :file:`lsbackend` directory.  
To start the backend server, run:

.. code-block:: sh

    python lsbackend/main.py --host 0.0.0.0 --port 8080

