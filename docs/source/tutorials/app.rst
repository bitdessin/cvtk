Web Application
###############


Object Classification
*********************

Build a model for object classification.

.. code-block:: sh
    
    cvtk create --script cls.py --task cls --module vanilla

    python cls.py train \
        --label ./data/fruits/label.txt \
        --train ./data/fruits/train.txt \
        --valid ./data/fruits/valid.txt \
        --test ./data/fruits/test.txt \
        --output_weights ./output/fruits.pth


Build a demo application with FastAPI.

.. code-block:: sh
    
    cvtk app --project clsapp \
       --source cls.py \
       --label ./data/fruits/label.txt \
       --model resnet18 \
       --weights ./cls_output/fruits.pth


Source code can be generated from FastAPI without importing cvtk.


.. code-block:: sh

    cvtk app --project clsapp \
       --source cls.py \
       --label ./data/fruits/label.txt \
       --model resnet18 \
       --weights ./cls_output/fruits.pth \
       --module vanilla



.. code-block:: sh

    uvicorn main:app --host 0.0.0.0 --port 8080
    


Object Detection
****************



Build a model for object classification.

.. code-block:: sh

    cvtk create --script det.py --task det --module vanilla

    python det.py train \
        --label ./data/strawberry/label.txt \
        --train ./data/strawberry/train/bbox.json \
        --valid ./data/strawberry/valid/bbox.json \
        --test ./data/strawberry/test/bbox.json \
        --output_weights ./segm_output/strawberry.pth



Build a demo application with FastAPI.


.. code-block:: sh

    cvtk app --project detapp \
        --source det.py \
        --label ./data/strawberry/label.txt \
        --model ./segm_output/strawberry.py \
        --weights ./segm_output/strawberry.pth






Instance Segmentation
*********************



Build a model for object classification.

.. code-block:: sh

    cvtk create --script segm.py --task segm --module vanilla

    python segm.py train \
        --label ./data/strawberry/label.txt \
        --train ./data/strawberry/train/segm.json \
        --valid ./data/strawberry/valid/segm.json \
        --test ./data/strawberry/test/segm.json \
        --output_weights ./segm_output/strawberry.pth



Build a demo application with FastAPI.


.. code-block:: sh

    cvtk app --project segmapp \
        --source segm.py \
        --label ./data/strawberry/label.txt \
        --model ./segm_output/strawberry.py \
        --weights ./segm_output/strawberry.pth







