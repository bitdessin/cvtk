Data Preprocessing
##################

This section presents functions
that are useful for data preprocessing in computer vision tasks.


Text Data Files
***************

For accurate model evaluation, it is necessary to split the data into subsets such as training,
validation, and test sets for training and evaluating the model.
The **cvtk** package provides a convenient command for splitting a single dataset into multiple subsets.

If the dataset is saved in a text file, use the ``cvtk split`` command.
For example, suppose you have a tab-delimited text file :file:`data.txt`
with the image file paths in the first column and the label names in the second column:


.. code-block:: none
    :caption: data.txt

    data/fruits/strawberry/68e35228.jpg     strawberry
    data/fruits/eggplant/833bda67.jpg       eggplant
    data/fruits/cucumber/c1a79fff.jpg       cucumber
    data/fruits/eggplant/c2e2291e.jpg       eggplant
    data/fruits/tomato/3ee5d80e.jpg tomato
    data/fruits/eggplant/3da0be49.jpg       eggplant
    ...


To split this data into training, validation, and test sets in a 6:2:2 ratio,
run the following command.
Note that, by adding the ``--shuffle`` argument, the data is shuffled before splitting.


.. code-block:: sh

    cvtk split --input data.txt --output data_subset.txt --ratios 6:2:2 --shuffle


The command generates the files
:file:`data_subset.txt.0`, :file:`data_subset.txt.1`, and :file:`data_subset.txt.2`
in the current directory.
The number of samples in each file will roughly match the ratio specified by ``--ratios``.


.. code-block:: sh

    wc -l data.txt
    # 400 data.txt

    wc -l data_subset.txt.0 data_subset.txt.1 data_subset.txt.2
    # 240 data_subset.txt.0
    # 80 data_subset.txt.1
    # 80 data_subset.txt.2
    # 400 total


Theoretically, shuffling the data should give approximately the same proportion of each class in each subset.
However, if the dataset is imbalanced, the distribution of classes in each subset may not be the same
(e.g., data in minor classes may appear in several subsets but not in others).
In such cases, users can use the ``--stratify`` option
to ensure that each subset has a uniform class distribution.


.. code-block:: sh

    cvtk split --input all.txt --output data_subset.txt --ratios 6:2:2 --shuffle --stratify




COCO Format Files
*****************


The **cvtk** package offers several commands
that allow users to combine, split, and retrieve statistics from COCO format files.
The following examples demonstrate how to use these commands effectively.


File Combining
==============

To combine multiple COCO format files, use the ``cvtk coco-combine`` command.
In the example below, we combine :file:`train.json`, :file:`valid.json`, and :file:`test.json` into a single file, :file:`dataset.json`.
Note that when specifying multiple files, separate them with commas without spaces.


.. code-block::

    cvtk coco-combine
        --inputs train.json,valid.json,test.json \
        --output dataset.json


When executed correctly, the :file:`dataset.json` ile is generated.
The number of images in :file:`dataset.json` will be the sum of images from the input files.
Additionally, the categories in :file:`dataset.json` will be the union of categories from the input files,
with newly assigned category IDs.

This functionality can also be executed from Python using the :func:`combine <cvtk.format.coco.combine>` method.



File Splitting
==============


To split a single COCO format file into multiple files, use the ``cvtk coco-split`` command.
In the example below, we shuffle :file:`dataset.json`` and then split it into three files in a 6:2:2 ratio,
saving the output as subset.json.

.. code-block::

    cvtk coco-split
        --input ./data/strawberry/train/bbox.json \
        --output ./output/subset.bbox.json \
        --ratios 6:2:2 \
        --shuffle

Upon successful execution, three files :file:`subset.json.0`, :file:`subset.json.1`, and :file:`subset.json.2` are generated,
each containing images in the specified 6:2:2 ratio.

This functionality can also be executed from Python using the :func:`split <cvtk.format.coco.split>` method.



Image Cropping
==============


To crop images using the bounding box information in a COCO format file, use the ``cvtk coco-crop`` command.
Ensure that the ``file_name`` in the COCO format file correctly points to the actual image file paths.
Convert to absolute paths if necessary.

.. code-block::

    cvtk coco-crop
        --input dataset.json \
        --output cropped_images


After execution, a cropped_images directory is created, containing cropped images.
The filenames of the cropped images follow the format: "original_image_filename__category_name__coordinates.extension",
where the coordinates are the bounding box's top-left and bottom-right coordinates
as integers connected by hyphens (e.g., img01__strawberry__10-20-200-310.jpg).

This functionality can also be executed from Python using the :func:`crop <cvtk.format.coco.crop>` method.


Retrieving Statistics
=====================


To obtain statistics from a COCO format file,
such as the number of images, number of categories,
and the number of objects annotated for each category, use the ``cvtk coco-stat`` command.

.. code-block::

    cvtk coco-stat
        --input ./data/strawberry/train/bbox.json


The statistics are displayed in the standard output.
If you wish to save the statistics in a JSON file or another format,
use the :func:`stats <cvtk.format.coco.stats>` method directly from Python.


