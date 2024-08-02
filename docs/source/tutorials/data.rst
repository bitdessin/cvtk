Data Preprocessing
##################


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
The ``--shuffle`` option shuffles the data before splitting.


.. code-block:: sh

    cvtk split --input data.txt --output data_subset.txt --ratios 6:2:2 --shuffle


If the command runs successfully,
it generates the files :file:`data_subset.txt.0`, :file:`data_subset.txt.1`,
and :file:`data_subset.txt.2`` in the current directory.
The number of samples in each file will roughly match the ratio specified by ``--ratios``.


.. code-block:: sh

    wc -l data.txt
    # 400 data.txt

    wc -l data_subset.txt.0 data_subset.txt.1 data_subset.txt.2
    # 240 data_subset.txt.0
    # 80 data_subset.txt.1
    # 80 data_subset.txt.2
    # 400 total



In general, shuffling the data ensures that each subset contains data from all classes.
However, if the dataset is imbalanced, the class distribution in each subset may not be uniform.
In such cases, user can use the ``--stratify`` option
to split the data so that each subset has a uniform class distribution.


.. code-block:: sh

    cvtk split --input all.txt --output data_subset.txt --ratios 6:2:2 --shuffle --stratify




COCO Format Files
*****************

To split COCO format files into multiple subsets, use the ``cvtk cocosplit`` command.


.. code-block:: sh

    cvtk cocosplit --input bbox.json --output data_subset.json --ratios 6:2:2 --shuffle


The command generates the files :file:`data_subset.json.0`, :file:`data_subset.json.1`,
and :file:`data_subset.json.2`` in the directory.
The number of samples in each file will roughly match the ratio specified by ``--ratios``.


The **cvtk** package also provides a command to combine multiple COCO format files into a single file.


You can also combine multiple COCO format files into a single file using the ``cvtk cococombine`` command. 
Note that, specify the paths of the files to be combined as a comma-separated list without spaces between the files.

.. code-block:: sh

    cvtk cococombine --input data_subset.json.0,data_subset.json.1,data_subset.json.2 \
                     --output data_combined.json

