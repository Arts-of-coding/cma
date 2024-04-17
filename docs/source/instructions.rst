Instructions
===== 

.. _instructions:
CLI commands
------------
cPredictor's cli commands are similar across distributions.

Two functions are needed to predict on your query dataset of interest and to import the predictions back into your object.
To see all options from these commands, please run:

.. code-block:: console

   $ SVM_predict --help
   $ SVM_import --help

Another function can be used to automatically retrieve accuracy, recall and F1 scores by using a 5-fold cross-validation on training data.
To see all options from this command, please run:

.. code-block:: console

   $ SVM_performance --help

There is also a fuction currently in development to directly make pseudobulk tables from predicted cell states compared to another single-cell object of interest.
Documentation of this function will be extended in future versions.

.. code-block:: console

   $ SVM_pseudobulk --help

Docker
------------
Documentation will be extended on how to use Docker containers with pre-trained models.

.. _usage:

Download atlases
------------
Documentation will be extended on how to download meta-atlases to load into cPredictor.

Please use the most recent version. Previous versions are included for completeness.