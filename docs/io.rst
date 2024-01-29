Loading Data
============
There are multiple ways to load AIRR-seq data in ``hicutils``:

#. **(Recomended)** Using existing un-pooled AIRR-formatted files with a
   metadata file with one row per file.

#. Using existing pooled AIRR-formatted files exported from ImmuneDB, where
   pooling metadata is embedded in the file names.

#. Directly downloading and loading data from a hosted ImmuneDB instance using
   its URL and database name.

Examples
--------
.. raw:: html
   :file: notebooks/loading_data.html


API Documentation
-----------------
.. automodule:: hicutils.core.io
   :members:
