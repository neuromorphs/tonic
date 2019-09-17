.. Telluride data augmentation documentation master file, created by
   sphinx-quickstart on Thu Sep 12 10:54:12 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Telluride data augmentation's documentation!
=======================================================

**Telluride Spike Augmentation** is a tool to reduce overfitting in learning algorithms by providing implementations of data augmentation methods for event/spike recordings.

-------------------

**Behold, the power of Requests**::

    >>> r = requests.get('https://api.github.com/user', auth=('user', 'pass'))
    >>> r.status_code
    200
    >>> r.headers['content-type']
    'application/json; charset=utf8'
    >>> r.encoding
    'utf-8'
    >>> r.text
    u'{"type":"User"...'
    >>> r.json()
    {u'private_gists': 419, u'total_private_repos': 77, ...}

The User Guide
--------------

This part of the documentation will provide you with information with each of the augmentation methods.

.. toctree::
   :maxdepth: 2

   methods-index
