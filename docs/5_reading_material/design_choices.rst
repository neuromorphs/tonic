Tonic design choices
====================

This document explains some of the design decision we took and for what reason.

Events are structured numpy arrays
----------------------------------
This was a difficult one to figure out. The options were essentially:

* Raw numpy integer array. This seemed like a natural choice that everyone would understand and Pytorch can easily convert it to tensors. However, this format does not take into account the many different variations in event channel ordering. Some datasets use (xytp), (txyp), (pxyt) and so on. Plus, there are also audio datasets with (tx) or (txp). So there is not a common ordering of event channels in a list of events. 
* Own 'event' class, potentially based on numpy. This homebrewed option would be very obscure to users.
* Structured numpy arrays. They provide two essential advantages:
  1. Columns can be indexed with a string such as events["x"], so the respective ordering doesn't matter anymore.
  2. Dtypes can be set for each channel separately. Polarities typically are of type boolean, whereas timestamps might need float precision. To specialise the dtype for each column saves disk space when storing the arrays. 
  The downside of this format is that it cannot directly be converted to pytorch tensors. A structured array first needs to be converted to an unstructured array.




