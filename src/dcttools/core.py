# src/dcttools/core.py
"""
A collection of dictionairy utilities.

Used to grow on the fly. Aims to provide general purpose abstact
functionalities.

.. note::
    All core functionalities are known to the toplevel dcttools model.
    Meaning they can be used like::

        dcttools.depth(...)

.. autosummary::
   :nosignatures:

   depth
   kfltr
   kfrep
   kswap
   flaggregate
   naggregate
   maggregate
   to_dataframe
"""

import logging
import math
from collections import defaultdict, deque
from itertools import zip_longest

import pandas as pd

logger = logging.getLogger(__name__)


def depth(dct):
    """Find out a dictionairy's depth.

    Convention::

        depth({}) = 0,
        depth({1: 1}) = 1
        depth({1: 1, 2: 1}) = 1
        depth({1: {1: 1}}) = 2

    Parameters
    ----------
    dct : dict
        Dictionairy of which the depth is to be assessed.

    Returns
    -------
    int
        Depth of :paramref:`depth.dct`

    Raises
    ------
    TypeError
        Raised when :paramref:`depth.dct` is ot of instance dict.

    Examples
    --------
    >>> depth({})
    0

    >>> depth({1: 1})
    1

    >>> depth({1: 1, 2: 1})
    1

    >>> depth({1: {1: 1}})
    2

    """
    if not isinstance(dct, dict):
        raise TypeError(f"'{dct}' of type '{type(dct)}' not 'dict'")

    queue = deque([(id(dct), dct, 1)])
    memo = set()
    while queue:
        id_, obj, level = queue.popleft()
        if id_ in memo:
            continue
        memo.add(id_)
        if isinstance(obj, dict):
            queue += ((id(v), v, level + 1) for v in obj.values())
    return level - 1


def kfltr(dcts=(), fltr="", xcptns=(), **kwargs):
    r"""Key Filter out any unwanted entries.

    Return a new dictionairy containing only items with keys containing
    :paramref:`~kfltr.fltr` or keys beeing listed in :paramref:`~kfltr.xcptns`.
    (kfltr = abbrevation of key filter)

    Parameters
    ----------
    dcts: :class:`~collections.abc.Container`, default=()
        Container of Dictionaires of which the keys are to be filtered
        (dcts = dictionairies)

    fltr: str, default=''
        String that is searched for in the dictionairy keys
        (fltr = abbrevation for filter)

    xcptns: :class:`~collections.abc.Container`, default=()
        Container of strings not to be frepped. Aka keys to be ignored.
        (xcptns = abbrevation for exceptions)

    kwargs
        Key words . Of course you can always just
        add the kwargs to be frepped to the dcts iterable.

    Returns
    -------
    list
        List of dicts and kwargs that have been frepped.
        If you only provide one dct and want it to be returned as only one use
        the tuple syntax::

            returned_dict, = kfltr()

    Examples
    --------
    Generating a new dict containing only items with ``tweak_`` in their keys:

    >>> import pprint
    >>> kwargs = {'t': {'cat1': 1, 'cat2': 2, 'cat3': 3},
    ...           'tweak_case': {'cat1': '1', 'cat2': '2', 'cat3': 1},
    ...           'tweak_txt': "See \'case\'"}
    >>> pprint.pprint(*kfltr(dcts=[kwargs], fltr='tweak_'), width=70)
    {'tweak_case': {'cat1': '1', 'cat2': '2', 'cat3': 1},
     'tweak_txt': "See 'case'"}


    Add a kwarg and an exception for on the fly manipulation
    (using aggregate to gather them in 1 dict instead of 2):

    >>> pprint.pprint(flaggregate(dcts=kfltr(
    ...     dcts=[kwargs], fltr='tweak', xcptns=['x'], x=10)))
    {'tweak_case': {'cat1': '1', 'cat2': '2', 'cat3': 1},
     'tweak_txt': "See 'case'",
     'x': 10}

    """
    # Create and empty list which is to be filled with the filtered dicts
    filtered = []

    # iterate through all dictionairies. Do not include kwargs if none
    # stated otherwise there will be an empty kwarg at the end of filtered
    # which leads to unexpected unpacking errors
    for dctnry in [*dcts, kwargs] if kwargs else dcts:

        # Start logging the filter process:
        logger.debug(50 * "-")
        # State code location for easier debugging:
        logger.debug(
            "Filtering a dict inside %s.kfltr with %s keys",
            __name__,
            len(dctnry.keys()),
        )
        # Explicitly log the filtering parmeters:
        logger.debug('Filter for "%s" (exceptions: %s)\n', fltr, xcptns)

        # create a temporary dict for storing the filtered entries.
        tmp_dct = {}

        # iterate through every item of the current dict:
        for key, value in dctnry.items():
            # look for fltr word in the key but respect the exceptions
            if fltr in key or key in xcptns:
                # update the temporary dict if filter hits
                tmp_dct.update({key: value})

                # Explicitly log the filtered key for prove of concept
                logger.debug("Filtered %s", key)

        # Add the filtered dict to output iterable of dicts:
        filtered.append(tmp_dct)

        # Log the end of kflts with a dashed line and a linebreak
        logger.debug("%s", 50 * "-" + "\n")  # use lazy evaluation

    return filtered


def kfrep(dcts=(), fnd="", rplc="", xcptns=(), **kwargs):
    r"""Key Find and Replace dictionairy entries.

    Find :paramref:`~kfrep.fnd` in dictionairy keys and replace them with
    :paramref:`~kfrep.rplc`.
    (kfrep = abbrevation of key find replace)

    Parameters
    ----------
    dcts: :class:`~collections.abc.Container`, default=()
        Iterable of dictionaires of which the keys are to be frepped.
        (dcts = abbrevation for dictionairies)

    fnd: str, default=''
        String that is searched for in the dictionairy keys
        (fnd = abbrevation for find)

    rplc: str, default=''
        String that the found string is replaced with.
        (Leaving it default just removes the found string)
        (rplc = abbrevation for replace)

    xcptns: :class:`~collections.abc.Iterable`, default=()
        Iterable of strings not to be frepped. Aka keys to be ignored.
        (xcptns = abbrevation for exceptions)

    kwargs
        Key words to be frepped. Of course you can always just
        add the kwargs to be frepped to the :paramref:`~kfrep.dcts` container.

    Returns
    -------
    list
        List of dicts and kwargs that have been frepped. Original dicts
        are not altered!

    Examples
    --------
    Include kwargs into the dcts argument and remove the ``first_`` prefices:

    >>> dct = {'txt': 'hi', 's': 5, 'kwarg': 1}
    >>> dflts = {'mst_hve': 'yes', 'first_kwarg': 0}
    >>> kwargs = {'first_txt': 'hi there', 'first_mst_hve': 'no',
    ...           'first_kwarg': 2}
    >>> dct, dflts, kwargs = kfrep(dcts=[dct, dflts, kwargs], fnd='first_')
    >>> print(dct, dflts)
    {'txt': 'hi', 's': 5, 'kwarg': 1} {'mst_hve': 'yes', 'kwarg': 0}
    >>> print(kwargs)
    {'txt': 'hi there', 'mst_hve': 'no', 'kwarg': 2}


    Provide kwargs as kwargs and replace ``first_`` prefices with ``second_``:

    >>> dct = {'txt': 'hi', 's': 5, 'kwarg': 1}
    >>> dflts = {'mst_hve': 'yes', 'first_kwarg': 0}
    >>> kwargs = {'first_txt': 'hi there', 'first_mst_hve': 'no',
    ...           'first_kwarg': 2}
    >>> dct, dflts, kwargs = kfrep(
    ...     dcts=[dct, dflts], fnd='first_', rplc='second_',
    ...     xcptns='first_kwarg', **kwargs)
    >>> print(dct, dflts)
    {'txt': 'hi', 's': 5, 'kwarg': 1} {'mst_hve': 'yes', 'first_kwarg': 0}
    >>> print(kwargs)
    {'first_kwarg': 2, 'second_txt': 'hi there', 'second_mst_hve': 'no'}


    Chaining filter and find and replace utility to only get kwargs previously
    containing a ``tweak_`` prefix, deleting this prefix, sothe kwargs are
    ready to be passed to a third party API:

    >>> kwargs = {'t': {'cat1': 1, 'cat2': 2, 'cat3': 3},
    ...           'tweak_case': {'cat1': '1', 'cat2': '2', 'cat3': 1},
    ...           'tweak_txt': "See \'case\'"}
    >>> print(*kfrep(dcts=kfltr(dcts=[kwargs], fltr='tweak_'),
    ...              fnd='tweak_'))
    {'case': {'cat1': '1', 'cat2': '2', 'cat3': 1}, 'txt': "See 'case'"}
    """
    # Create an empty iterable which is to be filled with the frepped dicts:
    frepped = []
    # iterate through all dictionairies.
    # Do not include empty kwargs in the process otherwise an empty dict will
    # be appended to frepped
    for dctnry in [*dcts, kwargs] if kwargs else dcts:

        dctnry = dctnry.copy()
        # Start logging the frepping process:
        logger.debug(50 * "-")
        # State code location for easier debugging:
        logger.debug(
            "Frepping a dict inside %s.kfrep with %s keys",
            __name__,
            len(dctnry.keys()),
        )
        # Explicitly log the filtering parmeters:
        logger.debug('Find "%s", replace: "%s" ', fnd, rplc)
        logger.debug("(exceptions: %s)\n", xcptns)

        # iterating through all keys in dict:
        # USE COPY OF A DICT HERE SINCE THE DICT IS CHANGED DURING ITERATION
        for key in dctnry.copy().keys():
            # is key to be filtered ?
            if fnd in key and key not in xcptns:
                # ... yes, so create a temp copy of original key:
                original_key = key
                # ... filter out the filter string:
                key = key.replace(fnd, rplc)
                # create a new entry with new key and delete the old one:
                dctnry[key] = dctnry.pop(original_key)

                # Explicitly log the frepped key for prove of concept
                logger.debug('Frepped "%s" with "%s"', original_key, key)

        # Add the found and replaced dict to output iterable of dicts:
        frepped.append(dctnry)

        # Log the end of kfrep with a dashed line and a linebreak
        logger.debug("%s", 50 * "-" + "\n")  # use lazy evaluation

    return frepped


def kswap(nstd_dct):
    r"""Key Swap, top and sublevel keys of a nested dict.

    Parameters
    ----------
    nstd_dct: dict
        Nested dictionairy of depth 1 of which the top level keys and the
        nested keys are to be swapped.
        (nstd_dct= abbrevation for nested dictionairy)

    Returns
    -------
    dict
        Nested dictionairy with its top and sublevel keys swapped.


    .. warning::

       Returns empty dictionairy on non-nested dictinairies!

    Examples
    --------
    >>> import pprint
    >>> nd = {'tweak_case': {'cat1': '1', 'cat2': '2', 'cat3': 1}}
    >>> pprint.pprint(kswap(nd))
    {'cat1': {'tweak_case': '1'},
     'cat2': {'tweak_case': '2'},
     'cat3': {'tweak_case': 1}}
    """
    # Start logging the filter process:
    logger.debug(50 * "-")
    # State code location for easier debugging:
    logger.debug(
        "Swapping keys inside %s.kswap with %s keys",
        __name__,
        len(nstd_dct.keys()),
    )

    key_swapped = defaultdict(dict)
    for tlkey in nstd_dct.keys():
        if isinstance(nstd_dct[tlkey], dict):
            for subkey, value in nstd_dct[tlkey].items():
                key_swapped[subkey].update({tlkey: value})

                logger.debug('Swapped "%s" with "%s"', tlkey, subkey)

    # Log the end of kswap with a dashed line and a linebreak
    logger.debug("%s", 50 * "-" + "\n")

    return dict(key_swapped)


def flaggregate(dcts=(), **kwargs):
    r"""Flat Aggregate single level dicts and kwargs.

    Keys are overriden depending on the order the dicts are provided.
    Python's last word policy is kept (i.e entries in :paramref:`kwargs` will
    override the others)

    Note
    ----
    Yes, it does the same as unpacking dcts, and then unpacking everything
    in the order it was supplied. Though it brings the benefit of having a
    concise name, while creating a detailed log. Has the potential to declutter
    your code.

    Parameters
    ----------
    dcts: :class:`~collections.abc.Container`, default=()
        Container of dictionaires which are to be aggregated. The order in
        which they are provided is crucial. The dict coming last will
        potentially override every other dict. (dcts = dictionairies)

    kwargs
        Key words to be aggregated. Of course you can always just
        add the kwargs to  :paramref:`~flaggregate.dcts`. This is especially
        usefull if u want them to be potentially overriden.

    Returns
    -------
    dict
        Dictionairy containing the aggregated dicts and kwargs.


    Examples
    --------
    >>> dct = {'txt': 'hi', 's': 5, 'kwarg': 1}
    >>> dflts = {'mst_hve': 'yes', 'first_kwarg': 0}
    >>> kwargs = {'first_txt': 'hi there', 'first_mst_hve': 'no',
    ...           'first_kwarg': 2}


    Aggregate dct, dflts and kwargs, providing kwargs as kwargs:

    >>> import pprint
    >>> pprint.pprint(flaggregate([dct, dflts], **kwargs))
    {'first_kwarg': 2,
     'first_mst_hve': 'no',
     'first_txt': 'hi there',
     'kwarg': 1,
     'mst_hve': 'yes',
     's': 5,
     'txt': 'hi'}

    Aggregate dct, dflts, and kwargs, providing kwargs as part of dcts:

    >>> pprint.pprint(flaggregate([dct, dflts, kwargs]))
    {'first_kwarg': 2,
     'first_mst_hve': 'no',
     'first_txt': 'hi there',
     'kwarg': 1,
     'mst_hve': 'yes',
     's': 5,
     'txt': 'hi'}

    Aggregate dct dflts and kwargs with prior filtering. Note how the dct
    provided last overrides the other keys:

    >>> print(flaggregate(
    ...     dcts=kfrep(dcts=[dct, dflts], fnd='first_', **kwargs)))
    {'txt': 'hi there', 's': 5, 'kwarg': 2, 'mst_hve': 'no'}

    """
    # Create an empty iterable to aggregate the dict items into
    aggregated = {}

    # Aggregate dcts and kwargs into one list if kwargs were uitlized:
    if kwargs:
        dctnrs = [*dcts, kwargs]
    else:
        dctnrs = dcts

    # Start logging the filter process:
    logger.debug(50 * "-")
    # State code location for easier debugging:
    logger.debug("Aggregating %s dicts inside %s.flaggregate\n", len(dctnrs), __name__)

    # iterate through all dictionairies.
    for dctnry in dctnrs:
        # get the key-value pair of the current  dict:
        for key, value in dctnry.items():
            # update the key-value pair
            # note this will potentially overwrite previous entries
            # (which is the desired behaviour)
            if key in aggregated:
                logger.debug('Value "%s" for key "%s" ', aggregated[key], key)
                logger.debug('is overridden by "%s"', value)

            aggregated[key] = value

    # Log the end of kfrep with a dashed line and a linebreak
    logger.debug("%s", 50 * "-" + "\n")  # lazy logging evaluation

    return aggregated


def naggregate(nstd_dcts=()):
    r"""Nested Aggregate, aggregate nested dicts of depth 1.

    Keys are overriden depending on the order the dicts are provided.
    Python's last word policy is kept (the last keyword will take precedence)

    Parameters
    ----------
    nstd_dcts: :class:`~collections.abc.Container`, default=()
        Container of nested dictionaires which are to be aggregated. The order
        in which they are provided is crucial. The dict coming last will
        potentially override every other.
        (nstd_dcts= abbrevation for nested dictionairies)

    Returns
    -------
    aggregated: dict
        Nested dictionairy containing all items present in
        :paramref:`~naggregate.nstd_dcts` dicts.

    Examples
    --------
    Aggregate two nested dicts. Note how the dct provided last overrides the
    other keys:

    >>> import pprint
    >>> nstd_dct = {'n1': {'txt': 'hi', 's': 5}, 'n3': {'s': 10}}
    >>> nstd_dct_2 = {'n1': {'txt': 'hey', 'mst_hve': 'yes'}, 'n2': {'s': 2}}
    >>> pprint.pprint(naggregate(nstd_dcts=[nstd_dct, nstd_dct_2]))
    {'n1': {'mst_hve': 'yes', 's': 5, 'txt': 'hey'},
     'n2': {'s': 2},
     'n3': {'s': 10}}

    Aggregate three nested dicts. Note how the dct provided last overrides the
    other keys:

    >>> nstd_dct = {'n1': {'txt': 'hi', 's': 5}, 'n3': {'s': 10}}
    >>> nstd_dct_2 = {'n1': {'txt': 'hey', 'mst_hve': 'yes'}, 'n2': {'s': 2}}
    >>> nstd_dct_3 = {'n2': {'txt': 'there', }, 'n3': {'s': 3}, 'n4': {'s': 4}}
    >>> pprint.pprint(naggregate(nstd_dcts=[nstd_dct, nstd_dct_2, nstd_dct_3]))
    {'n1': {'mst_hve': 'yes', 's': 5, 'txt': 'hey'},
     'n2': {'s': 2, 'txt': 'there'},
     'n3': {'s': 3},
     'n4': {'s': 4}}
    """
    # Start logging the aggregation process:
    logger.debug(50 * "-")
    # State code location for easier debugging:
    logger.debug(
        "Aggregating %s nested dicts inside %s.naggregate\n", len(nstd_dcts), __name__
    )

    # Use a temporary default dict to simplify aggregating algorithm
    # (Preventing KeyErrors, when adding new top level key entries)
    aggregated = defaultdict(dict)

    # iterate though all dictionairies in nstd_dicts:
    for dctnry in nstd_dcts:
        # iterate through all top level keys of the respective dict:
        for tlky in dctnry:
            # get the key-value pair of the current sublevel dict:
            for key, value in dctnry[tlky].items():
                # store this key-value pair under its top level key:
                # note this will potentially overwrite previous entries
                # (which is the desired behaviour)
                if tlky in aggregated:
                    if key in aggregated[tlky]:
                        logger.debug(
                            'Value "%s" for key "%s" ', aggregated[tlky][key], key
                        )
                        logger.debug('is overridden by "%s"', value)

                aggregated[tlky][key] = value
                logger.debug('Filled ["%s"]["%s"] with "%s"', tlky, key, value)

    # turn the default dict into an ordinary dict to allow for broader
    # applications (the user can retransform anytime anyways)
    aggregated = dict(aggregated)

    # Log the end of kfrep with a dashed line and a linebreak
    logger.debug("%s", 50 * "-" + "\n")

    return aggregated


def maggregate(tlkys=(), dcts=(), nstd_dcts=(), **kwargs):
    r"""Mixed Aggregate, aggregate non nested dicts, nested dicts and kwargs.

    This function returns a nested dictionary having a key-value pair for
    every kwarg provided in :paramref:`~maggregate.dcts` (and the kwargs
    already present in the respective top level entry) of each
    :paramref:`~maggregate.nstd_dct` for each key listed in tlkys.

    Hierarchy is as follows::paramref:`~maggregate.dcts` <
    :paramref:`~maggregate.nstd_dct` < :paramref:`~maggregate.kwargs`

    Designed to be used when dynamically splitting
    :paramref:`~maggregate.kwargs` into several kwargs stored in a nested dict
    to distribute them among sub function calls
    falling back on key-value pairs listed in the ordinary dicts.

    A (potentially api provided) bunch of default kwargs listed in an
    ordinary dct will be used to populate (potentially api provided
    nested dictionairies)  potentially tweaked by the enduser supplying custom
    kwargs, falling back on on the defaults if necessary.

    (For a practical use case see the interfaces.visualie.nx module. Those
    functions are designed to be parameterized in an automated fashion to
    allow for complex behaviour utilizing a nested dict. They however enable
    the user to also tweak any number of parameters by providing nested or
    single layer kwargs.)


    Parameters
    ----------
    tlkys: :class:`~collections.abc.Container`, default=()
        Iterable of keys the algorithm should be applied to
        (tlkys = abbrevation of top level keys)

    dcts: :class:`~collections.abc.Container`, default=()
        Container of (prefilled) dictionairies of default kwargs as in::

            [{key1: value1, keyN: valueN, ...}, ...]

        Will be overriden by :paramref:`~maggregate.nstd_dct`.
        (dcts = abbrevation of dictionaires)

    nstd_dcts: :class:`~collections.abc.Container`, default=()
        Container of (prefilled) nested dictionairies to be aggregated as
        in::

            [{tlky1: {key1: value1}, tlky2: {key2: value2}, ...}, ...]

        Key-value pairs will take precedence over respective pairs found in
        :paramref:`~maggregate.dcts` while beeing overriden by respective
        pairs found in :paramref:`~maggregate.kwargs`.
        (nstd_dicts = abbrevation of nested dictionairies)

    kwargs
        Keyword arguments to aggregate. Key-value pairs will take precedence
        over respective pairs found in :paramref:`~maggregate.nstd_dcts`.

    Returns
    -------
    dict
        A new nested dict containing the deep copies of all entries aggregated
        as in::

            {tlkys: {nlkeys: nlvalues}}

        (aggr_dict = abbrevation of aggregated dictionairy)

    Examples
    --------
    Using a non nested dict as defaults to ensure each kwarg is present in
    a nested dict (supposedly) provided by an api:

    >>> import pprint
    >>> parameters = {'cat1': {'txt': 'hi', 's': 5},
    ...               'cat2': {'txt': 'hey', 's': 7}}
    >>> dflts = {'type': 'default', 's': 3, 'first_step': 13,}
    >>> pprint.pprint(maggregate(
    ...     tlkys=list(parameters.keys()), dcts=[dflts,],
    ...     nstd_dcts=[parameters, ]))
    {'cat1': {'first_step': 13, 's': 5, 'txt': 'hi', 'type': 'default'},
     'cat2': {'first_step': 13, 's': 7, 'txt': 'hey', 'type': 'default'}}


    Using tlkys for adding new nodes to nested dictionairy (supposedly)
    provided by an api while using kwargs to provide and update entries:

    >>> parameters = {'cat1': {'txt': 'hi', 's': 5},
    ...               'cat2': {'txt': 'hey', 's': 7}}
    >>> tlkys = ['cat1', 'cat2', 'cat3']
    >>> kwargs = {'s': {'cat1': 1, 'cat2': 2, 'cat3': 3}}

    >>> pprint.pprint(
    ...    maggregate(tlkys=tlkys, nstd_dcts=[parameters,], **kwargs))
    {'cat1': {'s': 1, 'txt': 'hi'},
     'cat2': {'s': 2, 'txt': 'hey'},
     'cat3': {'s': 3}}


    Not providing fall back defaults as flat dicts can lead to None parameters
    if kwargs do not provide entries for each top level key:

    >>> parameters = {'cat1': {'txt': 'hi', 's': 5},
    ...               'cat2': {'txt': 'hey', 's': 7}}
    >>> tlkys = ['cat1', 'cat2', 'cat3']
    >>> kwargs = {'s': {'cat1': 1, 'cat2': 2, 'cat3': 3},
    ...           'txt': {'cat1': 'ovrrdn', 'cat2': 'overrdn'}}
    >>> pprint.pprint(
    ...     maggregate(tlkys=tlkys, nstd_dcts=[parameters,], **kwargs))
    {'cat1': {'s': 1, 'txt': 'ovrrdn'},
     'cat2': {'s': 2, 'txt': 'overrdn'},
     'cat3': {'s': 3, 'txt': None}}


    Design Case:

    Using dcts as defaults, nstd_dcts as (supposedly) api provided parameters
    tweaking them using kwargs prefixed by ``tweak_`` provided by an upper
    layer function potentially containing hundreds of different kwargs:

    Using a nested dict as (suppoedlyd) api provided parameters:
    (This wouild be outside the code you write and potentially be HUGE)

    >>> api_returns = {'cat1': {'txt': 'hi', 's': 5},
    ...                'cat2': {'txt': 'hey', 's': 7}}

    Using dcts as defaults (your own coding effort):

    >>> dflts = {'s': 0}


    Using a list of keys to expand the parameter set (your own coding effort):

    >>> tlkys = ['cat1', 'cat2', 'cat3']

    Manipulating certain top level keys and parameters using kwargs By using
    your own prefix, you can hard code a set of predefined behaviours
    unambiguously defined:

    >>> kwargs = {'t': {'cat1': 1, 'cat2': 2, 'cat3': 3},
    ...           'tweak_case': {'cat1': '1', 'cat2': '2', 'cat3': 1},
    ...           'tweak_txt': "See \'case\'"}

    Filter only the kwargs needed in this call and replace the prefix to
    match the api required keywords:

    >>> kwargs, = kfrep(dcts=kfltr(dcts=[kwargs], fltr='tweak_'), fnd='tweak_')

    Aggregate everything into a nested dict as the api expects, using your own
    defaults and alterations:

    >>> pprint.pprint(maggregate(tlkys=tlkys, dcts=[dflts, ],
    ...                          nstd_dcts=[api_returns, ], **kwargs))
    {'cat1': {'case': '1', 's': 5, 'txt': "See 'case'"},
     'cat2': {'case': '2', 's': 7, 'txt': "See 'case'"},
     'cat3': {'case': 1, 's': 0, 'txt': "See 'case'"}}
    """
    # Start logging the aggregation process:
    logger.debug(50 * "-")
    # State code location for easier debugging:
    logger.debug("Aggregating %s dict, %s nested dict ", len(dcts), len(nstd_dcts))
    logger.debug("and %s kwargs ", len(kwargs))
    logger.debug("in %s.maggregate\n", __name__)

    # Use a temporary default dict to simplify aggregating algorithm
    # (Preventing KeyErrors, when adding new top level key entries)
    aggregated = defaultdict(dict)

    # Aggregate non nested  dictionairies for decluttering the algorithm
    aggregates = flaggregate(dcts=dcts)

    # make nested_aggregates a defaultdict(dict) to automatically
    # add an empty dict during attempted. Prevents KeyErrors.
    # Note: this will not lead to falsy returned empty dicts, cause
    # theese dicts will be accessed for kwargs, via dict.get
    nested_aggregates = defaultdict(dict)
    # Aggregate nested dictionaires for decluttering aggregating algorithm
    nested_aggregates.update(naggregate(nstd_dcts=nstd_dcts))

    logger.debug("Finished pre aggragation, starting with maggregate")
    # Iterrate through every top level key, to have an entry for each:
    for tlky in tlkys:
        # iterate through every keyword argument...
        for kwarg in {**aggregates, **nested_aggregates[tlky], **kwargs}:

            # create a temporary source str for logging:
            src = ""
            # Accessing differs depending on current kwarg beeing a dict or not
            if isinstance(kwargs.get(kwarg), dict):
                # kwarg is a dict ENTRY (aka kwargs[kwarg] is a dict), so ...

                # ... if kwarg has an entry it takes precedence ...
                if tlky in kwargs[kwarg]:
                    aggregated[tlky][kwarg] = kwargs[kwarg][tlky]
                    src = "kwargs"

                # ...if not, nstd entry remains if present...
                elif kwarg in nested_aggregates[tlky]:
                    aggregated[tlky][kwarg] = nested_aggregates[tlky][kwarg]
                    src = "nested dicts"

                # ... no it is not present, so fall back on dct
                elif kwarg in aggregates:
                    aggregated[tlky][kwarg] = aggregates[kwarg]
                    src = "flat dicts"

                # dct didnt help either so fill value with None...
                else:
                    aggregated[tlky][kwarg] = None
                    src = "no source"

            else:
                # kwargs[kwarg] is NOT a dict, so ...

                # ... if kwarg has an entry it takes precedence ...
                if kwarg in kwargs:
                    aggregated[tlky][kwarg] = kwargs[kwarg]
                    src = "kwargs"

                # ...if not, nstd entry remains if present...
                elif kwarg in nested_aggregates[tlky]:
                    aggregated[tlky][kwarg] = nested_aggregates[tlky][kwarg]
                    src = "nested dicts"

                # ... no it is not present, so fall back on dct
                else:  # elif kwarg in aggregates:
                    aggregated[tlky][kwarg] = aggregates[kwarg]
                    src = "flat dicts"

                # # dct didnt help either so fill value with None...
                # not working, since kwarg is not iterated over if not
                # present in kwargs, defaults or nstd_dict[tlkey]
                # else:
                #     aggregated[tlky][kwarg] = None
                #     src = "no source"

                logger.debug(
                    'Filled ["%s"]["%s"] with "%s" from "%s"',
                    tlky,
                    kwarg,
                    aggregated[tlky][kwarg],
                    src,
                )

    # turn the default dict into an ordinary dict to allow for broader
    # applications (the user can retransform anytime anyways)
    aggregated = dict(aggregated)

    logger.debug("%s", 50 * "-" + "\n")

    return aggregated


def to_dataframe(mapping, columns, fillvalue=None, index=None):
    r"""Convert a mapping to a table controlling the number of columns.

    Parameters
    ----------
    mapping: :class:`~collections.abc.Mapping`
        Mapping to be converted into a :class:`pandas.DataFrame`

    columns: int
        Number of key-value column pairs of the result table.
        (i.e. columns=2 results in a total of 4 columns, 2 representing the
        keys, the other 2 representing the values)

    fillvalue: str, None, default=None
        If number of key-value pairs inside the mapping is not an integer
        multiple of :paramref:`~to_dataframe.columns` the modulus amount
        of entries is filled using :paramref:`~to_dataframe.fillvalue`.

        (See also :func:`~itertools.zip_longest`)

    index: :class:`pandas.Index`, default=None
        Index used for createing the table.

        .. warning::
            Make sure number of index entries equals::

                 math.ceil(len(mapping.keys()) / columns))

    Returns
    -------
    :class:`pandas.DataFrame`
        DataFrame holding the data of :paramref:`~to_dataframe.mapping`,
        using the number of :paramref:`~to_dataframe.columns` stated.

    Examples
    --------
    Design Case (emulateing an empty string (``''``) with
    ``'empty string'`` for verbosity):

    >>> mapping = {
    ...     'flow_costs': 0,
    ...     'co2_emissions': 0,
    ...     'installed_capacity': 0,
    ...     'accumulated_min': None,
    ...     'accumulated_max': None}
    >>> print(to_dataframe(mapping, columns=2, fillvalue='empty string'))
                      key value              key         value
    0          flow_costs     0  accumulated_min          None
    1       co2_emissions     0  accumulated_max          None
    2  installed_capacity     0     empty string  empty string


    Using default fill values and adding an Index:

    >>> import pandas, math
    >>> cols = 3
    >>> print(to_dataframe(
    ...       mapping, columns=cols,
    ...       index=pandas.Index(
    ...           ['i' + str(i) for i in range(
    ...               math.ceil(len(mapping.keys())/cols))])))
                  key value                 key value              key value
    i0     flow_costs     0  installed_capacity   0.0  accumulated_max  None
    i1  co2_emissions     0     accumulated_min   NaN             None  None
    """
    length = math.ceil(len(mapping.keys()) / columns)

    args = [iter(mapping.keys())] * length
    keys = list(zip(*zip_longest(*args, fillvalue=fillvalue)))
    args = [iter(mapping.values())] * length
    values = list(zip(*zip_longest(*args, fillvalue=fillvalue)))

    table = pd.concat(
        [
            pd.DataFrame(keys, index=index),
            pd.DataFrame(values, index=index),
        ],
        axis="columns",
    )

    # resort the table columns
    table = table[list(range(columns))]

    # relabel the table columns
    table.columns = columns * ["key", "value"]  # rename column pair lables
    return table
