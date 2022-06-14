# tests/test_api.py
"""Test core API."""
# standard library
import math

# third-party packages
import pandas as pd
import pytest

# local packages
import dcttools


# -------------- dcttools.depth ------------------
@pytest.mark.parametrize(
    ("dct", "expected_result"),
    [
        ({}, 0),
        ({1: 1}, 1),
        ({1: 1, 2: 1}, 1),
        ({1: {1: 2}}, 2),
    ],
)
def test_depth(dct, expected_result):
    """Test correct dcttools.depth functionality."""
    assert dcttools.depth(dct) == expected_result


def test_depth_exception():
    """Test the correct dcttools.depth functionality on wrong argument type."""
    with pytest.raises(Exception):
        dcttools.depth([])


# -------------- dcttools.kfltr ------------------
kfltr_kwargs = {
    "t": {"cat1": 1, "cat2": 2, "cat3": 3},
    "tweak_case": {"cat1": "1", "cat2": "2", "cat3": 1},
    "tweak_txt": "See 'case'",
}


def test_kfltr_simple():
    """Test the correct dcttools.kfltr functionality in simple usage."""
    expected_result = {
        "tweak_case": {"cat1": "1", "cat2": "2", "cat3": 1},
        "tweak_txt": "See 'case'",
    }
    results = dcttools.kfltr(dcts=[kfltr_kwargs], fltr="tweak_")

    assert results[0] == expected_result


def test_kfltr_kwargs():
    """Test the correct dcttools.kfltr functionality using kwargs."""
    expected_result = {
        "tweak_case": {"cat1": "1", "cat2": "2", "cat3": 1},
        "tweak_txt": "See 'case'",
        "x": 10,
    }

    rslts = dcttools.kfltr(dcts=[kfltr_kwargs], fltr="tweak", xcptns=["x"], x=10)
    # flat aggregate results, sinnce kfltr puts additional kwargs into
    # additional dicts by design

    flat_results = dcttools.flaggregate(rslts)

    assert flat_results == expected_result


# -------------- dcttools.kfrep ------------------
kfrep_dct = {"txt": "hi", "s": 5, "kwarg": 1}
kfrep_dflts = {"mst_hve": "yes", "first_kwarg": 0}
kfrep_kwargs = {"first_txt": "there", "first_mst_hve": "no", "first_kwarg": 2}


def test_kfrep_kwargs_as_dict():
    """Test dcttools.kfrep functionality providing kwargs as dict."""
    expected_result = [
        {"txt": "hi", "s": 5, "kwarg": 1},
        {"mst_hve": "yes", "kwarg": 0},
        {"txt": "there", "mst_hve": "no", "kwarg": 2},
    ]

    # Include kwargs into the dcts argument and remove the first_ prefices
    # pylint: disable=unbalanced-tuple-unpacking
    dct, dflts, kwargs = dcttools.kfrep(
        dcts=[kfrep_dct, kfrep_dflts, kfrep_kwargs], fnd="first_"
    )

    assert [dct, dflts, kwargs] == expected_result


def test_kfrep_kwargs_as_dict_not_altering():
    """Test dcttools.kfrep not altering input providing kwargs as dict."""
    expected_result = [
        kfrep_dct.copy(),
        kfrep_dflts.copy(),
        kfrep_kwargs.copy(),
    ]

    # Include kwargs into the dcts argument and remove the first_ prefices
    dcttools.kfrep(dcts=[kfrep_dct, kfrep_dflts, kfrep_kwargs], fnd="first_")

    assert [kfrep_dct, kfrep_dflts, kfrep_kwargs] == expected_result


def test_kfrep_kwargs_as_kwargs_and_frep():
    """Test dcttools.kfrep functionality providing kwargs and kfrepping."""
    expected_result = [
        {"txt": "hi", "s": 5, "kwarg": 1},
        {"mst_hve": "yes", "first_kwarg": 0},
        {"first_kwarg": 2, "second_txt": "there", "second_mst_hve": "no"},
    ]

    # Provide kwargs as kwargs and replace first_ prefices with second_
    # pylint: disable=unbalanced-tuple-unpacking
    dct, dflts, kwargs = dcttools.kfrep(
        dcts=[kfrep_dct, kfrep_dflts],
        fnd="first_",
        rplc="second_",
        xcptns="first_kwarg",
        **kfrep_kwargs
    )

    assert [dct, dflts, kwargs] == expected_result


def test_kfrep_kwargs_as_kwargs_and_frep_not_altering():
    """Test dcttools.kfrep providing kwargs and kfrepping not altering input."""
    expected_result = [
        kfrep_dct.copy(),
        kfrep_dflts.copy(),
        kfrep_kwargs.copy(),
    ]

    # Provide kwargs as kwargs and replace first_ prefices with second_
    dcttools.kfrep(
        dcts=[kfrep_dct, kfrep_dflts],
        fnd="first_",
        rplc="second_",
        xcptns="first_kwarg",
        **kfrep_kwargs
    )

    assert [kfrep_dct, kfrep_dflts, kfrep_kwargs] == expected_result


def test_kfrep_design_case():
    """Test dcttools.kfrep design case."""
    new_kfrep_kwargs = {
        "t": {"cat1": 1, "cat2": 2, "cat3": 3},
        "tweak_case": {"cat1": "1", "cat2": "2", "cat3": 1},
        "tweak_txt": "See 'case'",
    }

    expected_result = {
        "case": {"cat1": "1", "cat2": "2", "cat3": 1},
        "txt": "See 'case'",
    }

    # Chaining filter and find and replace utility to only get kwargs
    # previously containing a tweak_ prefix, deleting this prefix, so
    # the kwargs are ready to be passed to a third party API.

    third_party_api_kwargs = dcttools.kfrep(
        dcts=dcttools.kfltr(dcts=[new_kfrep_kwargs], fltr="tweak_"), fnd="tweak_"
    )

    assert third_party_api_kwargs[0] == expected_result


# -------------- dcttools.kswap ------------------
kswap_nd = {"tweak_case": {"cat1": "1", "cat2": "2", "cat3": 1}}


def test_kswap():
    """Test the correct dcttools.kswap functionality."""
    expected_result = {
        "cat1": {"tweak_case": "1"},
        "cat2": {"tweak_case": "2"},
        "cat3": {"tweak_case": 1},
    }
    results = dcttools.kswap(kswap_nd)

    assert results == expected_result


def test_kswap_not_altering():
    """Test the correct dcttools.kswap functionality not altering the input."""
    expected_result = kswap_nd.copy()
    dcttools.kswap(kswap_nd)

    assert kswap_nd == expected_result


def test_kswap_on_flat():
    """Test dcttools.kswap on non-nested dicts."""
    assert not dcttools.kswap({"cat1": "1", "cat2": "2", "cat3": 1})


def test_kswap_similar():
    """Test the correct dcttools.kswap functionality on similar nested keys."""
    nested_dct = {
        "tweak_case": {"cat1": "1", "cat2": "2", "cat3": 1},
        "use_case": {"cat1": 1, "cat2": 2, "cat3": 1},
    }

    expected_result = {
        "cat1": {"tweak_case": "1", "use_case": 1},
        "cat2": {"tweak_case": "2", "use_case": 2},
        "cat3": {"tweak_case": 1, "use_case": 1},
    }

    assert dcttools.kswap(nested_dct) == expected_result


def test_kswap_multiple():
    """Test the correct dcttools.kswap functionality on multiple keys."""
    nested_dct = {
        "tweak_case": {"cat1": "1", "cat2": "2", "cat3": 1},
        "use_case": {"chap1": 1, "chap2": 2, "chap3": 1},
    }

    expected_result = {
        "cat1": {"tweak_case": "1"},
        "cat2": {"tweak_case": "2"},
        "cat3": {"tweak_case": 1},
        "chap1": {"use_case": 1},
        "chap2": {"use_case": 2},
        "chap3": {"use_case": 1},
    }

    assert dcttools.kswap(nested_dct) == expected_result


# -------------- dcttools.flaggregate ------------------
flagg_dct = {"txt": "hi", "s": 5, "kwarg": 1}
flagg_dflts = {"mst_hve": "yes", "first_kwarg": 0}
flagg_kwargs = {"first_txt": "hi there", "first_mst_hve": "no", "first_kwarg": 2}


def test_flaggregate_kwargs_as_kwargs():
    """Test correct dcttools.flaggregate functionality using kwargs as kwargs."""
    expected_result = {
        "first_kwarg": 2,
        "first_mst_hve": "no",
        "first_txt": "hi there",
        "kwarg": 1,
        "mst_hve": "yes",
        "s": 5,
        "txt": "hi",
    }

    # Aggregate dct, dflts and kwargs, providing kwargs as kwargs:
    result = dcttools.flaggregate([flagg_dct, flagg_dflts], **flagg_kwargs)

    assert result == expected_result


def test_flaggregate_kwargs_as_dicts():
    """Test correct dcttools.flaggregate functionality using kwargs as dicts."""
    expected_result = {
        "first_kwarg": 2,
        "first_mst_hve": "no",
        "first_txt": "hi there",
        "kwarg": 1,
        "mst_hve": "yes",
        "s": 5,
        "txt": "hi",
    }

    # Aggregate dct, dflts, and kwargs, providing kwargs as part of dcts:
    result = dcttools.flaggregate([flagg_dct, flagg_dflts, flagg_kwargs])

    assert result == expected_result


def test_flaggregate_design():
    """Test correct dcttools.flaggregate design case functionality."""
    expected_result = {"txt": "hi there", "s": 5, "kwarg": 2, "mst_hve": "no"}

    # Aggregate dct dflts and kwargs with prior filtering. Note how python's
    # last word policy is kept
    results = dcttools.flaggregate(
        dcts=dcttools.kfrep(dcts=[flagg_dct, flagg_dflts], fnd="first_", **flagg_kwargs)
    )

    assert results == expected_result


# -------------- dcttools.naggregate ------------------
nagg_dct = {"n1": {"txt": "hi", "s": 5}, "n3": {"s": 10}}
nagg_dct_2 = {"n1": {"txt": "hey", "mst_hve": "yes"}, "n2": {"s": 2}}
nagg_dct_3 = {
    "n2": {
        "txt": "there",
    },
    "n3": {"s": 3},
    "n4": {"s": 4},
}


def test_naggregate_two():
    """Test correct dcttools.naggregate using two dicts."""
    expected_result = {
        "n1": {"mst_hve": "yes", "s": 5, "txt": "hey"},
        "n2": {"s": 2},
        "n3": {"s": 10},
    }
    # Aggregate two nested dicts. Note how the dct provided last overrides the
    # other keys:
    results = dcttools.naggregate(nstd_dcts=[nagg_dct, nagg_dct_2])
    assert results == expected_result


def test_naggregate_three():
    """Test correct dcttools.naggregate design case functionality."""
    expected_result = {
        "n1": {"mst_hve": "yes", "s": 5, "txt": "hey"},
        "n2": {"s": 2, "txt": "there"},
        "n3": {"s": 3},
        "n4": {"s": 4},
    }
    # Aggregate three nested dicts. Note how the dct provided last overrides
    # the other keys:
    result = dcttools.naggregate(nstd_dcts=[nagg_dct, nagg_dct_2, nagg_dct_3])
    assert result == expected_result


list_of_naggs = [nagg_dct, nagg_dct_2, nagg_dct_3]


@pytest.mark.parametrize(
    ("dcts", "original", "expected_result"),
    [
        (list_of_naggs, nagg_dct.copy(), nagg_dct),
        (list_of_naggs, nagg_dct_2.copy(), nagg_dct_2),
        (list_of_naggs, nagg_dct_3.copy(), nagg_dct_3),
    ],
)
def test_naggregate_not_altering(dcts, original, expected_result):
    """Test dcttools.naggregate not altering the input."""
    dcttools.naggregate(nstd_dcts=dcts)

    assert original == expected_result


# -------------- dcttools.maggregate ------------------
magg_params = {"cat1": {"txt": "hi", "s": 5}, "cat2": {"txt": "hey", "s": 7}}
magg_tlkys = ["cat1", "cat2", "cat3"]
magg_kwargs = {"s": {"cat1": 1, "cat2": 2, "cat3": 3}}


def test_maggregate_defaults_as_dct():
    """Test correct dcttools.maggregate using defaults as flat dict."""
    expected_result = {
        "cat1": {"first_step": 13, "s": 5, "txt": "hi", "type": "default"},
        "cat2": {"first_step": 13, "s": 7, "txt": "hey", "type": "default"},
    }
    dflts = {"type": "default", "s": 3, "first_step": 13}

    # Using a non nested dict as defaults to ensure each kwarg is present in a
    # nested dict (supposedly) provided by an api:
    result = dcttools.maggregate(
        tlkys=list(magg_params.keys()),
        dcts=[dflts],
        nstd_dcts=[magg_params],
    )

    assert result == expected_result


def test_maggregate_new_tlkeys_and_updated_values():
    """Test correct dcttools.maggregate expaning toplevel and updating values."""
    expected_result = {
        "cat1": {"s": 1, "txt": "hi"},
        "cat2": {"s": 2, "txt": "hey"},
        "cat3": {"s": 3},
    }
    # Using tlkys for adding new nodes to nested dictionairy (supposedly)
    # provided by an api while using kwargs to provide and update entries:
    result = dcttools.maggregate(
        tlkys=magg_tlkys, nstd_dcts=[magg_params], **magg_kwargs
    )

    assert result == expected_result


def test_maggregate_no_fall_back_values():
    """Test correct dcttools.maggregate not using fallback values."""
    expected_result = {
        "cat1": {"s": 1, "txt": "ovrrdn"},
        "cat2": {"s": 2, "txt": "overrdn"},
        "cat3": {"s": 3, "txt": None},
    }

    magg_kwargs_fallback = {
        "s": {"cat1": 1, "cat2": 2, "cat3": 3},
        "txt": {"cat1": "ovrrdn", "cat2": "overrdn"},
    }

    # Not providing fall back defaults as flat dicts can lead to None
    # parameters if kwargs do not provide entries for each top level key:
    result = dcttools.maggregate(
        tlkys=magg_tlkys, nstd_dcts=[magg_params], **magg_kwargs_fallback
    )

    assert result == expected_result


def test_maggregate_design():
    """Test correct dcttools.maggregate design functionality."""
    # suppose this is provided by the api your using (potentially huge)
    api_returns = {
        "cat1": {"txt": "hi", "s": 5},
        "cat2": {"txt": "hey", "s": 7},
    }

    # add yout own defaults
    dflts = {"s": 0}

    # Manipulate certain top level keys and parameters using kwargs.
    # By using your own prefix, you can hard code a set of predefined
    # behaviours unambiguously defined.
    design_kwargs = {
        "t": {"cat1": 1, "cat2": 2, "cat3": 3},
        "tweak_case": {"cat1": "1", "cat2": "2", "cat3": 1},
        "tweak_txt": "See 'case'",
    }

    # Filter only the kwargs needed in this call and replace the prefix to
    # match the api required keywords
    # pylint: disable=unbalanced-tuple-unpacking
    (des_kwargs,) = dcttools.kfrep(
        dcts=dcttools.kfltr(dcts=[design_kwargs], fltr="tweak_"), fnd="tweak_"
    )

    # Aggregate all back into as nested dict as the api expects, using your
    # own defaults and alterations
    result = dcttools.maggregate(
        tlkys=magg_tlkys, dcts=[dflts], nstd_dcts=[api_returns], **des_kwargs
    )

    expected_result = {
        "cat1": {"case": "1", "s": 5, "txt": "See 'case'"},
        "cat2": {"case": "2", "s": 7, "txt": "See 'case'"},
        "cat3": {"case": 1, "s": 0, "txt": "See 'case'"},
    }

    assert result == expected_result


def test_maggregate_nstdkwargs_not_overriding_nstddicts():
    """Test dcttools.maggregate not overriding nstd_dicts with nested kwargs."""
    parameters = {
        "cat1": {"txt": "hi", "s": 5},
        "cat2": {"txt": "hey", "s": 7},
    }
    tlkys = parameters.keys()
    kwargs = {
        "s": {
            "cat1": 1,
        }
    }

    result = dcttools.maggregate(tlkys=tlkys, nstd_dcts=[parameters], **kwargs)

    expected_result = {
        "cat1": {"txt": "hi", "s": 1},  # overriden by kwargs
        "cat2": {"txt": "hey", "s": 7},  # not provided by kwargs
    }

    assert result == expected_result


def test_maggregate_fallback_on_kwargs_not_overriding():
    """Test dcttools.maggregate using dcts fallback when using nested kwargs."""
    parameters = {
        "cat1": {"txt": "hi", "s": 5},
        "cat2": {"txt": "hey", "s": 7},
    }
    tlkys = ["cat1", "cat2", "cat3"]
    kwargs = {"s": {"cat1": 1, "cat2": 2}}
    dflts = {"s": "default"}

    # defaults fallback is used, since kwarg does not provide for cat3 which
    # is required by 'tlkys'
    result = dcttools.maggregate(
        tlkys=tlkys, dcts=[dflts], nstd_dcts=[parameters], **kwargs
    )

    expected_result = {
        "cat1": {"s": 1, "txt": "hi"},
        "cat2": {"s": 2, "txt": "hey"},
        "cat3": {"s": "default"},
    }

    assert result == expected_result


# -------------- dcttools.to_dataframe ------------------
to_dataframe_mapping = {
    "flow_costs": 0,
    "co2_emissions": 0,
    "installed_capacity": 0,
    "accumulated_min": None,
    "accumulated_max": None,
}


def test_to_dataframe_desing_case():
    """Test dcttools.to_dataframe functionality."""
    result_list = [
        ["flow_costs", 0, "accumulated_min", None],
        ["co2_emissions", 0, "accumulated_max", None],
        [
            "installed_capacity",
            0,
            "empty string",
            "empty string",
        ],
    ]
    expected_result = pd.DataFrame(
        result_list, columns=2 * ["key", "value"], index=range(3)
    )

    # Design Case (emulateing an empty string ('') with 'empty string' for
    # verbosity):
    result = dcttools.to_dataframe(
        to_dataframe_mapping, columns=2, fillvalue="empty string"
    )

    assert result.equals(expected_result)


def test_to_dataframe_fill_values_and_index():
    """Test dcttools.to_dataframe functionality."""
    result_list = [
        ["flow_costs", 0, "installed_capacity", 0.0, "accumulated_max", None],
        ["co2_emissions", 0, "accumulated_min", None, "es", "es"],
    ]

    cols = 3
    # Using fill values and adding an Index:
    result = dcttools.to_dataframe(
        to_dataframe_mapping,
        columns=cols,
        fillvalue="es",
        index=pd.Index(
            [
                "i" + str(i)
                for i in range(math.ceil(len(to_dataframe_mapping.keys()) / cols))
            ]
        ),
    )

    expected_result = pd.DataFrame(
        result_list, columns=3 * ["key", "value"], index=["i0", "i1"]
    )

    assert result.equals(expected_result)
