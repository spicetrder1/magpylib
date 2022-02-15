from copy import deepcopy
import pytest
import param
from magpylib._src.defaults.defaults_utility import (
    MagicParameterized,
    color_validator,
    get_defaults_dict,
    update_nested_dict,
    magic_to_dict,
    linearize_dict,
    COLORS_MATPLOTLIB_TO_PLOTLY,
)


def test_update_nested_dict():
    """test all argument combinations of `update_nested_dicts`"""
    # `d` gets updated, that's why we deepcopy it
    d = {"a": 1, "b": {"c": 2, "d": None}, "f": None, "g": {"c": None, "d": 2}, "h": 1}
    u = {"a": 2, "b": 3, "e": 5, "g": {"c": 7, "d": 5}, "h": {"i": 3}}
    res = update_nested_dict(
        deepcopy(d), u, same_keys_only=False, replace_None_only=False
    )
    assert res == {
        "a": 2,
        "b": 3,
        "e": 5,
        "f": None,
        "g": {"c": 7, "d": 5},
        "h": {"i": 3},
    }, "failed updating nested dict"
    res = update_nested_dict(
        deepcopy(d), u, same_keys_only=True, replace_None_only=False
    )
    assert res == {
        "a": 2,
        "b": 3,
        "f": None,
        "g": {"c": 7, "d": 5},
        "h": {"i": 3},
    }, "failed updating nested dict"
    res = update_nested_dict(
        deepcopy(d), u, same_keys_only=True, replace_None_only=True
    )
    assert res == {
        "a": 1,
        "b": {"c": 2, "d": None},
        "f": None,
        "g": {"c": 7, "d": 2},
        "h": 1,
    }, "failed updating nested dict"
    res = update_nested_dict(
        deepcopy(d), u, same_keys_only=False, replace_None_only=True
    )
    assert res == {
        "a": 1,
        "b": {"c": 2, "d": None},
        "f": None,
        "g": {"c": 7, "d": 2},
        "e": 5,
        "h": 1,
    }, "failed updating nested dict"


def test_magic_to_dict():
    """test all argument combinations of `magic_to_dict`"""
    d = {"a_b": 1, "c_d_e": 2, "a": 3, "c_d": {"e": 6}}
    res = magic_to_dict(d, separator="_")
    assert res == {"a": 3, "c": {"d": {"e": 6}}}
    d = {"a.b": 1, "c": 2, "a": 3, "c.d": {"e": 6}}
    res = magic_to_dict(d, separator=".")
    assert res == {"a": 3, "c": {"d": {"e": 6}}}
    with pytest.raises(AssertionError):
        magic_to_dict(0, separator=".")
    with pytest.raises(AssertionError):
        magic_to_dict(d, separator=0)


def test_linearize_dict():
    """test all argument combinations of `magic_to_dict`"""
    mydict = {
        "line": {"width": 1, "style": "solid", "color": None},
        "marker": {"size": 1, "symbol": "o", "color": None},
    }
    res = linearize_dict(mydict, separator=".")
    assert res == {
        "line.width": 1,
        "line.style": "solid",
        "line.color": None,
        "marker.size": 1,
        "marker.symbol": "o",
        "marker.color": None,
    }, "linearization of dict failed"
    with pytest.raises(AssertionError):
        magic_to_dict(0, separator=".")
    with pytest.raises(AssertionError):
        magic_to_dict(mydict, separator=0)


@pytest.mark.parametrize(
    "color, allow_None, color_expected",
    [
        (None, True, None),
        ("blue", True, "blue"),
        ("r", True, "red"),
        (0, True, "#000000"),
        (0.5, True, "#7f7f7f"),
        ("0.5", True, "#7f7f7f"),
        ((127, 127, 127), True, "#7f7f7f"),
        ("rgb(127, 127, 127)", True, "#7f7f7f"),
    ]
    + [(shortC, True, longC) for shortC, longC in COLORS_MATPLOTLIB_TO_PLOTLY.items()],
)
def test_good_colors(color, allow_None, color_expected):
    """test color validator based on matploblib validation"""

    assert color_validator(color, allow_None=allow_None) == color_expected


@pytest.mark.parametrize(
    "color, allow_None, expected_exception",
    [
        (None, False, ValueError),
        (-1, False, ValueError),
        ((0, 0, 0, 0), False, ValueError),
        ((-1, 0, 0), False, ValueError),
        ((0, 0, 260), False, ValueError),
        ((0, "0", 200), False, ValueError),
        ("rgb(a, 0, 260)", False, ValueError),
        ("2", False, ValueError),
        ("mybadcolor", False, ValueError),
    ],
)
def test_bad_colors(color, allow_None, expected_exception):
    """test color validator based on matploblib validation"""

    with pytest.raises(expected_exception):
        color_validator(color, allow_None=allow_None)


def test_MagicParameterized():
    """test MagicParameterized class"""

    class MagicParam1(MagicParameterized):
        "MagicParameterized test subclass"

        listparam = param.List()

    class MagicParam2(MagicParameterized):
        "MagicParameterized test subclass"

        tupleparam = param.Tuple(allow_None=True)
        classselector = param.ClassSelector(MagicParam1, default=MagicParam1())

    mp1 = MagicParam1(listparam=(1,))

    # check setting attribute/property
    assert mp1.listparam == [1], "`mp1.listparam` should be `[1]`"
    with pytest.raises(AttributeError):
        setattr(mp1, "listparame", 2)  # only properties are allowed to be set

    # check assigning class to subproperty
    mp2 = MagicParam2(tupleparam=[2, 2])
    mp2.classselector = mp1

    # check as_dict method
    assert mp1.as_dict() == {"listparam": [1]}, "`as_dict` method failed"

    # check update method with different parameters
    assert mp2.update(classselector_listparam=[10]).as_dict() == {
        "tupleparam": (2, 2),
        "classselector": {"listparam": [10]},
    }, "magic property setting failed"

    # check wrong attribute name in nested dict
    with pytest.raises(ValueError):
        mp1.update(listparam=dict(tupleparam=10))

    # check match properties=False
    assert mp2.update(
        classselector_listparam=(10,), prop4=4, _match_properties=False
    ).as_dict() == {
        "tupleparam": (2, 2),
        "classselector": {"listparam": [10]},
    }, "magic property setting failed, should ignore `'prop4'`"

    # check replace None only
    mp2.tupleparam = None
    assert mp2.update(
        classselector_listparam=(25,), tupleparam=[1, 1], _replace_None_only=True
    ).as_dict() == {
        "tupleparam": (1, 1),
        "classselector": {"listparam": [10]},
    }, "magic property setting failed, `tupleparam` should be remained unchanged `(1, 1)`"

    # check copy method
    mp3 = mp2.copy()
    assert mp3 is not mp2, "failed copying, should return a different id"
    assert (
        mp3.as_dict() == mp2.as_dict()
    ), "failed copying, should return the same property values"

    # check update with param object
    assert mp2.update(mp3) is mp2

    # check flatten dict
    assert mp3.as_dict(flatten=True) == mp2.as_dict(
        flatten=True
    ), "failed copying, should return the same property values"

    # check failing init
    with pytest.raises(AttributeError):
        MagicParam1(a=0)  # `a` is not a property in the class


def test_get_defaults_dict():
    """test get_defaults_dict"""
    s0 = get_defaults_dict("display.style")
    s1 = get_defaults_dict()["display"]["style"]
    assert s0 == s1, "dicts don't match"
