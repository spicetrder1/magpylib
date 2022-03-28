"""Collection class code"""

from collections import Counter
from magpylib._src.utility import (
    format_obj_input,
    LIBRARY_SENSORS,
    LIBRARY_SOURCES,
    rec_obj_remover,
)

from magpylib._src.obj_classes.class_BaseGeo import BaseGeo
from magpylib._src.obj_classes.class_BaseDisplayRepr import BaseDisplayRepr
from magpylib._src.fields.field_wrap_BH_level2 import getBH_level2
from magpylib._src.defaults.defaults_utility import validate_style_keys
from magpylib._src.exceptions import MagpylibBadUserInput
from magpylib._src.input_checks import check_format_input_obj


def repr_obj(obj, desc="type+id+label"):
    """Returns obj repr based on description paramter string"""
    rp = ""
    lbl = "label" in desc and getattr(getattr(obj, "style", False), "label", False)
    if "type" in desc or not lbl:
        rp += f"{type(obj).__name__}"
    if lbl:
        rp += f" {obj.style.label}"
    if "id" in desc or not lbl:
        id_str = f"id={id(obj)}"
        rp += f" ({id_str})" if rp else id_str
    return rp.strip()


def collection_tree_generator(
    dir_child,
    prefix="",
    space="    ",
    branch="│   ",
    tee="├── ",
    last="└── ",
    desc="type+id+label",
    max_elems=20,
    properties=False,
):
    """A recursive generator, given a collection child object
    will yield a visual tree structure line by line
    with each line prefixed by the same characters
    """
    # pylint: disable=protected-access
    # contents each get pointers that are ├── with a final └── :
    contents = []
    children = getattr(dir_child, "children", [])
    desc_func = getattr(dir_child, "_get_description", False)
    props = []
    if properties and desc_func:
        desc_out = desc_func(
            exclude=("children", "parent", "style", "sources", "sensors", "collections")
        )
        props = [d.strip() for d in desc_out[1:]]
    if len(children) > max_elems:
        counts = Counter([c._object_type for c in children])
        children = [f"{v}x {k}s" for k, v in counts.items()]
    contents.extend(props)
    contents.extend(children)
    pointers = [tee] * (len(contents) - 1) + [last]
    pointers[: len(props)] = [branch if children else space] * len(props)
    for pointer, child in zip(pointers, contents):
        child_repr = child if isinstance(child, str) else repr_obj(child, desc)
        yield prefix + pointer + child_repr
        if getattr(child, "children", False) or (
            getattr(dir_child, "_get_description", False) and properties
        ):  # extend the prefix and recurse:
            extension = branch if pointer == tee else space
            # i.e. space because last, └── , above so no more |
            yield from collection_tree_generator(
                child,
                prefix=prefix + extension,
                space=space,
                branch=branch,
                tee=tee,
                last=last,
                desc=desc,
                max_elems=max_elems,
                properties=properties,
            )


class BaseCollection(BaseDisplayRepr):
    """Collection base class without BaseGeo properties"""

    def __init__(self, *children, override_parent=False):

        self._object_type = "Collection"

        BaseDisplayRepr.__init__(self)

        self._children = []
        self._sources = []
        self._sensors = []
        self._collections = []
        self.add(*children, override_parent=override_parent)

    # property getters and setters
    @property
    def children(self):
        """An ordered list of all children in the collection."""
        return self._children

    @children.setter
    def children(self, children):
        """Set Collection children."""
        # pylint: disable=protected-access
        for child in self._children:
            child._parent = None
        self._children = []
        self.add(*children)

    @property
    def sources(self):
        """An ordered list of all source objects in the collection."""
        return self._sources

    @sources.setter
    def sources(self, sources):
        """Set Collection sources."""
        # pylint: disable=protected-access
        new_children = []
        for child in self._children:
            if child in self._sources:
                child._parent = None
            else:
                new_children.append(child)
        self._children = new_children
        src_list = format_obj_input(sources, allow="sources")
        self.add(*src_list)

    @property
    def sensors(self):
        """An ordered list of all sensor objects in the collection."""
        return self._sensors

    @sensors.setter
    def sensors(self, sensors):
        """Set Collection sensors."""
        # pylint: disable=protected-access
        new_children = []
        for child in self._children:
            if child in self._sensors:
                child._parent = None
            else:
                new_children.append(child)
        self._children = new_children
        sens_list = format_obj_input(sensors, allow="sensors")
        self.add(*sens_list)

    @property
    def collections(self):
        """An ordered list of all collection objects in the collection."""
        return self._collections

    @collections.setter
    def collections(self, collections):
        """Set Collection collections."""
        # pylint: disable=protected-access
        new_children = []
        for child in self._children:
            if child in self._collections:
                child._parent = None
            else:
                new_children.append(child)
        self._children = new_children
        coll_list = format_obj_input(collections, allow="collections")
        self.add(*coll_list)

    # dunders
    def __iter__(self):
        yield from self._children

    def __getitem__(self, i):
        return self._children[i]

    def __len__(self):
        return len(self._children)

    def _repr_html_(self):
        lines = []
        lines.append(repr_obj(self))
        for line in collection_tree_generator(
            self, desc="type+label+id", max_elems=10, properties=False
        ):
            lines.append(line)
        return f"""<pre>{'<br>'.join(lines)}</pre>"""

    def describe(self, *,  desc="type+label+id", max_elems=10, properties=False):
        # pylint: disable=arguments-differ
        """Returns a tree view of the nested collection elements.

        Parameters
        ----------
        desc: bool, default="type+label+id"
            Object description.
        max_elems:
            If number of children at any level is higher than `max_elems`, elements are replaced by
            counters by object type.
        properties: bool, default=False
            If True, adds object properties to the view
        """
        print(repr_obj(self, desc))
        for line in collection_tree_generator(
            self, desc=desc, max_elems=max_elems, properties=properties
        ):
            print(line)

    # methods -------------------------------------------------------
    def add(self, *children, override_parent=False):
        """Add sources, sensors or collections.

        Parameters
        ----------
        children: sources, sensors or collections
            Add arbitrary sources, sensors or other collections to this collection.

        override_parent: bool, default=`True`
            Accept objects as children that already have parents. Automatically
            removes such objects from previous parent collection.

        Returns
        -------
        self: `Collection` object

        Examples
        --------
        In this example we add a sensor object to a collection:

        >>> import magpylib as magpy
        >>> x1 = magpy.Sensor(style_label='x1')
        >>> coll = magpy.Collection(x1, style_label='coll')
        >>> coll.describe(labels=True)
        coll
        └── x1

        >>> x2 = magpy.Sensor(style_label='x2')
        >>> coll.add(x2)
        >>> coll.describe(labels=True)
        coll
        ├── x1
        └── x2
        """
        # pylint: disable=protected-access
        # check and format input
        obj_list = check_format_input_obj(
            children,
            allow="sensors+sources+collections",
            recursive=False,
            typechecks=True,
        )

        # assign parent
        for obj in obj_list:
            if obj._parent is None:
                obj._parent = self
            elif override_parent:
                obj._parent.remove(obj)
                obj._parent = self
            else:
                raise MagpylibBadUserInput(
                    f"Cannot add {obj!r} to {self!r} because it already has a parent."
                    "Consider using `override_parent=True`."
                )

        # set attributes
        self._children += obj_list
        self._update_src_and_sens()

        return self

    def _update_src_and_sens(self):
        # pylint: disable=protected-access
        """updates sources, sensors and collections attributes from children"""
        self._sources = [
            obj for obj in self._children if obj._object_type in LIBRARY_SOURCES
        ]
        self._sensors = [
            obj for obj in self._children if obj._object_type in LIBRARY_SENSORS
        ]
        self._collections = [
            obj for obj in self._children if obj._object_type == "Collection"
        ]

    def remove(self, *children, recursive=True, errors="raise"):
        """Remove children from the collection tree.

        Parameters
        ----------
        children: child objects
            Remove the given children from the collection.

        recursive: bool, default=`True`
            Remove children also when they are in child collections.

        errors: str, default=`'raise'`
            Can be `'raise'` or `'ignore'` to toggle error output when child is
            not found for removal.

        Returns
        -------
        self: `Collection` object

        Examples
        --------
        In this example we remove a child from a Collection:

        >>> import magpylib as magpy
        >>> x1 = magpy.Sensor(style_label='x1')
        >>> x2 = magpy.Sensor(style_label='x2')
        >>> col = magpy.Collection(x1, x2, style_label='col')
        >>> col.describe(labels=True)
        col
        ├── x1
        └── x2

        >>> col.remove(x1)
        >>> col.describe(labels=True)
        col
        └── x2
        """
        # pylint: disable=protected-access

        # check and format input
        remove_objects = check_format_input_obj(
            children,
            allow="sensors+sources+collections",
            recursive=False,
            typechecks=True,
        )
        self_objects = check_format_input_obj(
            self,
            allow="sensors+sources+collections",
            recursive=recursive,
        )
        for child in remove_objects:
            if child in self_objects:
                rec_obj_remover(self, child)
                child._parent = None
            else:
                if errors == "raise":
                    raise MagpylibBadUserInput(
                        f"Cannot find and remove {child} from {self}."
                    )
                if errors != "ignore":
                    raise MagpylibBadUserInput(
                        "Input `errors` must be one of ('raise', 'ignore').\n"
                        f"Instead received {errors}."
                    )
        return self

    def set_children_styles(self, arg=None, recursive=True, _validate=True, **kwargs):
        """Set display style of all children in the collection. Only matching properties
        will be applied.

        Parameters
        ----------
        arg: style dictionary or style underscore magic input
            Style arguments to be applied.

        recursive: bool, default=`True`
            Apply styles also to children of child collections.

        Returns
        -------
        self: `Collection` object

        Examples
        --------
        In this example we start by creating a collection from three sphere magnets:

        >>> import magpylib as magpy
        >>> col = magpy.Collection()
        >>> for i in range(3):
        >>>     col = col + magpy.magnet.Sphere((0,0,1), 1, position=(i,0,0))

        We apply styles using underscore magic for magnetization vector size and a style
        dictionary for the color.

        >>> col.set_children_styles(magnetization_size=0.5)
        >>> col.set_children_styles({'color':'g'})

        Finally we create a separate sphere magnet to demonstrate the default style and display
        the collection and the separate magnet with Matplotlib:

        >>> src = magpy.magnet.Sphere((0,0,1), 1, position=(3,0,0))
        >>> magpy.show(col, src)
        ---> graphic output
        """
        # pylint: disable=protected-access

        if arg is None:
            arg = {}
        if kwargs:
            arg.update(kwargs)
        style_kwargs = arg
        if _validate:
            style_kwargs = validate_style_keys(arg)

        for child in self._children:
            # match properties false will try to apply properties from kwargs only if it finds it
            # without throwing an error
            if child._object_type == "Collection" and recursive:
                self.__class__.set_children_styles(child, style_kwargs, _validate=False)
            style_kwargs_specific = {
                k: v
                for k, v in style_kwargs.items()
                if k.split("_")[0] in child.style.as_dict()
            }
            child.style.update(**style_kwargs_specific, _match_properties=True)
        return self

    def _validate_getBH_inputs(self, *inputs):
        """validate Collection.getBH inputs"""
        # pylint: disable=protected-access
        # pylint: disable=too-many-branches
        current_sources = format_obj_input(self, allow="sources")
        current_sensors = format_obj_input(self, allow="sensors")
        if current_sensors and current_sources:
            sources, sensors = self, self
            if inputs:
                raise MagpylibBadUserInput(
                    "Collections with sensors and sources do not allow `collection.getB()` inputs."
                    "Consider using `magpy.getB()` instead."
                )
        elif not current_sources:
            sources, sensors = inputs, self
        elif not current_sensors:
            sources, sensors = self, inputs
        return sources, sensors

    def getB(self, *inputs, squeeze=True):
        """Compute B-field in [mT] for given sources and observer inputs.

        Parameters
        ----------
        sources_observers: source or observer inputs
            Input can only be observers if the collection contains only sources. In this case the
            collection behaves like a single source.
            Input can only be sources if the collection contains only sensors. In this case the
            collection behaves like a list of all its sensors.

        squeeze: bool, default=`True`
            If `True`, the output is squeezed, i.e. all axes of length 1 in the output (e.g.
            only a single source) are eliminated.

        Returns
        -------
        B-field: ndarray, shape squeeze(m, k, n1, n2, ..., 3)
            B-field at each path position (m) for each sensor (k) and each sensor pixel
            position (n1,n2,...) in units of [mT]. Sensor pixel positions are equivalent
            to simple observer positions. Paths of objects that are shorter than m will be
            considered as static beyond their end.

        Examples
        --------
        In this example we create a collection from two sources and two sensors:

        >>> import magpylib as magpy
        >>> src1 = magpy.magnet.Sphere((0,0,1000), 1)
        >>> src2 = src1.copy()
        >>> sens1 = magpy.Sensor(position=(0,0,1))
        >>> sens2 = sens1.copy()
        >>> col = src1 + src2 + sens1 + sens2

        The following computations all give the same result:

        >>> B = col.getB()
        >>> B = magpy.getB(col, col)
        >>> B = magpy.getB(col, [sens1, sens2])
        >>> B = magpy.getB([src1, src2], col)
        >>> B = magpy.getB([src1, src2], [sens1, sens2])
        >>> print(B)
        [[  0.           0.         166.66666667]
         [  0.           0.         166.66666667]]
        """

        sources, sensors = self._validate_getBH_inputs(*inputs)
        return getBH_level2(sources, sensors, sumup=False, squeeze=squeeze, field="B")

    def getH(self, *inputs, squeeze=True):
        """Compute H-field in [kA/m] for given sources and observer inputs.

        Parameters
        ----------
        sources_observers: source or observer inputs
            Input can only be observers if the collection contains only sources. In this case the
            collection behaves like a single source.
            Input can only be sources if the collection contains sensors. In this case the
            collection behaves like a list of all its sensors.

        squeeze: bool, default=`True`
            If `True`, the output is squeezed, i.e. all axes of length 1 in the output (e.g.
            only a single source) are eliminated.

        Returns
        -------
        H-field: ndarray, shape squeeze(m, k, n1, n2, ..., 3)
            H-field at each path position (m) for each sensor (k) and each sensor pixel
            position (n1,n2,...) in units of [kA/m]. Sensor pixel positions are equivalent
            to simple observer positions. Paths of objects that are shorter than m will be
            considered as static beyond their end.

        Examples
        --------
        In this example we create a collection from two sources and two sensors:

        >>> import magpylib as magpy
        >>> src1 = magpy.magnet.Sphere((0,0,1000), 1)
        >>> src2 = src1.copy()
        >>> sens1 = magpy.Sensor(position=(0,0,1))
        >>> sens2 = sens1.copy()
        >>> col = src1 + src2 + sens1 + sens2

        The following computations all give the same result:

        >>> H = col.getH()
        >>> H = magpy.getH(col, col)
        >>> H = magpy.getH(col, [sens1, sens2])
        >>> H = magpy.getH([src1, src2], col)
        >>> H = magpy.getH([src1, src2], [sens1, sens2])
        >>> print(H)
        [[  0.           0.         66.31455962]
         [  0.           0.         66.31455962]]
        """

        sources, sensors = self._validate_getBH_inputs(*inputs)

        return getBH_level2(sources, sensors, sumup=False, squeeze=squeeze, field="H")


class Collection(BaseGeo, BaseCollection):
    """Group multiple children (sources, sensors and collections) in a collection for
    common manipulation.

    Collections span a local reference frame. All objects in a collection are held to
    that reference frame when an operation (e.g. move, rotate, setter, ...) is applied
    to the collection.

    Collections can be used as `sources` and `observers` input for magnetic field
    computation. For magnetic field computation a collection that contains sources
    functions like a single source. When the collection contains sensors
    it functions like a list of all its sensors.

    Parameters
    ----------
    children: sources, `Sensor` or `Collection objects
        An ordered list of all children in the collection.

    sensors: `Sensor` objects
        An ordered list of all sensor objects in the collection.

    sources: source objects
        An ordered list of all source objects`(magnets, currents, misc) in the collection.

    collections: `Collection` objects
        An ordered list of all collection objects in the collection.

    position: array_like, shape (3,) or (m,3), default=`(0,0,0)`
        Object position(s) in the global coordinates in units of [mm]. For m>1, the
        `position` and `orientation` attributes together represent an object path.

    orientation: scipy `Rotation` object with length 1 or m, default=`None`
        Object orientation(s) in the global coordinates. `None` corresponds to
        a unit-rotation. For m>1, the `position` and `orientation` attributes
        together represent an object path.

    parent: `Collection` object or `None`
        The object is a child of it's parent collection.

    style: dict
        Object style inputs must be in dictionary form, e.g. `{'color':'red'}` or
        using style underscore magic, e.g. `style_color='red'`.

    Returns
    -------
    collection: `Collection` object

    Examples
    --------
    Collections function as groups of multiple magpylib objects. In this example
    we create a collection with two sources and move the whole collection:

    >>> import magpylib as magpy
    >>> src1 = magpy.magnet.Sphere((1,2,3), 1, position=(2,0,0))
    >>> src2 = magpy.current.Loop(1, 1, position=(-2,0,0))
    >>> col = magpy.Collection(src1, src2)
    >>> col.move(((0,0,2)))
    >>> print(src1.position)
    >>> print(src2.position)
    >>> print(col.position)
    [2. 0. 2.]
    [-2.  0.  2.]
    [0. 0. 2.]

    We can still directly access individual objects by name and by index:

    >>> src1.move((2,0,0))
    >>> col[1].move((-2,0,0))
    >>> print(src1.position)
    >>> print(src2.position)
    >>> print(col.position)
    [4. 0. 2.]
    [-4.  0.  2.]
    [0. 0. 2.]

    The field can be computed at position (0,0,0) as if the collection was a single source:

    B = col.getB((0,0,0))
    print(B)
    [ 0.00126232 -0.00093169 -0.00034448]

    We add a sensor at position (0,0,0) to the collection:

    >>> sens = magpy.Sensor()
    >>> col.add(sens)
    >>> print(col.children)
    [Sphere(id=2236606344304), Loop(id=2236606344256), Sensor(id=2236606343584)]

    and can compute the field of the sources in the collection seen by the sensor with
    a single command:

    B = col.getB()
    print(B)
    [ 0.00126232 -0.00093169 -0.00034448]
    """

    def __init__(
        self,
        *args,
        position=(0, 0, 0),
        orientation=None,
        style=None,
        override_parent=False,
        **kwargs,
    ):
        BaseGeo.__init__(
            self,
            position=position,
            orientation=orientation,
            style=style,
            **kwargs,
        )
        BaseCollection.__init__(self, *args, override_parent=override_parent)
