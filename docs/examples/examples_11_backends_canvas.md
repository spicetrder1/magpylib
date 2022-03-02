---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.6
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

(examples-backends-canvas)=

# Backend and canvas

(examples-backend)=

## Graphic backend

Magpylib supports Matplotlib and Plotly as possible graphic backends.
If a backend is not specified, the library default stored in `magpy.defaults.display.backend` will be used.
The value can bei either `'matplotlib'` or `'plotly'`.

To select a graphic backend one can
1. change the library default with the command `magpy.defaults.display.backend = 'plotly'`.
2. set the `backend` kwarg in the `show` function, `show(..., backend='matplotlib')`

```{note}
There is a high level of **feature parity** between the two backends but there are also some key differences, e.g. when displaying magnetization of an object or when animating. In addition, some common Matplotlib syntax (e.g. color `'r'`, linestyle `':'`) is automatically translated to Plotly and vice versa.
```

The following example shows first Matplotlib and then Plotly output:

```{code-cell} ipython3
import numpy as np
import magpylib as magpy

# define sources and paths
src1 = magpy.current.Loop(current=1, diameter=1, position=(0,0,-3))
src2 = magpy.magnet.Cylinder(magnetization=(0,-100,0), dimension=(1,2), position=(0,-3,0))

src1.move(np.linspace((0,0,0.1), (0,0,6), 20))
src2.rotate_from_angax(np.linspace(0, 300, 50), 'z', anchor=0)

# display the system with both backends
magpy.show(src1, src2)
magpy.show(src1, src2, backend='plotly')
```

(examples-canvas)=

## Output in own figure

When calling `show`, a Matplotlib or Plotly figure is automatically generated and displayed. It is also possible to display the `show` output on a given user-defined canvas (Plotly `Figure` object or Matplotlib `Axis3d` object) by using the `canvas` kwarg.

In the following example we show how to combine a 2D plot with the 3D `show` output in **Matplotlib**:

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
import magpylib as magpy

# setup matplotlib figure and subplots
fig = plt.figure(figsize=(12, 4))
ax1 = fig.add_subplot(121,)                  # 2D-axis
ax2 = fig.add_subplot(122, projection="3d")  # 3D-axis

# define sources and paths
src1 = magpy.current.Loop(current=-1, diameter=2, position=(0,0,-4))
src2 = magpy.magnet.Cylinder(magnetization=(0,-100,0), dimension=(1,2), position=(0,-3,0))

src1.move(np.linspace((0,0,0.1), (0,0,8), 40))
src2.rotate_from_angax(np.linspace(0, 300, 40), 'z', anchor=0)

# compute field and plot in 2D-axis
B = magpy.getB([src1, src2], (0,0,0), sumup=True)
ax1.plot(B)

# display show() output in 3D-axis
magpy.show(src1, src2, canvas=ax2)

# generate figure
plt.tight_layout()
plt.show()
```

In the next example we show how to combine a 2D plot with the 3D `show` output in **Plotly**:

```{code-cell} ipython3
import numpy as np
import plotly.graph_objects as go
import magpylib as magpy

# setup plotly figure and subplots
fig = go.Figure().set_subplots(rows=1, cols=2, specs=[[{"type": "xy"}, {"type": "scene"}]])

# define sources and paths
src1 = magpy.current.Loop(current=-1, diameter=2, position=(0,0,-4))
src2 = magpy.magnet.Cylinder(magnetization=(0,-100,0), dimension=(1,2), position=(0,-3,0))

src1.move(np.linspace((0,0,0), (0,0,8), 41)[1:])
src2.rotate_from_angax(np.linspace(0, 300, 41)[1:], 'z', anchor=0)

# compute field and plot in 2D-axis
B = magpy.getB([src1, src2], (0,0,0), sumup=True)
for i in range(3):
    fig.add_trace(go.Scatter(x=np.linspace(0,1,40), y=B[:,i]))

# display show() output in 3D-axis
temp_fig = go.Figure()
magpy.show(src1, src2, canvas=temp_fig, backend='plotly')
fig.add_traces(temp_fig.data, rows=1, cols=2)
fig.layout.scene3 = temp_fig.layout.scene

# generate figure
fig.show()
```
