# ✨ Mystics of Python 🐍

## Exploring the unexplored 🌍 🗺

------------------

#### Image shapes:

+ Numpy / Matplotlib / opencv treats image with shape H,W,C

+ Pytorch requires image to be in thes shape C,H,W

#### * and ** operators

------------------

Below are 6 different use cases for * and ** in python programming:
[source : StackOverflow : https://stackoverflow.com/a/59630576/12988588](https://stackoverflow.com/a/59630576/12988588)

+ __To accept any number of positional arguments using *args:__

```python
def foo(*args):
 pass
```

 here foo accepts any number of positional arguments,
 i. e., the following calls are valid ``foo(1)``, ``foo(1, 'bar')``

+ __To accept any number of keyword arguments using **kwargs:__

```python
def foo(**kwargs):
    pass
```

here 'foo' accepts any number of keyword arguments, 
i. e., the following calls are valid ``foo(name='Tom')``,``foo(name='Tom', age=33)``

+ __To accept any number of positional and keyword arguments using *args, **kwargs:__

```python
def foo(*args, **kwargs):
    pass
```

here foo accepts any number of positional and keyword arguments,
i. e., the following calls are valid ``foo(1,name='Tom')``, ``foo(1, 'bar', name='Tom', age=33)
``

+ __To enforce keyword only arguments using *:__

```python
def foo(pos1, pos2, *, kwarg1):
    pass
```

here * means that foo only accept keyword arguments after pos2, hence foo(1, 2, 3) raises TypeError
but ``foo(1, 2, kwarg1=3)`` is ok.

+ __To express no further interest in more positional arguments using `*_` (Note: this is a convention only):__

```python
def foo(bar, baz, *_):
    pass 
```

means (by convention) foo only uses bar and baz arguments in its working and will ignore others.

+ __To express no further interest in more keyword arguments using `\**_` (Note: this is a convention only):__

```python
def foo(bar, baz, **_):
    pass 
```

means (by convention) foo only uses bar and baz arguments in its working and will ignore others.

+ __BONUS__: From python 3.8 onward, one can use `/` in function definition to enforce positional only parameters.
In the following example, parameters a and b are positional-only, while c or d can be positional or keyword,
and e or f are required to be keywords:

```python
def f(a, b, /, c, d, *, e, f):
    pass
```


