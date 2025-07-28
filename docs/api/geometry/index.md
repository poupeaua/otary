# Geometry

The `geometry` module provides a comprehensive set of tools for working with geometric shapes and entities.

## Geometric Objects

The `geometry` module is organized into two main categories `discrete` and `continuous`.
Each category contains other sub-categories like shape, linear, etc.

- **Discrete:** Represents geometric entities that are defined by a finite set of points:
    - **[Point](discrete/point.md)**
    - **[Segment](discrete/linear/segment.md)**
    - **[Linear Spline](discrete/linear/linear_spline.md)**
    - **[Vector](discrete/linear/directed/vector.md)**
    - **[Vectorized Linear Spline](discrete/linear/directed/vectorized_linear_spline.md)**
    - **[Polygon](discrete/shape/polygon.md)**
    - **[Rectangle](discrete/shape/rectangle.md)**
    - **[Triangle](discrete/shape/triangle.md)**

- **Continuous:** Represents geometric entities that are defined by continuous mathematical functions:
    - **[Circle](continuous/circle.md)**
    - **[Ellipse](continuous/ellipse.md)**

## Available Modules

Below is a list of available modules and their functionalities:

### Base Geometry

::: otary.geometry.entity

### Continuous Geometry

::: otary.geometry.continuous.entity

#### Shape

::: otary.geometry.continuous.shape.circle
::: otary.geometry.continuous.shape.ellipse

### Discrete Geometry

::: otary.geometry.discrete.entity
::: otary.geometry.discrete.point

#### Shape

::: otary.geometry.discrete.shape.polygon
::: otary.geometry.discrete.shape.rectangle
::: otary.geometry.discrete.shape.triangle

#### Linear

::: otary.geometry.discrete.linear.entity
::: otary.geometry.discrete.linear.segment
::: otary.geometry.discrete.linear.linear_spline

##### Linear Directed

::: otary.geometry.discrete.linear.directed.entity
::: otary.geometry.discrete.linear.directed.vector
::: otary.geometry.discrete.linear.directed.vectorized_linear_spline
