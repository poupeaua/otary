# API

Otary is built upon two core modules **Image** and **Geometry**, each designed with a distinct architectural pattern to provide flexibility and power.

## [Image: A Composition-Based Approach](image/index.md)

The `image` module is designed following a **composition over inheritance** principle. This allows you to build complex image processing pipelines by combining smaller, independent, and reusable components. Instead of inheriting properties from a monolithic class, you can dynamically assemble functionality, leading to a more flexible and maintainable codebase.

For example, an image object can be composed of a `Reader` to load data, a `Transformer` to apply modifications, and a `Drawer` to add overlays, all working together seamlessly.

## [Geometry: An Inheritance-Based Structure](geometry/index.md)

The `geometry` module uses a more traditional **inheritance** model. It provides a clear and hierarchical structure for geometric entities. Base classes define common behaviors, and specialized subclasses inherit and extend this functionality. This approach is ideal for creating a well-defined and logical classification of shapes and geometric objects.
