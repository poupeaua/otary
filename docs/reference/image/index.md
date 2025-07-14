# Image

The `image` module provides a flexible and powerful way to work with images using a composition-based design. This allows you to dynamically build image objects with the exact functionality you need.

## Core Components

The `image` module is built around the following core components:

- **Image:** The main class that represents an image. It is composed of other components to provide its functionality.
- **Reader:** Responsible for loading image data from various sources.
- **Drawer:** Provides methods for drawing shapes and text on the image.
- **Transformer:** Allows you to apply various transformations to the image, such as resizing, cropping, and color adjustments.

## Available Modules

Below is a list of available modules and their functionalities:

::: otary.image.base
::: otary.image.components.drawer.drawer
::: otary.image.components.io.reader
::: otary.image.components.io.writer
::: otary.image.components.transformer.transformer
