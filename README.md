# stylemod

[![PyPI - Version](https://img.shields.io/pypi/v/stylemod?color=AE81FF)](https://pypi.org/project/stylemod/)
[![PyPI - License](https://img.shields.io/pypi/l/stylemod)](https://github.com/ooojustin/stylemod/blob/main/LICENSE)

Modular [neural style transfer (NST)](https://en.wikipedia.org/wiki/Neural_style_transfer) library designed to make it easy to integrate and customize different deep learning models for artistic style transfer.

## Table of Contents

- [Installation](#installation)
- [Architecture](#modular-architecture)
- [Models](#model-superclasses)
  - [BaseModel](#basemodel)
  - [CNNBaseModel](#cnnbasemodel)
  - [TransformerBaseModel](#transformerbasemodel)
- [Model Factory](#modelfactory)
- [CLI Usage](#cli-usage)
- [License](#license)

### Key Features

- Plug-and-play architecture for integrating new models.
- Support for CNN-based and Transformer-based models.
- Easy customization of style and content loss computation.
- Command-line interface (CLI) for easy interaction.
- Provides out-of-the-box functionality for managing models, utilized layers/weights, normalizations, and more.

### Modular Architecture

Here is a visualization of the class hierarchy for the `stylemod` library:

![Class Hierarchy](https://github.com/ooojustin/stylemod/blob/main/img/class_hierarchy.png?raw=true)

## Installation

### Option 1: Install via PyPI (Recommended)

```bash
pip install stylemod
```

This will automatically install required dependencies.
You can also view the package on [PyPI](https://pypi.org/project/stylemod/).

### Option 2: Install from Source (For Development)

1. **Clone the repository**:

   ```bash
   git clone https://github.com/ooojustin/stylemod.git
   cd stylemod
   ```

2. **Install dependencies**:
   Make sure you have PyTorch and other required libraries installed:

   ```bash
   pip install -r requirements.txt
   ```

### Install Graphviz (Optional)

If you wish to use the built-in Graphviz integration for architecture visualization, ensure Graphviz is installed:

- **Windows**  
  You can download Graphviz for Windows from the official website:  
  [Windows Download](https://graphviz.gitlab.io/_pages/Download/Download_windows.html)

  Alternatively, you can install it using popular package managers:

  ```bash
  # Using Chocolatey
  choco install graphviz

  # Using Scoop
  scoop install graphviz
  ```

- **Unix-based Systems**

  ```bash
  # For Linux (Debian/Ubuntu)
  sudo apt-get install graphviz

  # For Linux (Red Hat/CentOS)
  sudo yum install graphviz

  # For macOS
  brew install graphviz
  ```

> **Note**: If you try to invoke `stylemod.generate_class_hierarchy()` or `model.visualize()` without graphviz installed, stylemod will attempt to install it automatically via your package manager on Linux/MacOS.

## Model Superclasses

In the `stylemod` library, models used for neural style transfer are designed to be modular and extensible. They inherit from two primary classes: `AbstractBaseModel`, which provides a blueprint for all models, and `BaseModel`, which extends `AbstractBaseModel` to provide common functionality for most neural style transfer tasks. Subclasses like `CNNBaseModel` and `TransformerBaseModel` extend `BaseModel` with architecture-specific logic.

### AbstractBaseModel

The `AbstractBaseModel` is an abstract class that defines the required interface for all neural style transfer models. It does not provide any concrete implementations but instead acts as a blueprint to ensure that all models follow a consistent structure. Each model must implement methods for initialization, feature extraction, loss calculation, and visualization.

Below is a table summarizing the key abstract methods that subclasses must implement:

| **Abstract Method**                                                             | **Description**                                                                                                      |
| ------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| `initialize_module()`                                                           | Initializes the model architecture and loads any required weights.                                                   |
| `get_model_module()`                                                            | Returns the initialized model, ensuring that it has been properly set up.                                            |
| `eval()`                                                                        | Switches the model to evaluation mode, disabling training-specific operations (like dropout or batch normalization). |
| `set_device(device)`                                                            | Moves the model to the specified device (CPU/GPU).                                                                   |
| `normalize_tensor(tensor)`                                                      | Normalizes the input tensor according to the model’s pre-defined normalization (if applicable).                      |
| `denormalize_tensor(tensor)`                                                    | Reverts normalization applied to a tensor, returning it to its original scale and distribution.                      |
| `get_features(image, layers)`                                                   | Extracts feature maps from the given image at specified model layers.                                                |
| `calc_gram_matrix(tensor: torch.Tensor)`                                        | Calculates the gram matrix of a tensor, which is used to capture style information in style transfer models.         |
| `calc_content_loss(target, content_features)`                                   | Computes the content loss by comparing the target image's features to the content image’s features.                  |
| `calc_style_loss(target, style_features, *args, **kwargs)`                      | Computes the style loss by comparing the target image's style features with those from the style image.              |
| `forward(target, content_image, style_image, content_features, style_features)` | Combines content and style losses into a single scalar value for optimization.                                       |
| `visualize()`                                                                   | Visualizes the model’s architecture, typically outputting a Graphviz diagram.                                        |

### BaseModel

The `BaseModel` class extends `AbstractBaseModel` by providing core functionality such as model initialization, normalization, feature extraction, and content/style loss computation. This class is designed to reduce repetitive code, allowing subclasses to focus on model-specific logic.

- **Initialization**: The model can be initialized with a callable function (`model_fn`) to load the architecture and optional pre-trained weights.
- **Normalization**: Handles input tensor normalization and denormalization, ensuring consistent image processing.
- **Feature Extraction**: Extracts feature maps from intermediate layers of the model.
- **Gram Matrix Calculation**: Provides a default implementation to calculate gram matrices, used for style transfer tasks.

### CNNBaseModel

The `CNNBaseModel` class extends `BaseModel` to implement style transfer logic specific to Convolutional Neural Networks (CNNs), such as VGG and ResNet. It adds methods for calculating content and style losses based on feature maps extracted from the network.

- **Content Loss**: Calculated as the mean squared difference between the target and content image's feature maps at a specific layer.
- **Style Loss**: Computed by comparing the gram matrices of the style and target images across multiple layers.

### TransformerBaseModel

The `TransformerBaseModel` extends `BaseModel` to support transformer architectures which rely heavily on attention mechanisms. This class introduces functionality for computing and using attention maps in style transfer.

- **Attention Mechanism**: Requires an implementation of `get_attention()`, as the attention mechanism varies across different transformer architectures.
- **Style Loss**: Uses attention-based style loss by comparing the gram matrices of the attention maps for the style and target images.

## CLI Usage

The `stylemod` library provides a command-line interface (CLI) for running style transfer and visualizing model architectures.

### Available Commands

#### 1. Running Style Transfer

You can run the style transfer directly from the CLI by providing the paths to your content and style images, along with other optional parameters like the output filename, the number of steps, and the model to use.

```bash
stylemod run --content-image "content.png" --style-image "style.png" --steps 500 --model VGG19
```

##### Options:

- `-ci, --content-image`: **(Required)** Path to the content image.
- `-si, --style-image`: **(Required)** Path to the style image.
- `-o, --output-image`: Filename for the output image. _(Default: `output_image.png`)_
- `-s, --steps`: Number of optimization steps. _(Default: `1000`)_
- `-ms, --max-size`: Maximum size of input images. _(Default: `400`)_
- `-m, --model`: The model to use for style transfer. _(Default: `VGG19`)_
- `-gpu, --gpu-index`: GPU index to use. _(Default: 0, if available)_

#### 2. Visualizing Model Architecture

You can visualize the architecture of a specific model using the `visualize` command:

```bash
stylemod visualize VGG19 --output "model_visualization.png" --dpi 300
```

##### Options:

- `{model name}`: The model architecture to visualize.
- `-o, --output`: Optional path to save the visualization image (e.g., `model_vis.png`).
- `-d, --dpi`: Set the DPI (dots per inch) for the rendered image. _(Default: `400`)_

> To see an example of what the output of visualizing VGG19 would look like, see [visualize_vgg19.png](https://github.com/ooojustin/stylemod/blob/main/img/visualize_vgg19.png).

#### 3. Visualizing Class Hierarchy

You can visualize the class hierarchy of the `stylemod` library, which shows the relationships between different model classes.

```bash
stylemod class-hierarchy --save --show-funcs
```

##### Options:

- `-s, --save`: Save the rendered class hierarchy to a file (`img/class_hierarchy.png`).
- `-f, --show-funcs`: Show the abstract functions that should be implemented by subclasses.
- `-d, --dpi`: Set the DPI (dots per inch) for the rendered image. _(Default: `200`)_

## ModelFactory

The `ModelFactory` class is responsible for dynamically creating instances of models used in the `stylemod` library. It provides a flexible and extensible mechanism for handling different model architectures and implementation without needing to hard-code their instantiations.

The `ModelFactory` automatically registers any model that extends `AbstractBaseModel` found in the `stylemod.models` package. Additional models can be registered manually if needed.

#### Key Features:

- **Dynamic Model Creation**: Allows creating model instances by name or enum value, where `**kwargs` are forwarded to the constructor via the `create()` method.
- **Automatic Model Registration**: Automatically scans and registers all models that inherit from `AbstractBaseModel`.
- **Model Registry**: Maintains a registry of available models and their corresponding classes.
- **Custom Model Registration**: Allows registering custom models by name.

#### Factory Methods:

| **Method**                                                        | **Description**                                                                                                               |
| ----------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `create(model: Union[str, Model], **kwargs)`                      | Creates and returns an instance of a registered model. Accepts either a string representing the model name or a `Model` enum. |
| `register(model_name: str, model_class: Type[AbstractBaseModel])` | Registers a new model to the factory by name. If a model with the same name is already registered, an error is raised.        |
| `get_models()`                                                    | Returns a list of all registered model classes.                                                                               |
| `_register_models()`                                              | Scans the `stylemod.models` package and automatically registers all classes inheriting from `AbstractBaseModel`.              |

#### Example Usage:

```python
from stylemod.core.factory import ModelFactory

# Create a model by its enum name (assuming Model.VGG19 is registered)
model = ModelFactory.create("VGG19", content_layer="conv4_2", style_weights={"conv1_1": 1.0})

# Alternatively, create a model by passing a Model enum
from stylemod.models import Model
model = ModelFactory.create(Model.VGG19, content_layer="conv4_2", style_weights={"conv1_1": 1.0})

# Register a custom model
class MyCustomModel(BaseModel):
  ...

ModelFactory.register("MY_CUSTOM_MODEL", MyCustomModel)

# Create an instance of the custom model
custom_model = ModelFactory.create("MY_CUSTOM_MODEL", content_layer='conv4_2', style_weights={'conv1_1': 1.0})
```

## License

stylemod is licensed under the MIT License. See the [LICENSE](https://github.com/ooojustin/stylemod/blob/main/LICENSE) file for details.
