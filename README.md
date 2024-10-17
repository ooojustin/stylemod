# stylemod

Modular [neural style transfer (NST)](https://en.wikipedia.org/wiki/Neural_style_transfer) library designed to make it easy to integrate and customize different deep learning models for artistic style transfer.

## Table of Contents

- [Installation](#installation)
- [Architecture](#modular-architecture)
- [Models](#model-superclasses)
  - [BaseModel](#basemodel)
  - [CNNBaseModel](#cnnbasemodel)
  - [TransformerBaseModel](#transformerbasemodel)
- [ModelFactory](#modelfactory)
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

![Class Hierarchy](./img/class_hierarchy.png)

## Installation

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

3. **Install Graphviz** (Optional):
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

   > **Note**: If you try to invoke `stylemod.generate_class_hierarchy` without graphviz installed, stylemod will attempt to install it automatically via your package manager on Linux/MacOS.

## Model Superclasses

In the `stylemod` library, models used for neural style transfer are designed to be modular and extensible. They inherit from two primary classes: `AbstractBaseModel`, which provides a blueprint for all models, and `BaseModel`, which extends `AbstractBaseModel` to provide common functionality for most neural style transfer tasks. Subclasses like `CNNBaseModel` and `TransformerBaseModel` extend `BaseModel` with architecture-specific logic.

### AbstractBaseModel

The `AbstractBaseModel` is an abstract class that defines the required interface for all neural style transfer models. It does not provide any concrete implementations but instead acts as a blueprint to ensure that all models follow a consistent structure. Each model must implement methods for initialization, feature extraction, loss calculation, and visualization.

Below is a table summarizing the key abstract methods that subclasses must implement:

| **Abstract Method**                                                                                                 | **Description**                                                                                                      |
| ------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| `initialize_module()`                                                                                               | Initializes the model architecture and loads any required weights.                                                   |
| `get_model_module()`                                                                                                | Returns the initialized model, ensuring that it has been properly set up.                                            |
| `eval()`                                                                                                            | Switches the model to evaluation mode, disabling training-specific operations (like dropout or batch normalization). |
| `set_device(device: torch.device)`                                                                                  | Moves the model to the specified device (CPU/GPU).                                                                   |
| `normalize_tensor(tensor: torch.Tensor)`                                                                            | Normalizes the input tensor according to the model’s pre-defined normalization (if applicable).                      |
| `denormalize_tensor(tensor: torch.Tensor)`                                                                          | Reverts normalization applied to a tensor, returning it to its original scale and distribution.                      |
| `get_features(image: torch.Tensor, layers: List[str])`                                                              | Extracts feature maps from the given image at specified model layers.                                                |
| `calc_gram_matrix(tensor: torch.Tensor)`                                                                            | Calculates the gram matrix of a tensor, which is used to capture style information in style transfer models.         |
| `calc_content_loss(target: torch.Tensor, content_features: Dict[str, torch.Tensor])`                                | Computes the content loss by comparing the target image's features to the content image’s features.                  |
| `calc_style_loss(target: torch.Tensor, style_features: Dict[str, torch.Tensor], *args, **kwargs)`                   | Computes the style loss by comparing the target image's style features with those from the style image.              |
| `forward(target: torch.Tensor, content_features: Dict[str, torch.Tensor], style_features: Dict[str, torch.Tensor])` | Combines content and style losses into a single scalar value for optimization.                                       |
| `visualize()`                                                                                                       | Visualizes the model’s architecture, typically outputting a Graphviz diagram.                                        |

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

stylemod also includes a command-line interface to perform style transfer and visualize the projects class hierarchy.

- ### Running Style Transfer from CLI

  ```bash
  python -m stylemod run --content-image "img/content.jpg" --style-image "img/style.jpg" --model VGG19
  ```

- ### Visualizing Class Hierarchy

  ```bash
  python -m stylemod class-hierarchy --save --show-funcs
  ```

- ### Visualizing Model Architecture

  To visualize the architecture of a specific model:

  ```bash
  python -m stylemod visualize VGG19 --dpi 200 --output model_vis.png
  ```

  - Replace `VGG19` with any supported model name.
  - Use the `--output` option to save the visualization as an image file (e.g., `model_vis.png`). If not provided, the visualization will be displayed without saving.
  - You can adjust the DPI of the visualization using the `--dpi` option.

> To see an example of what the output of visualizing VGG19 would look like, see [visualize_vgg19.png](./img/visualize_vgg19.png).

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

stylemod is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.
