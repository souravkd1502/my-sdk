"""
# ImageHandler Module

This module provides a robust interface for handling image-based tasks using OpenAI's API. The `ImageHandler` class encapsulates functionalities such as:

1. **Question Answering (QA) from Images**: Users can ask questions about one or more images and receive detailed responses.
2. **Image Generation**: Create images based on descriptive prompts using OpenAI's image generation models.

## Features
- Supports multiple image inputs (local or URL-based) for QA tasks.
- Utilizes a variety of OpenAI models for specific tasks.
- Encodes local image files to Base64 format for seamless API interaction.
- Provides customizable parameters such as token limits and output formats.

## Dependencies
- **OpenAI Python Library**: The primary dependency for API interactions.
- **Base64**: For encoding local image files.
- **Logging**: Used extensively for error reporting and debugging.
- **Typing**: For type hinting support, ensuring code clarity and robustness.

## Error Handling
This module includes detailed error handling for various failure scenarios:
- Invalid model or API key inputs.
- Missing or incorrect image paths.
- Failures in API responses.
- Missing dependencies such as the OpenAI library.

## Example Usage
```python
from image_handler import ImageHandler

# Initialize the handler
handler = ImageHandler(model="gpt-4o", api_key="your-api-key")

# Perform QA on an image
response = handler.qa_from_images(
    image_inputs="path/to/image.jpg",
    question="What objects are in this image?"
)
print(response)

# Generate an image from a prompt
images = handler.image_generation(
    prompt="A futuristic cityscape at sunset",
    n=2
)
print(images)
```

This module simplifies interaction with OpenAI's image-related capabilities, offering an intuitive interface for developers.

TODO:
1. Add docstrings for each method and the module 
2. Image generation - generating images from a given prompt (DONE)
3. Image-text-to-text - querying an image for text based on a given prompt (DONE)

FIXME:

Author:
Sourav Das

Date:
2025-07-01
"""

# Adding directories to system path to allow importing custom modules
import sys

sys.path.append("./")
sys.path.append("../")

# Importing necessary libraries and modules
import base64
import logging
from openai import OpenAI

from typing import List, Union, Literal

# Set up logging
_logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s - line: %(lineno)d",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class ImageHandler:
    """
    A wrapper class for the OpenAI API to handle tasks such as:

    1. Question-answering (QA) from images - asking questions about images.
    2. Image generation - generating images from a given prompt.
    """

    def __init__(
        self,
        model: str,
        api_key: str,
    ) -> None:
        """
        Initialize the ImageHandler with model details and API credentials.

        Args:
            model (str): The name of the OpenAI model to use for image generation.
            api_key (str): The API key to use for authentication.

        Raises:
            ValueError: If an unsupported model name is provided.
            ValueError: If the API key is invalid.
            ImportError: If the OpenAI library is not installed.
        """

        # Assign parameters to instance variables
        self.model = model
        self.api_key = api_key

        # List of supported models
        self.supported_models = [
            "gpt-4o",
            "gpt-4o-mini",
            "o1",
            "o1-preview",
            "o1-mini",
        ]

        # Validate the model
        if self.model not in self.supported_models:
            _logger.error(f"Unsupported model: {self.model}")
            raise ValueError(
                f"Unsupported model. Provided: {self.model}. Supported: {self.supported_models}"
            )

        # Validate the API key
        if not self.api_key or not self.api_key.startswith("sk"):
            _logger.error("Invalid API key provided.")
            raise ValueError("Invalid API key. Please provide a valid OpenAI API key.")
        
        self.client = OpenAI(api_key=self.api_key)

    def _encode_image(self, image_path: str) -> str:
        """
        Encodes an image from the local file system into a base64 string.

        Args:
            image_path (str): Path to the local image file.

        Returns:
            str: Base64-encoded string of the image.

        Raises:
            FileNotFoundError: If the image file is not found at the provided path.
            ValueError: If the file is not a valid image.
        """
        try:
            with open(image_path, "rb") as image_file:
                image_data = image_file.read()
                return base64.b64encode(image_data).decode("utf-8")
        except FileNotFoundError:
            _logger.error(f"Image file not found: {image_path}")
            raise FileNotFoundError(f"Image file not found: {image_path}")
        except Exception as e:
            _logger.exception("Failed to encode image.")
            raise ValueError(f"Failed to encode image: {str(e)}")

    def qa_from_images(
        self,
        image_inputs: Union[str, List[str]],
        question: str,
        max_tokens: int = None,
        returned_as_dict: bool = False,
    ) -> Union[str, dict]:
        """
        Performs question-answering (QA) from one or multiple images.

        Args:
            image_inputs (Union[str, List[str]]): A single image path/URL or a list of paths/URLs.
            question (str): The question to ask about the images.
            max_tokens (int, optional): Maximum number of tokens for the response. Default is None.
            returned_as_dict (bool, optional): Whether to return the response as a dictionary. Default is False.

        Returns:
            Union[str, dict]: The response from the model.

        Raises:
            ValueError: If question or image inputs are invalid.
            Exception: If the API call fails.
        """
        if not question:
            _logger.error("Question is required.")
            raise ValueError("Question cannot be empty.")

        if not image_inputs:
            _logger.error("Image inputs are required.")
            raise ValueError("At least one image input is required.")

        if isinstance(image_inputs, str):
            image_inputs = [image_inputs]

        messages = [{"role": "user", "content": [{"type": "text", "text": question}]}]

        for image_input in image_inputs:
            try:
                if image_input.startswith("http://") or image_input.startswith(
                    "https://"
                ):
                    messages[0]["content"].append(
                        {"type": "image_url", "image_url": {"url": image_input}}
                    )
                else:
                    base64_image = self._encode_image(image_input)
                    messages[0]["content"].append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        }
                    )
            except Exception as e:
                _logger.error(
                    f"Failed to process image input: {image_input}. Error: {str(e)}"
                )
                raise

        try:
            response = (
                self.client.chat.completions.create(
                    model=self.model, messages=messages, max_tokens=max_tokens
                )
                if max_tokens
                else self.client.chat.completions.create(
                    model=self.model, messages=messages
                )
            )

            _logger.info("QA from images completed successfully.")
            return (
                response.to_dict()
                if returned_as_dict
                else response
            )

        except Exception as e:
            _logger.exception("Failed to perform QA from images.")
            raise RuntimeError(f"Failed to perform QA: {str(e)}")

    def image_generation(
        self,
        prompt: str,
        image_generation_model: str = "dall-e-3",
        n: int = 1,
        size: str = "1024x1024",
    ) -> List[str]:
        """
        Generate images based on a given prompt.

        Args:
            prompt (str): The prompt for generating images.
            image_generation_model (str): The model to use for image generation. Default is "dall-e-3".
            n (int): The number of images to generate. Default is 1.
            size (str): The size of the generated images. Default is "1024x1024".

        Returns:
            List[str]: A list of URLs to the generated images.

        Raises:
            ValueError: If the prompt is invalid.
            Exception: If the API call fails.
        """
        if not prompt:
            _logger.error("Prompt is required.")
            raise ValueError("Prompt cannot be empty.")

        try:
            response = self.client.images.generate(
                model=image_generation_model,
                prompt=prompt,
                n=n,
                size=size,
            )

            _logger.info("Image generation completed successfully.")
            return [image["url"] for image in response["data"]]

        except Exception as e:
            _logger.exception("Failed to generate images.")
            raise RuntimeError(f"Failed to generate images: {str(e)}")
        
    def image_variation(
        self,
        image_path: str = None,
        image_bytes: bytearray = None,
        model: str = "dall-e-2",
        size: Literal['256x256', '512x512', '1024x1024'] = "1024x1024",
        n: int=1,
        response_format: Literal["url", "b64_json"] = "url",
    ) -> List[str]:
        """
        Generate variations of an image.

        Args:
            image_path (str): The path to the image file.
            image_bytes (bytearray): The image data as a bytearray.
            model (str): The model to use for image variation. Default is "dall-e-2".
            size (str): The size of the generated images. Default is "1024x1024".
            n (int): The number of variations to generate. Default is 1.

        Returns:
            List[str]: A list of URLs to the generated variations.

        Raises:
            ValueError: If the image path or bytes are invalid.
            Exception: If the API call fails.
        """
        if not image_path and not image_bytes:
            _logger.error("Image path or bytes are required.")
            raise ValueError("Image path or bytes are required.")
        if image_path and image_bytes:
            _logger.error("Only one of image path or bytes should be provided.")
            raise ValueError("Only one of image path or bytes should be provided.")

        try:
            if image_path:
                image_data = open(image_path, "rb").read()
            else:
                image_data = image_bytes

            response = self.client.images.create_variation(
                model=model,
                image=image_data,
                size=size,
                n=n,
                response_format=response_format,
            )

            _logger.info("Image variation completed successfully.")
            return [image["url"] for image in response["data"]]
        except Exception as e:
            _logger.exception("Failed to generate image variations.")
            raise RuntimeError(f"Failed to generate image variations: {str(e)}")
