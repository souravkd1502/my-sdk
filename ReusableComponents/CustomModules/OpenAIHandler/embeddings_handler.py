"""
embeddings_handler.py
----------------
This module contains the class definition for OpenAIEmbeddingsHandler, which is a wrapper class for OpenAI API to handle embeddings tasks.

Requirements:
-------------
- python-dotenv==1.0.1
- openai==1.59.3
- plotly==5.24.1
- numpy==1.26.4
- pandas==2.2.3
- scipy==1.15.0
- matplotlib==3.10.0
- plotly==5.24.1
- scikit-learn==1.6.0

Description:
------------
This module provides utility methods for handling embeddings tasks using OpenAI API. 
The OpenAIEmbeddingsHandler class provides methods to get embeddings for text, calculate distances between embeddings, 
and perform dimensionality reduction using PCA and t-SNE. It also includes methods for plotting precision-recall curves 
for multiclass classification tasks.

Functions:
----------
The OpenAIEmbeddingsHandler class has the following methods:
- get_embedding(text: str, **kwargs) -> List[float]: Get the embedding vector for a given text.
- aget_embedding(text: str, **kwargs) -> List[float]: Asynchronous version of get_embedding.
- get_embeddings(list_of_text: List[str], **kwargs) -> List[List[float]]: Get the embedding vectors for a list of texts.
- aget_embeddings(list_of_text: List[str], **kwargs) -> List[List[float]]: Asynchronous version of get_embeddings.
- cosine_similarity(a, b) -> float: Calculate the cosine similarity between two vectors.
- euclidean_distance(a, b) -> float: Calculate the Euclidean distance between two vectors.
- distances_from_embeddings(query_embedding: List[float], embeddings: List[List[float]], distance_metric: str = "cosine") -> List[float]: Calculate the distances between a query embedding and a list of embeddings.
- indices_of_nearest_neighbors_from_distances(distances) -> np.ndarray: Determine the indices of the nearest neighbors from a list of distances.
- pca_components_from_embeddings(embeddings: List[List[float]], n_components=2) -> np.ndarray: Compute the principal component analysis (PCA) components of embeddings.
- tsne_components_from_embeddings(embeddings: List[List[float]], n_components: int = 2, **kwargs) -> np.ndarray: Computes t-SNE (t-distributed Stochastic Neighbor Embedding) components of a list of embeddings.
- plot_multiclass_precision_recall(y_score: np.ndarray, y_true_untransformed: np.ndarray, class_list: List[str], classifier_name: str) -> None: Plot precision-recall curves for a multiclass problem.
- chart_from_components(components: np.ndarray, labels: Optional[List[str] = None, strings: Optional[List[str] = None, x_title: str = "Component 0", y_title: str = "Component 1", mark_size: int = 5, **kwargs) -> plotly.graph_objs._figure.Figure: Return an interactive 2D chart of embedding components.

Environment Variables:
----------------------
- OPENAI_API_KEY: The API key for OpenAI API.

TODO:
-----

FIXME:
------

Author:
-------
Sourav Das

Date:
-----
16.01.2025
"""

# Adding directories to system path to allow importing custom modules
import sys

sys.path.append("./")
sys.path.append("../")

# Importing necessary libraries and modules
import asyncio
import logging
import numpy as np
import pandas as pd
import textwrap as tr
from scipy import spatial
from openai import OpenAI
import plotly.express as px
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, precision_recall_curve

from typing import List, Optional

# Set up logging
_logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s - line: %(lineno)d",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Load environment variables
load_dotenv(override=True)


class EmbeddingHandler:
    """
    This class provides utility methods for handling embeddings tasks using OpenAI API.
    """

    EMBEDDING_MODEL_LIST = [
        "text-embedding-3-large",
        "text-embedding-3-small",
        "text-embedding-ada-002",
    ]

    def __init__(self, api_key: str, model: str = "text-embedding-3-small") -> None:
        """
        Initialize the OpenAIEmbeddingsHandler class.

        Args:
        -----
        - api_key (str): The API key for OpenAI API.
        - model (str): The model to be used for embeddings tasks.
        
        Example:
        --------
        >>> handler = OpenAIEmbeddingsHandler(api_key="your_api_key", model="text-embedding-3-small")
        """
        self.api_key = api_key
        self.client = OpenAI(api_key=self.api_key)
        _logger.info(f"Provided model name: {model}")

        self.model = self._validate_model(model)

    def _validate_model(self, model: str) -> str:
        """
        Validate the model name.

        Args:
        -----
        - model (str): The model name to be validated.

        Returns:
        --------
        - str: The validated model name.

        Raises:
        -------
        - ValueError: If the model name is not supported.
        - ValueError: If the model name is not provided.

        Notes:
        ------
        - text-embedding-3-large:
            - Most capable embedding model for both english and non-english tasks
            - Output Dimension: 3,072
        - text-embedding-3-small
            - Increased performance over 2nd generation ada embedding model
            - 1,536
        - text-embedding-ada-002
            - Most capable 2nd generation embedding model, replacing 16 first generation models
            - Output Dimension: 1,536
            
        Example:
        --------
        >>> handler = OpenAIEmbeddingsHandler(api_key="your_api_key", model="text-embedding-3-small")
        >>> model = handler._validate_model("text-embedding-3-small")
        """
        if model not in self.EMBEDDING_MODEL_LIST:
            _logger.error(f"Unsupported model: {model} used")
            raise ValueError(
                f"Unsupported model: {model}. Please provide one of these models: {self.EMBEDDING_MODEL_LIST}"
            )

        if model is None:
            _logger.error("Model not provided")
            raise ValueError(
                f"Unsupported model: {model}. Please provide one of these models: {self.EMBEDDING_MODEL_LIST}"
            )

        return model

    def get_embedding(self, text: str, **kwargs) -> List[float]:
        """
        Get the embedding vector for a given text.

        Args:
            text (str): The text for which to get the embedding.

        Returns:
            List[float]: The embedding vector.

        Raises:
            ValueError: If the model name is not supported.
        
        Example:
        --------
        >>> handler = OpenAIEmbeddingsHandler(api_key="your_api_key", model="text-embedding-3-small")
        >>> embedding = handler.get_embedding("Hello, world!")
        """
        # Replace newlines, which can negatively affect performance.
        text = text.replace("\n", " ")

        _logger.info(f"Text: {text}")
        _logger.info(f"Model: {self.model}")
        _logger.info(f"Kwargs: {kwargs}")

        # Create the request
        response = self.client.embeddings.create(
            input=[text], model=self.model, **kwargs
        )

        _logger.info(f"Response: {response}")

        # Return the embedding vector
        return response.data[0].embedding

    async def aget_embedding(self, text: str, **kwargs) -> List[float]:
        """
        Asynchronous version of get_embedding.

        Args:
            text (str): The text for which to get the embedding.

        Returns:
            List[float]: The embedding vector.

        Raises:
            ValueError: If the model name is not supported.
            
        Example:
        --------
        >>> handler = OpenAIEmbeddingsHandler(api_key="your_api_key", model="text-embedding-3-small")
        >>> embedding = await handler.aget_embedding("Hello, world!")
        """
        # Replace newlines, which can negatively affect performance.
        text = text.replace("\n", " ")

        _logger.info(f"Text: {text}")
        _logger.info(f"Model: {self.model}")
        _logger.info(f"Kwargs: {kwargs}")

        # Run the synchronous method in a separate thread
        response = await asyncio.to_thread(
            self.client.embeddings.create, input=[text], model=self.model, **kwargs
        )

        _logger.info(f"Response: {response}")

        # Return the embedding vector
        return response.to_dict()["data"][0]["embedding"]

    def get_embeddings(self, list_of_text: List[str], **kwargs) -> List[List[float]]:
        """
        Get the embedding vectors for a list of texts.

        Args:
            list_of_text (List[str]): The list of texts to get the embedding vectors for.

        Returns:
            List[List[float]]: The list of embedding vectors, each of which is a list of floats.

        Raises:
            ValueError: If the batch size is larger than 2048.
        
        Example:
        --------
        >>> handler = OpenAIEmbeddingsHandler(api_key="your_api_key", model="text-embedding-3-small")
        >>> embeddings = handler.get_embeddings(["Hello, world!", "Goodbye, world!"])
        """
        assert (
            len(list_of_text) <= 2048
        ), "The batch size should not be larger than 2048."

        _logger.info(f"Length of list_of_text: {len(list_of_text)}")
        _logger.info(f"list_of_text: {list_of_text}")
        _logger.info(f"Model: {self.model}")
        _logger.info(f"Kwargs: {kwargs}")

        # replace newlines, which can negatively affect performance.
        list_of_text = [text.replace("\n", " ") for text in list_of_text]

        _logger.info(f"Updated list_of_text: {list_of_text}")

        # Create the request
        data = self.client.embeddings.create(
            input=list_of_text, model=self.model, **kwargs
        ).data

        _logger.info(f"Response: {data}")

        # Return the embedding vectors
        return [d.embedding for d in data]

    async def aget_embeddings(
        self, list_of_text: List[str], **kwargs
    ) -> List[List[float]]:
        """
        Get the embedding vectors for a list of texts.

        Args:
            list_of_text (List[str]): The list of texts to get the embedding vectors for.

        Returns:
            List[List[float]]: The list of embedding vectors, each of which is a list of floats.

        Raises:
            ValueError: If the batch size is larger than 2048.
        
        Example:
        --------
        >>> handler = OpenAIEmbeddingsHandler(api_key="your_api_key", model="text-embedding-3-small")
        >>> embeddings = await handler.aget_embeddings(["Hello, world!", "Goodbye, world!"])
        """
        assert (
            len(list_of_text) <= 2048
        ), "The batch size should not be larger than 2048."

        _logger.info(f"Length of list_of_text: {len(list_of_text)}")
        _logger.info(f"list_of_text: {list_of_text}")
        _logger.info(f"Model: {self.model}")
        _logger.info(f"Kwargs: {kwargs}")

        # Replace newlines, which can negatively affect performance.
        list_of_text = [text.replace("\n", " ") for text in list_of_text]

        _logger.info(f"Updated list_of_text: {list_of_text}")

        # Create the request
        # Run the synchronous method in a separate thread
        response = await asyncio.to_thread(
            self.client.embeddings.create,
            input=list_of_text,
            model=self.model,
            **kwargs,
        )

        _logger.info(f"Response: {response.to_dict()['data']}")
        _logger.info(
            f"Type: {type([d['embedding'] for d in response.to_dict()['data']][0][0])}"
        )

        # Return the embedding vectors
        return [d["embedding"] for d in response.to_dict()["data"]]

    @staticmethod
    def cosine_similarity(a, b):
        """
        Calculate the cosine similarity between two vectors.

        Args:
            a (np.ndarray): First vector.
            b (np.ndarray): Second vector.

        Returns:
            float: The cosine similarity between vectors a and b.
            
        Example:
        --------
        >>> a = np.array([1, 2, 3])
        >>> b = np.array([4, 5, 6])
        >>> similarity = cosine_similarity(a, b)
        >>> print(similarity)
        """
        # Compute the dot product of vectors a and b
        dot_product = np.dot(a, b)

        # Compute the L2 norms (magnitudes) of vectors a and b
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        # Calculate and return the cosine similarity
        return dot_product / (norm_a * norm_b)

    @staticmethod
    def euclidean_distance(a, b):
        """
        Calculate the Euclidean distance between two vectors.

        This method uses NumPy's linear algebra module to compute the
        Euclidean distance, which is the L2 norm of the difference
        between the two vectors.

        Args:
            a (np.ndarray): First vector. Must be a 1-dimensional array.
            b (np.ndarray): Second vector. Must be a 1-dimensional array.

        Returns:
            float: The Euclidean distance between vectors a and b.

        Raises:
            ValueError: If the input arrays do not have the same shape.

        Example:
            >>> a = np.array([1, 2, 3])
            >>> b = np.array([4, 5, 6])
            >>> EmbeddingHandler.euclidean_distance(a, b)
            5.196152422706632
        """
        # Ensure both vectors have the same shape
        if a.shape != b.shape:
            raise ValueError("Input vectors must have the same shape.")

        # Calculate and return the L2 norm (Euclidean distance) of their difference
        return np.linalg.norm(a - b)

    @staticmethod
    def distances_from_embeddings(
        query_embedding: List[float],
        embeddings: List[List[float]],
        distance_metric: str = "cosine",
    ) -> List[float]:
        """
        Calculate the distances between a query embedding and a list of embeddings.

        Args:
            query_embedding (List[float]): The query embedding.
            embeddings (List[List[float]]): The list of embeddings to compare to.
            distance_metric (str, optional): The distance metric to use. Defaults to "cosine".

        Returns:
            List[float]: The list of distances between the query embedding and each of the given embeddings.

        Raises:
            ValueError: If the distance metric is not recognized.

        Example:
            >>> query_embedding = [1, 2, 3]
            >>> embeddings = [[4, 5, 6], [7, 8, 9]]
            >>> distances = EmbeddingHandler.distances_from_embeddings(query_embedding, embeddings)
            >>> distances
            [5.196152422706632, 9.486832980505138]
        """
        # Define the supported distance metrics and their corresponding functions
        distance_metrics = {
            "cosine": spatial.distance.cosine,
            "L1": spatial.distance.cityblock,
            "L2": spatial.distance.euclidean,
            "Linf": spatial.distance.chebyshev,
        }

        # Check if the distance metric is supported
        if distance_metric not in distance_metrics:
            raise ValueError(f"Unsupported distance metric: {distance_metric}")

        # Calculate and return the distances
        distances = [
            distance_metrics[distance_metric](query_embedding, embedding)
            for embedding in embeddings
        ]
        return distances

    @staticmethod
    def indices_of_nearest_neighbors_from_distances(distances) -> np.ndarray:
        """
        Determine the indices of the nearest neighbors from a list of distances.

        Args:
            distances (List[float]): A list of distances from which to find nearest neighbors.

        Returns:
            np.ndarray: An array of indices that sorts the distances in ascending order.

        Example:
            >>> distances = [0.2, 0.5, 0.1]
            >>> indices = EmbeddingHandler.indices_of_nearest_neighbors_from_distances(distances)
            >>> indices
            array([2, 0, 1])
        """
        # Use numpy's argsort to get indices that would sort the list in ascending order.
        return np.argsort(distances)

    @staticmethod
    def pca_components_from_embeddings(
        embeddings: List[List[float]], n_components=2
    ) -> np.ndarray:
        """
        Compute the principal component analysis (PCA) components of embeddings.

        Args:
            embeddings (List[List[float]]): A list of embeddings to perform PCA on.
            n_components (int, optional): The number of principal components to return. Defaults to 2.

        Returns:
            np.ndarray: The PCA-transformed components of the embeddings.

        Example:
        --------
        >>> embeddings = [[1, 2], [3, 4], [5, 6]]
        >>> pca_components = EmbeddingHandler.pca_components_from_embeddings(embeddings)
        >>> pca_components
        array([[-0.70710678, -0.70710678],
                [ 0.70710678, -0.70710678]])
        """
        # Initialize PCA with the desired number of components
        pca = PCA(n_components=n_components)

        # Convert the list of embeddings to a NumPy array for PCA
        array_of_embeddings = np.array(embeddings)

        # Fit PCA on the embeddings and transform them
        return pca.fit_transform(array_of_embeddings)

    @staticmethod
    def tsne_components_from_embeddings(
        embeddings: List[List[float]], n_components: int = 2, **kwargs
    ) -> np.ndarray:
        """
        Computes t-SNE (t-distributed Stochastic Neighbor Embedding) components of a list of embeddings.

        Args:
            embeddings (List[List[float]]): A list of embeddings to perform t-SNE on.
            n_components (int, optional): The number of principal components to return. Defaults to 2.
            **kwargs: Additional keyword arguments to pass to the t-SNE algorithm.

        Returns:
            np.ndarray: The t-SNE-transformed components of the embeddings.

        Notes:
            The t-SNE algorithm is a non-linear dimensionality reduction technique that is particularly well-suited
            for visualizing high-dimensional data. It is based on Stochastic Neighbor Embedding (SNE) and works by
            minimizing the Kullback-Leibler divergence between the distribution of the data in the high-dimensional
            space and the distribution of the data in the low-dimensional space.

            The t-SNE algorithm can be sensitive to the choice of hyperparameters, so it is a good idea to experiment
            with different values for the hyperparameters to find the best results for your specific use case.
        
        Example:
        --------
        >>> embeddings = [[1, 2], [3, 4], [5, 6]]
        >>> tsne_components = EmbeddingHandler.tsne_components_from_embeddings(embeddings)
        >>> tsne_components
        array([[-0.70710678, -0.70710678],
                [ 0.70710678, -0.70710678]])
        """
        # Convert input embeddings to a numpy array
        array_of_embeddings = np.array(embeddings)

        # Ensure perplexity is less than the number of samples
        n_samples = array_of_embeddings.shape[0]
        default_perplexity = kwargs.get("perplexity", 30)
        if default_perplexity >= n_samples:
            kwargs["perplexity"] = max(
                1, n_samples // 2
            )  # Set to half the samples or 1 if very few samples

        # Use better defaults for initialization and learning rate if not specified
        kwargs.setdefault("init", "pca")
        kwargs.setdefault("learning_rate", "auto")

        # Perform t-SNE
        tsne = TSNE(n_components=n_components, **kwargs)
        return tsne.fit_transform(array_of_embeddings)

    @staticmethod
    def plot_multiclass_precision_recall(
        y_score: np.ndarray,
        y_true_untransformed: np.ndarray,
        class_list: List[str],
        classifier_name: str,
    ) -> None:
        """
        Plot precision-recall curves for a multiclass problem.

        Parameters
        ----------
        y_score : np.ndarray of shape (n_samples, n_classes)
            The output of a classifier that assigns weights to samples for each of the n_classes classes.
        y_true_untransformed : np.ndarray of shape (n_samples,)
            The true labels of the samples. These labels are assumed to be integers in the range [0, n_classes-1].
        class_list : List[str]
            A list of strings representing the names of the classes.
        classifier_name : str
            The name of the classifier being evaluated.
            
        Example:
        --------
        >>> y_score = np.array([[0.1, 0.9], [0.8, 0.2]])
        >>> y_true_untransformed = np.array([1, 0])
        >>> class_list = ["class_0", "class_1"]
        >>> classifier_name = "TestClassifier"
        >>> EmbeddingHandler.plot_multiclass_precision_recall(y_score, y_true_untransformed, class_list, classifier_name)
        """
        n_classes = len(class_list)

        # Convert y_true_untransformed to one-hot encoded format
        y_true = pd.DataFrame(
            {
                class_list[i]: (y_true_untransformed == i).astype(int)
                for i in range(n_classes)
            }
        ).values

        _logger.info("y_score shape: %s", y_score.shape)
        _logger.info("y_true shape: %s", y_true.shape)

        # Compute precision, recall, and average precision for each class
        precision = {}
        recall = {}
        average_precision = {}
        for i in range(n_classes):
            _logger.info(
                "Computing precision, recall, and average precision for class %d", i
            )
            precision[i], recall[i], _ = precision_recall_curve(
                y_true[:, i], y_score[:, i]
            )
            average_precision[i] = average_precision_score(y_true[:, i], y_score[:, i])

        # Compute precision, recall, and average precision for all classes jointly
        precision_micro, recall_micro, _ = precision_recall_curve(
            y_true.ravel(), y_score.ravel()
        )
        average_precision_micro = average_precision_score(
            y_true, y_score, average="micro"
        )
        _logger.info(
            "%s - Average precision score over all classes: %.2f",
            classifier_name,
            average_precision_micro,
        )

        # Set up plot details
        plt.figure(figsize=(9, 10))
        f_scores = np.linspace(0.2, 0.8, num=4)
        lines = []
        labels = []
        for f_score in f_scores:
            x = np.linspace(0.01, 1)
            y = f_score * x / (2 * x - f_score)
            (l,) = plt.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)
            plt.annotate(f"f1={f_score:0.1f}", xy=(0.9, y[45] + 0.02))

        lines.append(l)
        labels.append("iso-f1 curves")
        (l,) = plt.plot(recall_micro, precision_micro, color="gold", lw=2)
        lines.append(l)
        labels.append(
            f"average Precision-recall (auprc = {average_precision_micro:0.2f})"
        )

        for i in range(n_classes):
            (l,) = plt.plot(recall[i], precision[i], lw=2)
            lines.append(l)
            labels.append(
                f"Precision-recall for class `{class_list[i]}` (auprc = {average_precision[i]:0.2f})"
            )

        fig = plt.gcf()
        fig.subplots_adjust(bottom=0.25, right=0.75)  # Adjust for legend space
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"{classifier_name}: Precision-Recall curve for each class")

        # Add legend outside the plot
        plt.legend(lines, labels, loc="center left", bbox_to_anchor=(1.05, 0.5))
        plt.show()

    @staticmethod
    def chart_from_components(
        components: np.ndarray,
        labels: Optional[List[str]] = None,
        strings: Optional[List[str]] = None,
        x_title: str = "Component 0",
        y_title: str = "Component 1",
        mark_size: int = 5,
        **kwargs,
    ):
        """
        Return an interactive 2D chart of embedding components.

        Args:
            components (np.ndarray): The 2D array of components.
            labels (Optional[List[str]], optional): The list of labels to color by. Defaults to None.
            strings (Optional[List[str]], optional): The list of strings to display in the tooltip. Defaults to None.
            x_title (str, optional): The title of the x-axis. Defaults to "Component 0".
            y_title (str, optional): The title of the y-axis. Defaults to "Component 1".
            mark_size (int, optional): The size of the points in the chart. Defaults to 5.
            **kwargs: Additional arguments to pass to the plotly express scatter function.

        Returns:
            plotly.graph_objs._figure.Figure: The interactive 2D chart of embedding components.
        
        Example:
        --------
        >>> components = np.array([[1, 2], [3, 4], [5, 6]])
        >>> labels = ["A", "B", "C"]
        >>> strings = ["String A", "String B", "String C"]
        >>> chart = EmbeddingHandler.chart_from_components(components, labels, strings, x_title="Component 1", y_title="Component 2", mark_size=10)
        >>> chart.show()
        """

        empty_list = ["" for _ in components]
        data = pd.DataFrame(
            {
                x_title: components[:, 0],
                y_title: components[:, 1],
                "label": labels if labels else empty_list,
                "string": (
                    ["<br>".join(tr.wrap(string, width=30)) for string in strings]
                    if strings
                    else empty_list
                ),
            }
        )
        chart = px.scatter(
            data,
            x=x_title,
            y=y_title,
            color="label" if labels else None,
            symbol="label" if labels else None,
            hover_data=["string"] if strings else None,
            **kwargs,
        ).update_traces(marker=dict(size=mark_size))
        return chart
