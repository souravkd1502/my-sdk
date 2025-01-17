# Import necessary modules for testing
import asyncio
import unittest
import numpy as np
from ReusableComponents.CustomModules.OpenAI.embeddings_handler import EmbeddingHandler

# FILE: ReusableComponents/CustomModules/OpenAI/embeddings_handler.py


class TestEmbeddingHandler(unittest.TestCase):

    def setUp(self):
        """
        Set up the test environment for EmbeddingHandler tests.

        This method initializes the necessary variables and objects
        required for testing the EmbeddingHandler class.
        """
        # API key for OpenAI API
        self.api_key = "<YOUR_API_KEY>"
        
        # Model to be used for embeddings
        self.model = "text-embedding-3-small"
        
        # Initialize the EmbeddingHandler with the API key and model
        self.handler = EmbeddingHandler(api_key=self.api_key, model=self.model)
        
        # Sample text for generating embeddings
        self.text = "Hello, world!"
        
        # List of sample texts for batch embedding generation
        self.texts = ["Hello, world!", "How are you?"]
        

    def test_get_embedding(self):
        """
        Test the synchronous get_embedding method to ensure it returns
        a list of float values as embeddings for a given input text.
        """
        # Get the embedding for the given text
        embedding = self.handler.get_embedding(self.text)
        
        # Assert that the embedding is a list of float values
        self.assertIsInstance(embedding, list)
        self.assertTrue(all(isinstance(x, float) for x in embedding))

    def test_aget_embedding(self):
        """
        Test the asynchronous aget_embedding method to ensure it returns
        a list of float values as embeddings for a given input text.

        This test method uses asyncio to run the asynchronous aget_embedding
        method and asserts that the returned embedding is a list of float
        values.

        """
        async def run_test():
            """
            Run the aget_embedding asynchronous method and assert the returned
            embedding is a list of float values.

            """
            embedding = await self.handler.aget_embedding(self.text)  # Await the async method
            self.assertIsInstance(embedding, list)
            self.assertTrue(all(isinstance(x, float) for x in embedding))

        asyncio.run(run_test())  # Run the async test using asyncio

    def test_get_embeddings(self):
        """
        Test the synchronous get_embeddings method to ensure it returns
        a list of list of float values as embeddings for a given list of texts.

        This test method asserts that the returned embeddings is a list of
        list of float values.

        """
        embeddings = self.handler.get_embeddings(self.texts)
        self.assertIsInstance(embeddings, list)
        self.assertTrue(all(isinstance(e, list) for e in embeddings))
        self.assertTrue(all(isinstance(x, float) for e in embeddings for x in e))

    def test_aget_embeddings(self):
        """
        Test the asynchronous aget_embeddings method to ensure it returns
        a list of lists of float values as embeddings for a given list of texts.

        This test method uses asyncio to run the asynchronous aget_embeddings
        method and asserts that the returned embeddings are in the correct format.
        """
        async def run_test():
            """
            Run the aget_embeddings asynchronous method and assert the returned
            embeddings are lists of float values.
            """
            # Await the async method to get embeddings
            embeddings = await self.handler.aget_embeddings(self.texts)
            
            # Assert that embeddings is a list
            self.assertIsInstance(embeddings, list)
            
            # Assert that each element in embeddings is a list
            self.assertTrue(all(isinstance(e, list) for e in embeddings))
            
            # Assert that each element in the lists is a float
            self.assertTrue(all(isinstance(x, float) for e in embeddings for x in e))

        # Run the async test using asyncio
        asyncio.run(run_test())

    def test_cosine_similarity(self):
        """
        Test the cosine_similarity method to ensure it returns a float
        value as the cosine similarity between two vectors.

        This test method asserts that the returned similarity is a float
        value.

        """
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        similarity = self.handler.cosine_similarity(a, b)
        self.assertIsInstance(similarity, float)

    def test_euclidean_distance(self):
        """
        Test the euclidean_distance method to ensure it returns a float
        value as the Euclidean distance between two vectors.

        This test method asserts that the returned distance is a float
        value.
        """
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        distance = self.handler.euclidean_distance(a, b)
        self.assertIsInstance(distance, float)

    def test_distances_from_embeddings(self):
        """
        Test the distances_from_embeddings method to ensure it returns
        a list of float values as the distances between a query embedding
        and a list of embeddings.

        This test method asserts that the returned distances is a list
        of float values.
        """
        query_embedding = [1, 2, 3]
        embeddings = [[4, 5, 6], [7, 8, 9]]
        distances = self.handler.distances_from_embeddings(query_embedding, embeddings)
        self.assertIsInstance(distances, list)
        self.assertTrue(all(isinstance(d, float) for d in distances))

    def test_indices_of_nearest_neighbors_from_distances(self):
        """
        Test the indices_of_nearest_neighbors_from_distances method to ensure it returns
        a numpy array of indices that sorts the distances in ascending order.

        This test method asserts that the returned indices are a numpy array.
        """
        distances = [0.2, 0.5, 0.1]
        indices = self.handler.indices_of_nearest_neighbors_from_distances(distances)
        self.assertIsInstance(indices, np.ndarray)
        # Check that the indices are correct
        self.assertTrue(np.array_equal(indices, np.array([2, 0, 1])))

    def test_pca_components_from_embeddings(self):
        """
        Test the pca_components_from_embeddings method to ensure it returns
        a numpy array of transformed components from the input embeddings.

        This test method asserts that the returned components are a numpy array.
        """
        # Sample embeddings to perform PCA on
        embeddings = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        
        # Call the PCA method with the sample embeddings
        components = self.handler.pca_components_from_embeddings(embeddings)
        
        # Assert that the result is a numpy array
        self.assertIsInstance(components, np.ndarray)

    def test_tsne_components_from_embeddings(self):
        """
        Test the tsne_components_from_embeddings method to ensure it returns
        a numpy array of transformed components from the input embeddings.

        This test method asserts that the returned components are a numpy array.
        """
        # Sample embeddings to perform t-SNE on
        embeddings = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        # Call the t-SNE method with the sample embeddings
        components = self.handler.tsne_components_from_embeddings(embeddings)

        # Assert that the result is a numpy array
        self.assertIsInstance(components, np.ndarray)

    def test_plot_multiclass_precision_recall(self):
        """
        Test the plot_multiclass_precision_recall method to ensure it
        plots the precision-recall curves for a multiclass classification problem.

        This test method does not assert any return value but ensures that
        the method runs without errors.
        """
        # Define the predicted scores for each class
        y_score = np.array([[0.1, 0.9], [0.8, 0.2]])
        
        # Define the true class labels
        y_true = np.array([1, 0])
        
        # Define the list of class names
        class_list = ["class_0", "class_1"]
        
        # Define the name of the classifier being tested
        classifier_name = "TestClassifier"
        
        # Call the plot_multiclass_precision_recall method
        # This method is expected to plot precision-recall curves
        self.handler.plot_multiclass_precision_recall(
            y_score, y_true, class_list, classifier_name
        )

    def test_chart_from_components(self):
        """
        Test the chart_from_components method to ensure it returns an
        interactive 2D chart of embedding components.

        This test method asserts that the returned chart is not None.
        """
        # Sample components to plot
        components = np.array([[1, 2], [3, 4], [5, 6]])

        # Call the chart_from_components method with the sample components
        chart = self.handler.chart_from_components(components)

        # Assert that the result is not None
        self.assertIsNotNone(chart)


if __name__ == "__main__":
    unittest.main()
