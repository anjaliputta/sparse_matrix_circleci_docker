import numpy as np

class SparseMatrix:

    def __init__(self):
        self.sparse_matrix = {}

    def get(self, row, col):
        # if sparse matrix is empty
        if len(self.sparse_matrix.keys()) == 0:
            raise KeyError("Matrix is empty.")
        
        max_row = max([row for (row, col) in self.sparse_matrix.keys()])
        max_col = max([col for (row, col) in self.sparse_matrix.keys()])

        # if trying to access values using negative indices
        if row < 0 or col < 0:
            raise KeyError("Row and column indices must be non-negative.")
        
        # if trying to access values at out of bound matrix locations
        if row > max_row or col > max_col:
            raise KeyError("Index out of bounds.")
        
        return self.sparse_matrix.get((row, col), 0)

    def set(self, row, col, value):
        # if trying to place values at negative indices of matrix
        if row < 0 or col < 0:
            raise IndexError("Row and column indices must be non-negative.")
        if value != 0:
            self.sparse_matrix[(row, col)] = value

    def recommend(self, vector):
        # if vector is empty
        if len(vector) == 0:
            raise ValueError("Given user vector is empty.")
        
        max_row_len = max([row for (row, col) in self.sparse_matrix.keys()]) + 1
        max_col_len = max([col for (row, col) in self.sparse_matrix.keys()]) + 1
        vector_row_len = len(vector)
        vector_col_len = len(vector[0])
        
        # if col size of sparse matrix not equals to row size of vector
        if(max_col_len != vector_row_len):
            raise ValueError("Matrix and vector dimensions do not match for multiplication.")
    
        # if vector has multiple columns
        if vector_col_len != 1:
            raise ValueError("Vector should have a single column.")
        
        result_vector = np.zeros((max_row_len, 1))
        for (row, col), value in self.sparse_matrix.items():
            if vector[col][0]:
                result_vector[row][0] += value * vector[col][0]
            else:
                # raise the value error if vector data is invalid
                raise ValueError("Received unwanted/empty values in a vector")

        return result_vector          

    def add_movie(self, matrix):
        # if matrix is empty
        if len(matrix) == 0:
            raise ValueError("Given Movie matrix is empty.")
        
        matrix1_max_row = 0
        matrix1_max_col = 0

        if len(self.sparse_matrix.keys()) != 0:
            matrix1_max_row = max([row for (row, col) in self.sparse_matrix.keys()])
            matrix1_max_col = max([col for (row, col) in self.sparse_matrix.keys()])

        matrix2_max_col = max([col for (row, col) in matrix.keys()])

        # if col size of movie matrix exceeds the col size of sparse matrix 
        if matrix2_max_col > matrix1_max_col:
            raise IndexError("Column size did not match for adding new movie.")

        for (row, col) in matrix.keys():
            if matrix[(row, col)]:
                self.sparse_matrix[(matrix1_max_row + row + 1, col)] = matrix[(row, col)]
            else:
                # raise the value error if movie matrix data is invalid
                raise ValueError("Received unwanted/empty values in a movie matrix")
        
        return self.sparse_matrix

    def to_dense(self):
        # if sparse matrix is empty
        if len(self.sparse_matrix.keys()) == 0:
            return []
        
        max_row = max([row for (row, col) in self.sparse_matrix.keys()])
        max_col = max([col for (row, col) in self.sparse_matrix.keys()])
        dense_matrix = [[0]*(max_col+1) for _ in range(max_row+1)]

        for (row, col), value in self.sparse_matrix.items():
            dense_matrix[row][col] = value
            
        return dense_matrix
    

    
        
        