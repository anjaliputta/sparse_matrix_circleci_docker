import pytest
import numpy as np
from sparse_recommender import SparseMatrix

def test_get_set():
    sparse_matrix_obj = SparseMatrix()
    sparse_matrix_obj.set(0, 1, 1)
    sparse_matrix_obj.set(1, 0, 3)
    sparse_matrix_obj.set(1, 1, 9)
    sparse_matrix_obj.set(1, 2, 90)
    assert sparse_matrix_obj.get(0, 1) == 1
    assert sparse_matrix_obj.get(1, 0) == 3
    assert sparse_matrix_obj.get(1, 2) == 90

def test_add_movie():
    sparse_matrix_obj = SparseMatrix()
    sparse_matrix_obj.set(0, 1, 1)
    sparse_matrix_obj.set(1, 0, 3)
    sparse_matrix_obj.set(1, 1, 9)
    sparse_matrix_obj.set(1, 2, 90)

    new_matrix_obj = SparseMatrix()
    new_matrix_obj.set(0, 0, 2)
    new_matrix_obj.set(0, 1, 6)
    
    result_matrix = sparse_matrix_obj.add_movie(new_matrix_obj.sparse_matrix)
    assert sparse_matrix_obj.get(0, 0) == 0
    assert sparse_matrix_obj.get(0, 1) == 1
    assert sparse_matrix_obj.get(1, 0) == 3
    assert sparse_matrix_obj.get(1, 1) == 9
    assert sparse_matrix_obj.get(2, 0) == 2
    assert sparse_matrix_obj.get(2, 1) == 6

def test_to_dense():
    sparse_matrix_obj = SparseMatrix()
    dense_matrix = sparse_matrix_obj.to_dense()
    assert dense_matrix == []
    sparse_matrix_obj.set(0, 1, 11)
    sparse_matrix_obj.set(1, 1, 9)
    sparse_matrix_obj.set(2, 2, 13)
    dense_matrix = sparse_matrix_obj.to_dense()
    assert dense_matrix[0][0] == 0
    assert dense_matrix[1][1] == 9
    assert dense_matrix[2][2] == 13
    assert dense_matrix[2][1] == 0

def test_recommend():
    sparse_matrix_obj = SparseMatrix()
    sparse_matrix_obj.set(0, 1, 10)
    sparse_matrix_obj.set(1, 1, 9)
    sparse_matrix_obj.set(2, 2, 3)
    list1 = [7, 6, 3]
    user_array = np.array(list1).reshape(3, 1)
    new_matrix = sparse_matrix_obj.recommend(user_array)
    assert new_matrix[0][0] == 60
    assert new_matrix[1][0] == 54
    assert new_matrix[2][0] == 9

def test_get_set_exceptions():
    sparse_matrix_obj = SparseMatrix()
    
    # if trying to access values beyond matrix dimensions, catch keyerror
    with pytest.raises(KeyError):
        sparse_matrix_obj.get(2, 5)

    # if trying to access values at negative indices, raise index error
    with pytest.raises(IndexError):
        sparse_matrix_obj.set(-1, 1, 5)
    with pytest.raises(IndexError):
        sparse_matrix_obj.set(0, -1, 5)
    
    sparse_matrix_obj.set(0, 1, 10)
    # if trying to access values beyond matrix dimensions, catch keyerror
    with pytest.raises(KeyError):
        sparse_matrix_obj.get(1, 1)

def test_recommend_exceptions():
    sparse_matrix = SparseMatrix()
    sparse_matrix.set(0, 1, 10)
    sparse_matrix.set(1, 1, 9)
    sparse_matrix.set(2, 2, 3)

    # if no of rows of a vector not equlas to no of colmuns of sparse matrix, catch value error
    list1 = [7, 6, 3, 2]
    user_array = np.array(list1).reshape(4, 1)
    with pytest.raises(ValueError):
        sparse_matrix.recommend(user_array)

    # if vector has invalid values, catch valueerror
    list2 = [0, '', 3]
    user_array = np.array(list2).reshape(3, 1)
    with pytest.raises(ValueError):
        new_matrix = sparse_matrix.recommend(user_array)

    # if vector is empty, catch valueerror
    list3 = []
    with pytest.raises(ValueError):
        new_matrix = sparse_matrix.recommend(list3)

    # if vector has multiple columns, catch valueerror
    list4 = [[7, 6, 3], [9, 10, 43]]
    user_array = np.array(list4)
    with pytest.raises(ValueError):
        sparse_matrix.recommend(user_array)

def test_add_movie_exceptions():
    sparse_matrix_obj = SparseMatrix()
    sparse_matrix_obj.set(0, 1, 1)
    sparse_matrix_obj.set(1, 0, 3)
    sparse_matrix_obj.set(1, 1, 9)
    sparse_matrix_obj.set(1, 2, 90)

    new_matrix_obj = SparseMatrix()
    # if movie matrix is empty, catch valueerror
    with pytest.raises(ValueError):
        result_matrix = sparse_matrix_obj.add_movie(new_matrix_obj.sparse_matrix)

    new_matrix_obj.set(0, 0, 2)
    new_matrix_obj.set(0, 1, 6)
    new_matrix_obj.set(1, 2, 8)
    result_matrix = sparse_matrix_obj.add_movie(new_matrix_obj.sparse_matrix)
    assert result_matrix.get((2,0)) == 2
    assert result_matrix.get((3,2)) == 8

    # if movie matrix column size exceeds sparse matrix column size, catch indexerror
    new_matrix_obj2 = SparseMatrix()
    new_matrix_obj2.set(0, 0, 2)
    new_matrix_obj2.set(0, 1, 6)
    new_matrix_obj2.set(0, 3, 8)
    with pytest.raises(IndexError):
        result_matrix = sparse_matrix_obj.add_movie(new_matrix_obj2.sparse_matrix)

    # if movie matrix has invalid values, catch value error 
    new_matrix_obj3 = SparseMatrix()
    new_matrix_obj3.set(0, 0, "")
    new_matrix_obj3.set(0, 1, 6)
    new_matrix_obj3.set(0, 2, None)
    with pytest.raises(ValueError):
        result_matrix = sparse_matrix_obj.add_movie(new_matrix_obj3.sparse_matrix)

