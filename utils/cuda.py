from numba import cuda


@cuda.reduce
def sum_reduce(a, b):
    return a + b


@cuda.jit
def sum_reduce_2D(arr, result):
    # Calculate the 1D index in the original array
    x = cuda.grid(1)

    # Check the index is within bounds
    if x < arr.shape[0]:
        # Perform the sum along the two innermost dimensions
        temp_sum = 0
        for i in range(arr.shape[1]):
            for j in range(arr.shape[2]):
                temp_sum += arr[x, i, j]  # Assuming you want to sum over the 0th channel

        # Write the result to the output array
        result[x] = temp_sum
