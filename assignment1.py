###################################################################
# Do not include any additional import statements
import numpy as np
import time

###################################################################
# - Fill in the code below the comment Python and NumPy same as in example.
# - Follow instructions in document.


def examples():
    ###################################################################
    # Example: Create a zeros vector of size 10 and store it in variable tmp.
    # Python
    pythonStartTime = time.time()
    tmp_1 = [0 for i in range(10)]
    pythonEndTime = time.time()

    # NumPy
    numPyStartTime = time.time()
    tmp_2 = np.zeros(10)
    numPyEndTime = time.time()
    print('Python time: {0} sec.'.format(pythonEndTime-pythonStartTime))
    print('NumPy time: {0} sec.'.format(numPyEndTime-numPyStartTime))


def question_set_1():
    ################################################################
    # This set of tasks is designed to build familiarity with manipulating
    # the elements of arrays.
    ################################################################
    z_1 = None
    z_2 = None
    ################################################################
    # 1. Create a zeros array of size (3,5) and store in variable z.
    # Python
    # TODO 1a
    pythonStartTime = time.perf_counter()

    z_1 = [[0]*5 for _ in range(3)]

    pythonEndTime = time.perf_counter()

    # NumPy
    # TODO 1b
    numPyStartTime = time.perf_counter()

    z_2 = np.zeros((3, 5), dtype='int')

    numPyEndTime = time.perf_counter()
    print('Python time_1: {0} sec.'.format(pythonEndTime - pythonStartTime))
    print('NumPy time_1: {0} sec.'.format(numPyEndTime - numPyStartTime))

    #################################################
    # 2. Set all the elements in first row of z to 7.
    # Python
    # TODO 2a
    pythonStartTime = time.perf_counter()

    for i in range(5):
        z_1[0][i] = 7

    pythonEndTime = time.perf_counter()
    # NumPy
    # TODO 2b
    numPyStartTime = time.perf_counter()

    z_2[0] = 7

    numPyEndTime = time.perf_counter()
    print('Python time_2: {0} sec.'.format(pythonEndTime - pythonStartTime))
    print('NumPy time_2: {0} sec.'.format(numPyEndTime - numPyStartTime))

    #####################################################
    # 3. Set all the elements in second column of z to 9.
    # Python
    # TODO 3a
    pythonStartTime = time.perf_counter()

    for i in range(3):
        z_1[i][1] = 9

    pythonEndTime = time.perf_counter()

    # NumPy
    # TODO 3b
    numPyStartTime = time.perf_counter()

    z_2[:, 1] = 9

    numPyEndTime = time.perf_counter()
    print('Python time_3: {0} sec.'.format(pythonEndTime - pythonStartTime))
    print('NumPy time_3: {0} sec.'.format(numPyEndTime - numPyStartTime))

    #############################################################
    # 4. Set the element at (second row, third column) of z to 5.
    # Python
    # TODO 4a
    pythonStartTime = time.perf_counter()

    z_1[1][2] = 5

    pythonEndTime = time.perf_counter()

    # NumPy
    # TODO 4b
    numPyStartTime = time.perf_counter()

    z_2[1, 2] = 5

    numPyEndTime = time.perf_counter()
    print('Python time_4: {0} sec.'.format(pythonEndTime - pythonStartTime))
    print('NumPy time_4: {0} sec.'.format(numPyEndTime - numPyStartTime))

    ##############
    print(z_1)
    print(z_2)
    ##############
    # Do not modify the return statement.
    return z_1, z_2
    ##############


def question_set_2():
    ################################################################
    # This set of tasks is designed to build familiarity with creating
    # arrays with various entries.
    ################################################################
    x_1 = None
    x_2 = None
    ##########################################################################################
    # 5. Create a vector of size 50 with values ranging from 50 to 99 and store in variable x.
    # Python
    # TODO 5a
    pythonStartTime = time.perf_counter()

    x_1 = [i for i in range(50, 100)]

    pythonEndTime = time.perf_counter()

    # NumPy
    # TODO 5b
    numPyStartTime = time.perf_counter()

    x_2 = np.arange(50, 100)

    numPyEndTime = time.perf_counter()
    print('Python time_5: {0} sec.'.format(pythonEndTime - pythonStartTime))
    print('NumPy time_5: {0} sec.'.format(numPyEndTime - numPyStartTime))

    ##############
    print(x_1)
    print(x_2)
    ##############

    y_1 = None
    y_2 = None
    ##################################################################################
    # 6. Create a 4x4 matrix with values ranging from 0 to 15 and store in variable y.
    # Python
    # TODO 6a
    pythonStartTime = time.perf_counter()

    temp = [i for i in range(16)]
    y_1 = [[0] * 4 for _ in range(4)]
    x = 0
    for i in range(4):
        for j in range(4):
            y_1[i][j] = temp[x]
            x += 1

    pythonEndTime = time.perf_counter()


    # NumPy
    # TODO 6b
    numPyStartTime = time.perf_counter()

    y_2 = np.arange(16)
    y_2 = y_2.reshape(4, 4)

    numPyEndTime = time.perf_counter()
    print('Python time_6: {0} sec.'.format(pythonEndTime - pythonStartTime))
    print('NumPy time_6: {0} sec.'.format(numPyEndTime - numPyStartTime))

    ##############
    print(y_1)
    print(y_2)
    ##############

    tmp_1 = None
    tmp_2 = None
    ####################################################################################
    # 7. Create a 5x5 array with 1 on the border and 0 inside amd store in variable tmp.
    # Python
    # TODO 7a
    pythonStartTime = time.perf_counter()

    tmp_1 = [[0] * 5 for _ in range(5)]
    for i in range(5):
        for j in range(5):
            if i == 0 or i == 4 or j == 0 or j == 4:
                tmp_1[i][j] = 1

    pythonEndTime = time.perf_counter()

    # NumPy
    # TODO 7b
    numPyStartTime = time.perf_counter()

    tmp_2 = np.ones((5, 5))
    tmp_2[1:-1, 1:-1] = 0

    numPyEndTime = time.perf_counter()
    print('Python time_7: {0} sec.'.format(pythonEndTime - pythonStartTime))
    print('NumPy time_7: {0} sec.'.format(numPyEndTime - numPyStartTime))
    ##############
    print(tmp_1)
    print(tmp_2)
    ##############
    return x_1, x_2, y_1, y_2, tmp_1, tmp_2


def question_set_3():
    #############################################################################################
    # This final set focuses on actions on matrices.
    a_1 = None; a_2 = None
    b_1 = None; b_2 = None
    c_1 = None; c_2 = None
    #############################################################################################
    # 8. Generate a 50x100 array of integer between 0 and 5,000 and store in variable a.
    # Python
    # TODO 8a
    pythonStartTime = time.perf_counter()

    temp = [i for i in range(5000)]
    a_1 = [[0] * 100 for _ in range(50)]
    x = 0
    for i in range(50):
        for j in range(100):
            a_1[i][j] = temp[x]
            x += 1

    pythonEndTime = time.perf_counter()

    # NumPy
    # TODO 8b
    numPyStartTime = time.perf_counter()

    a_2 = np.arange(5000)
    a_2 = a_2.reshape(50, 100)

    numPyEndTime = time.perf_counter()
    print('Python time_8: {0} sec.'.format(pythonEndTime - pythonStartTime))
    print('NumPy time_8: {0} sec.'.format(numPyEndTime - numPyStartTime))
    ###############################################################################################
    # 9. Generate a 100x200 array of integer between 0 and 20,000 and store in variable b.
    # Python
    # TODO 9a
    pythonStartTime = time.perf_counter()

    temp = [i for i in range(20000)]
    b_1 = [[0] * 200 for _ in range(100)]
    x = 0
    for i in range(100):
        for j in range(200):
            b_1[i][j] = temp[x]
            x += 1

    pythonEndTime = time.perf_counter()

    # NumPy
    # TODO 9b
    numPyStartTime = time.perf_counter()

    b_2 = np.arange(20000)
    b_2 = b_2.reshape(100, 200)

    numPyEndTime = time.perf_counter()
    print('Python time_9: {0} sec.'.format(pythonEndTime - pythonStartTime))
    print('NumPy time_9: {0} sec.'.format(numPyEndTime - numPyStartTime))
    #####################################################################################
    # 10. Multiply matrix a and b together (real matrix product) and store to variable c.
    # Python
    # TODO 10a
    pythonStartTime = time.perf_counter()

    c_1 = [[0] * 200 for _ in range(50)]
    for i in range(50):
        for j in range(100):
            for k in range(200):
                c_1[i][k] = c_1[i][k] + a_1[i][j] * b_1[j][k]

    pythonEndTime = time.perf_counter()


    # NumPy
    # TODO 10b
    numPyStartTime = time.perf_counter()

    c_2 = np.matmul(a_2, b_2)

    numPyEndTime = time.perf_counter()
    print('Python time_10: {0} sec.'.format(pythonEndTime - pythonStartTime))
    print('NumPy time_10: {0} sec.'.format(numPyEndTime - numPyStartTime))

    d_1 = None; d_2 = None
    ################################################################################
    # 11. Normalize a 3x3 random matrix ((x-min)/(max-min)) and store to variable d.
    # Python
    # TODO 11a
    pythonStartTime = time.perf_counter()

    d_1 = [[0] * 3 for _ in range(3)]
    for i in range(3):
        for j in range(3):
            d_1[i][j] = np.random.randint(1000)
    d_1max = max(max(d_1))
    d_1min = min(min(d_1))
    for i in range(3):
        for j in range(3):
            d_1[i][j] = (d_1[i][j] - d_1min) / (d_1max - d_1min)

    pythonEndTime = time.perf_counter()


    # NumPy
    # TODO 11b
    numPyStartTime = time.perf_counter()

    d_2 = np.random.randint(1000, size=(3, 3))
    d_2max = d_2.max()
    d_2min = d_2.min()
    d_2 = (d_2 - d_2min) / (d_2max - d_2min)

    numPyEndTime = time.perf_counter()
    print('Python time_11: {0} sec.'.format(pythonEndTime - pythonStartTime))
    print('NumPy time_11: {0} sec.'.format(numPyEndTime - numPyStartTime))

    ##########
    print(d_1)
    print(d_2)
    #########

    ################################################
    # 12. Subtract the mean of each row of matrix a.
    # Python
    # TODO 12a
    pythonStartTime = time.perf_counter()

    m = [0 for i in range(len(a_1))]
    for i in range(len(a_1)):
        m[i] = sum(a_1[i])/len(a_1[i])
    x = 0
    for j in range(len(a_1)):
        for k in range(len(a_1[j])):
            a_1[j][k] = a_1[j][k] - m[x]
        x += 1

    pythonEndTime = time.perf_counter()

    # NumPy
    # TODO 12b
    numPyStartTime = time.perf_counter()

    a_2 = a_2 - np.mean(a_2, axis=1, keepdims=True)

    numPyEndTime = time.perf_counter()
    print('Python time_12: {0} sec.'.format(pythonEndTime - pythonStartTime))
    print('NumPy time_12: {0} sec.'.format(numPyEndTime - numPyStartTime))

    ###################################################
    # 13. Subtract the mean of each column of matrix b.
    # Python
    # TODO 13a
    pythonStartTime = time.perf_counter()

    t = [[b_1[j][i] for j in range(len(b_1))] for i in range(len(b_1[0]))]
    m = [0 for i in range(len(t))]
    for i in range(len(t)):
        m[i] = sum(t[i]) / len(t[i])
    x = 0
    for j in range(len(b_1)):
        x = 0
        for k in range(len(b_1[j])):
            b_1[j][k] = b_1[j][k] - m[x]
            x += 1

    pythonEndTime = time.perf_counter()


    # NumPy
    # TODO 13b
    numPyStartTime = time.perf_counter()

    b_2 = b_2 - np.mean(b_2, axis=0, keepdims=True)

    numPyEndTime = time.perf_counter()
    print('Python time_13: {0} sec.'.format(pythonEndTime - pythonStartTime))
    print('NumPy time_13: {0} sec.'.format(numPyEndTime - numPyStartTime))

    ################
    print(np.sum(a_1 == a_2))
    print(np.sum(b_1 == b_2))
    ################

    e_1 = None; e_2 = None
    ###################################################################################
    # 14. Transpose matrix c, add 5 to all elements in matrix, and store to variable e.
    # Python
    # TODO 14a
    pythonStartTime = time.perf_counter()

    t = [[c_1[j][i] for j in range(len(c_1))] for i in range(len(c_1[0]))]
    for i in range(len(t)):
        for j in range(len(t[0])):
            t[i][j] += 5
    e_1 = t

    pythonEndTime = time.perf_counter()

    # NumPy
    # TODO 14b
    numPyStartTime = time.perf_counter()

    e_2 = np.transpose(c_2)
    e_2 = e_2 + 5

    numPyEndTime = time.perf_counter()
    print('Python time_14: {0} sec.'.format(pythonEndTime - pythonStartTime))
    print('NumPy time_14: {0} sec.'.format(numPyEndTime - numPyStartTime))
    ##################
    print(np.sum(e_1 == e_2))
    ##################

    f_1 = None; f_2 = None
    g_1 = None; g_2 = None
    #####################################################################################
    # 15. Reshape matrix e to 1d array, store to variable f, and store the shape of f in
    # variable g. Finally, print g (the shape of f matrix).

    # Python
    # TODO 15a
    pythonStartTime = time.perf_counter()

    l = len(e_1)*len(e_1[0])
    f_1 = [[0] * 1 for _ in range(l)]
    k = 0
    for i in range(len(e_1)):
        for j in range(len(e_1[0])):
            f_1[k][0] = e_1[i][j]
            k += 1

    g_1 = [len(f_1), len(f_1[0])]
    print(g_1)

    pythonEndTime = time.perf_counter()


    # NumPy
    # TODO 15b
    numPyStartTime = time.perf_counter()

    f_2 = e_2.ravel()
    g_2 = np.shape(f_2)
    print(g_2)

    numPyEndTime = time.perf_counter()
    print('Python time_15: {0} sec.'.format(pythonEndTime - pythonStartTime))
    print('NumPy time_15: {0} sec.'.format(numPyEndTime - numPyStartTime))

    ################
    return a_1, a_2, b_1, b_2, c_1, c_2, d_1, d_2, e_1, e_2, f_1, f_2, g_1, g_2
    ################


#####################################################################################
# For your benefit, this section will help you check your answers. Feel free to comment out
# sections as you move through the problems.
if __name__ == "__main__":
    print("CSE 691 - Assignment #1 - Timing differences between Vanilla Python and Numpy")
    print("Set #1")
    question_set_1()
    print("Set #2")
    question_set_2()
    print("Set #3")
    question_set_3()
