import streamlit as st
from html.entities import codepoint2name

def app():

    st.title('Alternating Least Square ("ALS")')

    st.subheader('Short description of the algorithm')
    st.write("ALS is a latent factor model which was popularized by the Netflix Prize competition. It "
            "is based on matrix factorization process. The goal of the ALS is to find "
            "two matrices, U and V, such that their product is approximately equal to the original "
            "matrix of user and items.")

    st.subheader('Different steps of the process of the ALS')
    st.write('**Step 1: Creation of two sub-matrices U and V**')
    st.write('This matrix factorization process relies indeed on the characterization of both the users and items '
            'by vectors of factors inferred from item rating pattern. We therefore need to initialize randomly '
            'two matrices of smaller dimensions, U and V. The resulting dot product of U and V captures the interaction between a user and an item.')
    #st.image('../matrix_facto.png')
    st.image('https://miro.medium.com/max/1400/1*b4M7o7W8bfRRxdMxtFoVBQ.png')
    st.markdown('The above image has been borrowed from an article of James Le')

    st.write("**Step 2: Initialisation of a Loss function ('L')**")
    st.write("Once the two sub-matrices are initialized, we establish a loss function such that "
            "the difference between the input data of the original matrix  and the dot product be "
            " be minimized.")
    st.latex(r'''
            L = \sum(data[i]  - (U [row[i], : ] * V[ :, col[i]]))^2
            ''')

    st.write("**Step 3: Training of an algorithm 'Stochastic Gradient Descent'**")
    st.write(" As specified earlier, our goal is to minimize our loss function so that we can "
            "reconstitute our original matrix as precisely as we can. In ALS, our Stochasting Gradient Descent "
            "will calculate the next point using gradient at the current position, will scale it and finally "
            "will subtract the value that it obtained from the current position as we want to "
            "minimise the function. The particularity of the ALS is that it will alternatively train: the algorithm "
            "will first operate on matrix U by finding the global minimum "
            "and fix matrix V. Then it will apply on matrix V and fix matrix U. ")

    with st.expander("See python script & Gradient Descent graph"):
        code = '''
        from time import time
        t0 = time()
        eps = 0.1
        for i in range(len(row)):
            """
            L = (data[i] - U[row[i],:]@V[:,col[i])**2  #General form of the loss function
            """
            L_prime_U = -2*V[:,col[i]]*(list(data)[i] - U[row[i],:]@V[:,col[i]]) # Partial derivative of U
            L_prime_V = -2*U[row[i],:]*(list(data)[i] - U[row[i],:]@V[:,col[i]]) # Partial derivative of V
            U[row[i],:] -= eps*L_prime_U # Gradient descent with a step (eps).
            V[:,col[i]] -= eps*L_prime_V # Gradient descent with a step (eps)
        t1 = time(  )
        print("Execution time of the algo:", t1-t0)
        '''
        st.code(code, language='python')
        #st.image('../gradient_descent.png')
        st.image('https://miro.medium.com/max/1400/1*jNyE54fTVOH1203IwYeNEg.png')
        st.markdown('The above image has been borrowed from an article of Imad Dabbura')

    st.write("**Step 4: Deal with the 'Exploding Gradient'**")
    st.write("During the training of the Stochastic Gradient Descent algorithm, there might be times "
            "when the gradient grows exponentially, causing the algorithm to deviate completely from "
            "its objective of finding the global minimum. In this case, we apply a method known "
            "as 'Gradient clipping': we will normalize the gradient if its value goes over a "
            "limit that we fix.")
    #st.image('../gradient_clipping.png')
    st.image('https://miro.medium.com/max/700/1*vLFINWklJ0BtYtgzwK223g.png')
    st.markdown('The above image has been borrowed from an article of Wanshun Wong')

    st.write("The new script (applied on 10 000 tuples of user-item) would look like this:")
    with st.expander("See the updated python script"):


        code2 = '''
        from time import time
        from numpy import linalg as LA
        t0 = time()
        eps = 0.1
        #train model on 10 000 tuples (user-item) only
        keys_10000 = list(matrix_m_values.keys())[:10000]
        #train model on all tuples
        total_keys = list(matrix_m_values.keys())
        for key in total_keys:
            """
            L = (matrix_m_values[key] - U[key[0],:]@V[:,key[1]])**2  #General form of Loss function
            """
            L_prime_U = -2*V[:,key[1]]*(matrix_m_values[key] - U[key[0],:]@V[:,key[1]]) # partial derivative of U
            L_prime_V = -2*U[key[0],:]*(matrix_m_values[key] - U[key[0],:]@V[:,key[1]]) # partial derivative of V
        # Gradient descent with a step
        # Normalization of gradient if norm above 10 to avoid Exploding gradient
            if(LA.norm(L_prime_U) > 10):
            U[key[0],:] -= 10*eps*L_prime_U/LA.norm(L_prime_U)
            else:
            U[key[0],:] -= eps*L_prime_U
            if(LA.norm(L_prime_U) > 10):
            V[:,key[1]] -= 10*eps*L_prime_V/LA.norm(L_prime_V) # Descente de gradient avec un pas représenté par eps. On ajoute chaque nouvelle valeur à la matrice U
            else:
            V[:,key[1]] -= eps*L_prime_V # Idem
        t1 = time()
        print("Execution time of the algo:", t1-t0)
        '''
        st.code(code2, language='python')

    st.write("**Step 5: Reconstitution of the original matrix M**")
    st.write("As specified before, the original matrix will be reconstructed by taking the subproduct "
            "of the two sub-matrices U and V. ")