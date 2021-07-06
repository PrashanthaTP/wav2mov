def out_d(in_d,k,p,s):
    """calculates out shape of 2d cnn   

    Args:
        in_d (int): height or width
        k (int):kernel  height or width

        p (int): padding along height or width

        s (int): stride along height or width

    Returns:
        int
    """
    return ((in_d-k+2*p)/s)+1

def out_d_transposed(in_d,k,p,s):
    """ find z and p
        insert z number of zeros between each row and column (2*(i-1)x2*(i-1))
        pad with p number of zeros
        perform standard convolution
     """
    return ((in_d-1)*s)+k-(2*p)