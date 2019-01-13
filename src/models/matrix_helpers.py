import numpy as np
from scipy.stats import ortho_group
from scipy.linalg import hadamard

def triple_matmul(A,B,C):
    return np.matmul(A, np.matmul(B, C))

# Pad a matrix W to be a certain size
def zero_pad_matrix(W, d0, d1):
    assert d0 >= W.shape[0]
    assert d1 >= W.shape[1]
    # Default is to pad with zeros, which is what we want
    return np.pad(W, [(0, d0 - W.shape[0]), (0, d1 - W.shape[1])], mode='constant')


'''
balanced_init: performs balanced initialization procedure.

Arguments:
W0: the end-to-end matrix we want to intiialize at.
layers: number of layers (i.e. matrix factors)
hidden: size of square matrices in middle matrix factors.
'''
def balanced_init(W0, layers, hidden, randomize=True):
    W = np.copy(W0) # Don't modify W0

    if not isinstance(hidden, list):
        hidden = [hidden] * (layers - 1)

    if layers == 1:
        # return [W[:W0.shape[0],:W0.shape[1]]]
        return W
    U, s, V = np.linalg.svd(W, full_matrices=True)
    s_split = np.power(s, 1.0/layers)
    S = np.diag(s_split)
    W_arr = list()

    if randomize:
        random_orthos = [ortho_group.rvs(hidden[i]) for i in range(layers - 1)]
    else:
        random_orthos = [np.eye(hidden[i]) for i in range(layers - 1)]

    random_orthos = [np.transpose(V)] + random_orthos + [U]
    print(S.shape)
    print([m.shape for m in random_orthos])
    for i in range(layers):
        this_S = zero_pad_matrix(S, random_orthos[i+1].shape[1],
                                 random_orthos[i].shape[0])
        W_arr.append(triple_matmul(
            random_orthos[i+1], this_S, np.transpose(random_orthos[i])))

    return W_arr


def independent_init(layer_sizes, scale):
    W_arr = list()
    b_arr = list()
    for i in range(len(layer_sizes) - 1):
        W_arr.append(np.random.normal(
            scale=scale, size=[layer_sizes[i+1], layer_sizes[i]]))
        b_arr.append(np.random.normal(
            scale=scale, size=[layer_sizes[i+1]]))

    return W_arr, b_arr



'''
test_balanced_init: unit test for balanced_init
'''
def test_balanced_init(balanced=True, in_dim=None, out_dim=None):
    hidden = 10
    layers = 10
    if in_dim is None:
        in_dim = hidden
    if out_dim is None:
        out_dim = hidden

    W = np.random.normal(size=[out_dim, in_dim])
    W = W / np.linalg.norm(W)
    if balanced:
        W_arr = balanced_init(W, layers, hidden)
    else:
        W_arr = unbalanced_init(W, layers, hidden)

    W_rec = W_arr[0]
    for i in range(1, len(W_arr)):
        W_rec = np.matmul(W_arr[i], W_rec)

    print('Original W: ', W)
    print('Reconstructed W: ', W_rec)
    print('Difference: ', np.linalg.norm(W - W_rec))

    print('Balanced matrices:')
    for i in range(0, layers - 1):
        print('Layer %d' % i,
              np.linalg.norm(
                  np.matmul(np.transpose(W_arr[i+1]), W_arr[i+1]) - \
                  np.matmul(W_arr[i], np.transpose(W_arr[i]))))

    print('Matrix norms:')
    for i in range(layers):
        print('Layer %d: Frob-norm = %f, spec-norm = %f' % \
              (i, np.linalg.norm(W_arr[i]), np.linalg.norm(W_arr[i], ord=2)))


# run unit tests
if __name__ == '__main__':
    np.random.seed(0)
    test_balanced_init_2(False)
    # test_balanced_init(in_dim=100, out_dim=1)
    # test_balanced_init(False)
    # test_spec_norm_increase()



