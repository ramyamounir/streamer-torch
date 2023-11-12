import numpy as np


class MemBuffer():
    r"""
    Buffer class that maintains size of inputs to the temporal encoding model.
    The buffer replaces low prediction error inputs with high prediction error values
    while maintaining the order they were received in.

    :param int buffer_size: The maximum buffer size to maintain
    :param str dist_mode: The distance mode for error calculation (e.g., 'similarity', 'distance')

    """
    def __init__(self, buffer_size=20, dist_mode='similarity'):
        r"""
        init function of the MemBuffer Class
        """


        self.dist_mode = dist_mode
        self.buffer_size = buffer_size
        self.reset_buffer()

    def __len__(self):
        r"""
        Calculates the current length of the buffer

        :returns:
            (*int*): The size of the buffer

        """
        return len(self.inputs)

    def __bool__(self):
        r"""
        Evaluates if the buffer has at least one input

        :returns:
            (*bool*): True is the buffer has one or more inputs
        """
        return len(self.inputs) > 0

    def __repr__(self):
        r"""
        Returns a representation string of the buffer

        :returns:
            (*str*): String describing the number of inputs and error range
        """
        if not self:
            return f'Buffer empty'

        return f'Buffer has {len(self)} values with errors from {min(self.error)} to {max(self.error)}'

    def add_input(self, x, err):
        r"""
        Adds a single input and its corresponding error value to the buffer.
        If the buffer is full, this function takes care of replacing the low prediction error value if the new value is higher.

        :param torch.Tensor x: The input tensor to be added to the buffer
        :param torch.Tensor err: The corresponding error or similarity value.
        """

        self.counter += 1

        # buffer not full
        if len(self.inputs) < self.buffer_size:
            self.inputs.append(x)
            self.error.append(err)
            self.pos.append(len(self.inputs)-1)

        # buffer full
        else:

            if self.dist_mode == 'similarity':
                index = self.error.index(max(self.error))
                if err < self.error[index]:
                    self.replace(index, x, err)

            elif self.dist_mode == 'distance':

                index = self.error.index(min(self.error))
                if err > self.error[index]:
                    self.replace(index, x, err)

    def replace(self, index, x, err):
        r"""
        Replaces an index in the buffer with a new input and its corresponding error

        :param int index: The index of the input to be replaced
        :param torch.Tensor x: The input tensor to be added to the buffer
        :param torch.Tensor err: The corresponding error or similarity value.
        """
        self.inputs[index] = x
        self.error[index] = err

        pos_arr = np.array(self.pos)
        pos_arr[np.array(self.pos) > self.pos[index]] -= 1 
        pos_arr[index] = self.buffer_size -1
        self.pos = pos_arr.tolist().copy()


    def get_inputs(self):
        r"""
        Returns the current inputs in the buffer in the correct order they were received.
        
        :returns:
            (*List(torch.Tensor)*): A list of tensors in the buffer

        """
        if not self:
            return []

        return [self.inputs[self.pos.index(i)] for i in range(len(self.pos))]

    def reset_buffer(self):
        r"""
        Resets the buffer. Can be called after a boundary has been detected
        """
        self.counter = 0
        self.inputs = []
        self.pos = []
        self.error = []



