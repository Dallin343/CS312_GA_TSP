import numpy as np


class Queue:
    def __init__(self):
        self.queue = []
        self.max_size = 0

    def insert(self, state):
        self.queue.append(state)
        self.bubble_up(len(self.queue) - 1)
        self.max_size = max(self.max_size, len(self.queue))

    def delete_min(self):
        priority_state = self.queue[0]
        self.swap(0, len(self.queue) - 1)
        self.queue.pop()
        self.settle(0)
        return priority_state

    def bubble_up(self, child_index):
        if child_index == 0:
            return

        # calculate the index of the parent node in the queue
        # subtraction and bit shift are O(n)
        parent_index = (child_index - 1) >> 1

        child_priority_value = self.queue[child_index].get_priority_value()
        parent_priority_value = self.queue[parent_index].get_priority_value()

        # if child_priority_value is not None and (parent_priority_value is None
        # or parent_priority_value < child_priority_value):
        if parent_priority_value > child_priority_value:
            self.swap(child_index, parent_index)
            self.bubble_up(parent_index)

    def swap(self, index_a, index_b):
        # swap the node IDs in the queue
        temp_node_id = self.queue[index_a]
        self.queue[index_a] = self.queue[index_b]
        self.queue[index_b] = temp_node_id

    def settle(self, parent_index):

        child_1_index = ((parent_index + 1) << 1) - 1
        child_2_index = (parent_index + 1) << 1

        # both child 1 (and therefore child 2) are outside the range of the queue
        if child_1_index >= len(self.queue):
            return

        # only child 1 is inside the range of the queue
        if child_2_index >= len(self.queue):
            priority_child_index = child_1_index

        # both children are inside the range of the queue
        else:
            child_1_priority_value = self.queue[child_1_index].get_priority_value()
            child_2_priority_value = self.queue[child_2_index].get_priority_value()

            if child_1_priority_value <= child_2_priority_value:
                priority_child_index = child_1_index
            else:
                priority_child_index = child_2_index

        # check if the smaller child is smaller than the initial node
        parent_priority_value = self.queue[parent_index].get_priority_value()
        priority_child_priority_value = self.queue[priority_child_index].get_priority_value()

        if parent_priority_value > priority_child_priority_value:
            # swap with the smaller child and recursively settle it
            self.swap(parent_index, priority_child_index)
            self.settle(priority_child_index)

    def is_empty(self):
        return len(self.queue) == 0

    def get_max_size(self):
        return self.max_size

    def get_size(self):
        return len(self.queue)
