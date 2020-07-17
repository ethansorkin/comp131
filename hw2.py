# Victor Arsenescu
# 2/20/20
# COMP 131 HW 2

class Node(object):
    """Each node has a pancake list and scores"""

    def __init__(self, pancakes, g):
        self.pancakes = pancakes
        self.backwards = g
        self.forwards = self.heuristic(pancakes)
        self.total_score = self.backwards + self.forwards

    def heuristic(self, pancakes):
        h = 0
        for i in range(len(self.pancakes) - 1):
            h += 1 if abs(self.pancakes[i] - self.pancakes[i + 1]) > 1 else 0
        return h


def flip(pancakes, i):
    flippy = pancakes[-i:]
    flippy.reverse()
    end = len(pancakes) - i
    return pancakes[:end] + flippy


def generate_successors(smallest):
    successors = []
    for i in range(1, len(smallest.pancakes)):  # can't flip zero pancakes
        successors.append(Node(flip(smallest.pancakes, i + 1), smallest.backwards + 1))
    return successors


def isGoal(node):
    return True if node.forwards == 0 else False


def A_star(initial_state, generate_successors, isGoal):
    '''
	Arguments:
	- state_space is some list-like structure of length k.
	- successor is a function that returns a new state - aka modifies P.
	- path and heuristics are functions that return the relevant values.
	- isGoal is a function which performs a goal check and returns T/F.
	'''
    expanded, frontier = [], [initial_state]
    while frontier:
        smallest = min(frontier, key=lambda node: node.forwards)
        # print("Frontier: {} ".format([n.pancakes for n in frontier]))
        print("Smallest: {} --> {} / {}".format(smallest.pancakes, smallest.total_score, smallest.forwards))
        frontier.remove(smallest)
        # print("REMOVED SMALLEST: {}".format([n.pancakes for n in frontier]))
        if isGoal(smallest):
            return smallest.pancakes
        expanded.append(smallest)
        successors = generate_successors(smallest)
        # print("Successors: {}".format([s.pancakes for s in successors]))
        # print("Successor Heuristics: {}".format([s.forwards for s in successors]))
        for successor in successors:
            if successor in expanded:
                continue
            if successor in frontier:
                frontier[frontier.index(successor)] = min(successor, frontier[frontier.index(successor)],
                                                          key=lambda node: node.total_score)
            elif successor in expanded:
                expanded[expanded.index(successor)] = min(successor, expanded[expanded.index(successor)],
                                                          key=lambda node: node.total_score)
            else:
                frontier.append(successor)
        print("." * 50)
    return None


if __name__ == '__main__':
    pancakes = [1, 4, 12, 3, 7, 10, 6, 8, 2, 9, 5, 11]
    initial = Node(pancakes, 0)
    A_star(initial, generate_successors, isGoal)
