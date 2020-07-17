#
#   hw4.py
#
#    Ethan Sorkin
#    4/5/2020
#    COMP 131: Artificial Intelligence
#

# Constants
MAX_CAP = 250


# Returns optimized items, value, and weight of backpack after being filled
def knapsack(items):
    backpack = [[0]*(MAX_CAP+1) for i in range(len(items)+1)]
    trace = [[False]*(MAX_CAP+1) for i in range(len(items)+1)]

    for n in range(1, len(items)+1):
        for w in range(1, MAX_CAP+1):
            backpack[n][w], trace[n][w] = best_val(n, w, backpack, items)

    final_value = backpack[-1][-1]
    final_items = get_items(items, trace, len(items), MAX_CAP)
    final_items.sort()
    final_weight = 0
    for x in final_items:
        final_weight += items[x - 1][0]

    return final_items, final_value, final_weight


# Fitness function that determines whether or not to include an item in the backpack
def best_val(n, w, backpack, items):
    if w - items[n-1][0] >= 0 and \
       items[n-1][1] + backpack[n-1][w - items[n-1][0]] > backpack[n-1][w]:
        return items[n-1][1] + backpack[n-1][w - items[n-1][0]], True  # included item
    else:
        return backpack[n-1][w], False  # did not include item


# Compiles final item list by backtracking
def get_items(items, trace, n, w):
    if n == 0:
        return []
    elif not trace[n][w]:
        return get_items(items, trace, n-1, w)
    else:
        return [n] + get_items(items, trace, n-1, w - items[n-1][0])


def main():
    items = [(20, 6), (30, 5), (60, 8), (90, 7), (50, 6), (70, 9),
             (30, 4), (30, 5), (70, 4), (20, 9), (20, 2), (60, 1)]

    final_items, final_value, final_weight = knapsack(items)

    print("My backpack will include the folowing items: " + str(final_items))
    print("The total value is " + str(final_value))
    print("The total weight is " + str(final_weight))




if __name__ == "__main__":
    main()
