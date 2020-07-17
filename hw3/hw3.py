#
#   hw3.py
#
#    Ethan Sorkin, Min Hu
#    3/11/2020
#    COMP 131: Artificial Intelligence
#


# Constants
BOARD_SIZE = 9

def solve(board):
    empty = find_empty(board)

    # If there are no empty spots, the board is solved
    if empty == -1:
        return True

    row, col = empty
    for num in range(1, 10):
        if check_num(board, num, row, col):
            board[row][col] = num
            if solve(board):    # Recursive step
                return True
            else:
                board[row][col] = 0     # Reset empty spot after backtracking

    return False

# Finds the next empty spot on the board, if one exists
def find_empty(board):
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            if board[row][col] == 0:
                return (row, col)
    return -1


# Determines whether placing num in board[row][col] is valid
def check_num(board, num, row, col):
    # Check row
    for i in range(BOARD_SIZE):
        if board[row][i] == num:
            return False

    # Check col
    for i in range(BOARD_SIZE):
        if board[i][col] == num:
            return False

    # Check box
    box_r = 3*(row // 3)
    box_c = 3*(col // 3)

    for i in range(box_r, box_r + 3):
        for j in range(box_c, box_c + 3):
            if board[i][j] == num:
                return False

    return True




def main():
    board1 = [
        [6, 0, 8, 7, 0, 2, 1, 0, 0],
        [4, 0, 0, 0, 1, 0, 0, 0, 2],
        [0, 2, 5, 4, 0, 0, 0, 0, 0],
        [7, 0, 1, 0, 8, 0, 4, 0, 5],
        [0, 8, 0, 0, 0, 0, 0, 7, 0],
        [5, 0, 9, 0, 6, 0, 3, 0, 1],
        [0, 0, 0, 0, 0, 6, 7, 5, 0],
        [2, 0, 0, 0, 9, 0, 0, 0, 8],
        [0, 0, 6, 8, 0, 5, 2, 0, 3]
    ]
    print("Easy Puzzle:")
    for x in board1:
        print(x)
    solve(board1)
    print("")
    print("SOLVED:")
    for x in board1:
        print(x)

    print("----------------------------")
    print("")

    board2 = [
        [0, 7, 0, 0, 4, 2, 0, 0, 0],
        [0, 0, 0, 0, 0, 8, 6, 1, 0],
        [3, 9, 0, 0, 0, 0, 0, 0, 7],
        [0, 0, 0, 0, 0, 4, 0, 0, 9],
        [0, 0, 3, 0, 0, 0, 7, 0, 0],
        [5, 0, 0, 1, 0, 0, 0, 0, 0],
        [8, 0, 0, 0, 0, 0, 0, 7, 6],
        [0, 5, 4, 8, 0, 0, 0, 0, 0],
        [0, 0, 0, 6, 1, 0, 0, 5, 0]
    ]

    print("Evil Puzzle:")
    for x in board2:
        print(x)
    solve(board2)
    print("")
    print("SOLVED:")
    for x in board2:
        print(x)



if __name__ == "__main__":
    main(

)
