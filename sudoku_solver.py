def sudoku(f):

    def print_board(grid): # takes in the grid and prints it to the console, replaces the 0s with .
        for n, l in enumerate(grid):
            for m, c in enumerate(l):
                print(str(c).replace("0", "."), end="")
                if m in {2, 5}:
                    print("+", end="")
            print()
            if n in {2, 5}:
                print("+" * 11)


    def check_valid(grid_lines, extracted_numbers_arr):# checks if a certain cell is valid to put a number in
        line = set(extracted_numbers_arr[grid_lines[0]])
        line |= {extracted_numbers_arr[i][grid_lines[1]] for i in range(9)}
        k = grid_lines[0] // 3, grid_lines[1] // 3
        for i in range(3): # adds i and j as needed to go from start of grid to where we need to be
            line |= set(extracted_numbers_arr[k[0] * 3 + i][k[1] * 3:(k[1] + 1) * 3])
        return set(range(1, 10)) - line

    def check_valid_line(ln):
        lines = set(ln) - {0}
        for c in lines:
            if ln.count(c) != 1:
                return True
        return False

    print_board(f)

    extracted_numbers_arr = []# valid numebrs extracted from the image
    grid_lines_extracted = [] # lines that contain zeroes
    for nl, l in enumerate(f):
        try:
            n = list(map(int, l))
        except:
            print("Line #: " + str(nl + 1) + " Contains other symbols")
            return
        if len(n) != 9:
            print("Line #: " + str(nl + 1) + " does not contain digits.")
            return
        grid_lines_extracted += [[nl, i] for i in range(9) if n[i] == 0]
        extracted_numbers_arr.append(n)
    if nl != 8:
        print("there are lines " + str(nl + 1) + " instead of numbers")
        return

    for l in range(9):
        if check_valid_line(extracted_numbers_arr[l]):
            print("Line # " + str(l + 1) + " contains error")
            return
    for c in range(9):
        k = [extracted_numbers_arr[l][c] for l in range(9)]
        if check_valid_line(k):
            print("The Column " + str(c + 1) + " Contains error")
            return
    for l in range(3):
        for c in range(3):
            q = []
            for i in range(3):
                q += extracted_numbers_arr[l * 3 + i][c * 3:(c + 1) * 3]
            if check_valid_line(q):
                print("the cell (" + str(l + 1) + ";" +
                  str(c + 1) + ") contains error")
                return

    points = [[] for i in grid_lines_extracted]
    cr = 0

    while cr < len(grid_lines_extracted):
        points[cr] = check_valid(grid_lines_extracted[cr], extracted_numbers_arr)
        try:
            while not p[cr]:
                extracted_numbers_arr[grid_lines_extracted[cr][0]][grid_lines_extracted[cr][1]] = 0
                cr -= 1
        except:
            print("No Solution Found")
            return
        extracted_numbers_arr[grid_lines_extracted[cr][0]][grid_lines_extracted[cr][1]] = points[cr].pop()
        cr += 1

    print_board(extracted_numbers_arr)
    return(extracted_numbers_arr)
