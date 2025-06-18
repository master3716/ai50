import sys

from crossword import *
from typing import Dict, List
import math

class CrosswordCreator():

    def __init__(self, crossword):
        """
        Create new CSP crossword generate.
        """
        self.crossword: Crossword = crossword
        self.domains: Dict[Variable, List[str]] = {
            var: self.crossword.words.copy()
            for var in self.crossword.variables
        }

    def letter_grid(self, assignment):
        """
        Return 2D array representing a given assignment.
        """
        letters = [
            [None for _ in range(self.crossword.width)]
            for _ in range(self.crossword.height)
        ]
        for variable, word in assignment.items():
            direction = variable.direction
            for k in range(len(word)):
                i = variable.i + (k if direction == Variable.DOWN else 0)
                j = variable.j + (k if direction == Variable.ACROSS else 0)
                letters[i][j] = word[k]
        return letters

    def print(self, assignment):
        """
        Print crossword assignment to the terminal.
        """
        letters = self.letter_grid(assignment)
        for i in range(self.crossword.height):
            for j in range(self.crossword.width):
                if self.crossword.structure[i][j]:
                    print(letters[i][j] or " ", end="")
                else:
                    print("█", end="")
            print()

    def save(self, assignment, filename):
        """
        Save crossword assignment to an image file.
        """
        from PIL import Image, ImageDraw, ImageFont
        cell_size = 100
        cell_border = 2
        interior_size = cell_size - 2 * cell_border
        letters = self.letter_grid(assignment)

        # Create a blank canvas
        img = Image.new(
            "RGBA",
            (self.crossword.width * cell_size,
             self.crossword.height * cell_size),
            "black"
        )
        font = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 80)
        draw = ImageDraw.Draw(img)

        for i in range(self.crossword.height):
            for j in range(self.crossword.width):

                rect = [
                    (j * cell_size + cell_border,
                     i * cell_size + cell_border),
                    ((j + 1) * cell_size - cell_border,
                     (i + 1) * cell_size - cell_border)
                ]
                if self.crossword.structure[i][j]:
                    draw.rectangle(rect, fill="white")
                    if letters[i][j]:
                        _, _, w, h = draw.textbbox((0, 0), letters[i][j], font=font)
                        draw.text(
                            (rect[0][0] + ((interior_size - w) / 2),
                             rect[0][1] + ((interior_size - h) / 2) - 10),
                            letters[i][j], fill="black", font=font
                        )

        img.save(filename)

    def solve(self):
        """
        Enforce node and arc consistency, and then solve the CSP.
        """
        self.enforce_node_consistency()
        self.ac3()
        return self.backtrack(dict())

    def enforce_node_consistency(self):
        """
        Update `self.domains` such that each variable is node-consistent.
        (Remove any values that are inconsistent with a variable's unary
         constraints; in this case, the length of the word.)
        """
        for key, value in self.domains.items():
           copy_value = value.copy()
           for v in copy_value:
               if not key.length == len(v):
                   self.domains[key].remove(v)

    def revise(self, x:Variable, y:Variable):
        """
        Make variable `x` arc consistent with variable `y`.
        To do so, remove values from `self.domains[x]` for which there is no
        possible corresponding value for `y` in `self.domains[y]`.

        Return True if a revision was made to the domain of `x`; return
        False if no revision was made.
        """
        if not y in self.crossword.neighbors(x): return False

        revised = False
        copy_dmX = self.domains[x].copy()
        for d_x in copy_dmX:
            found = False
            for d_y in self.domains[y]:
                (i, j) = self.crossword.overlaps[x, y]
                if d_x[i] == d_y[j]:
                    found = True
            if not found:
                self.domains[x].remove(d_x)
                revised = True

        return revised
                

    def ac3(self, arcs=None):
        """
        Update `self.domains` such that each variable is arc consistent.
        If `arcs` is None, begin with initial list of all arcs in the problem.
        Otherwise, use `arcs` as the initial list of arcs to make consistent.

        Return True if arc consistency is enforced and no domains are empty;
        return False if one or more domains end up empty.
        """
            
        if arcs is None:
            arcs = set([(k, v) for k in self.domains.keys() for v in self.domains.keys() if v != k])
            arcs = list(arcs)
        elif len(arcs) == 0:
            return True
             
        while len(arcs) > 0:
            (x,y) = arcs.pop()
            if self.revise(x, y):
                if len(self.domains[x]) == 0:
                    return False
                for n in self.crossword.neighbors(x):
                    arcs.append((n, x))
        return True
        

    def assignment_complete(self, assignment: Dict[Variable, str]):
        """
        Return True if `assignment` is complete (i.e., assigns a value to each
        crossword variable); return False otherwise.
        """
        
        if len(self.crossword.variables) != len(list(assignment.keys())):
            return False
            
        
        for value in assignment.values():
            if not value:
                return False
            
        return True
    def consistent(self, assignment):
        """
        Return True if `assignment` is consistent (i.e., words fit in crossword
        puzzle without conflicting characters); return False otherwise.
        """
        for key, value in assignment.items():
            if not key.length == len(value):
                    return False
        for key1 in assignment.keys():
            for key2 in assignment.keys():
                if key1 == key2:
                    continue
                if key1 in self.crossword.neighbors(key2):
                    (i, j) = self.crossword.overlaps[key1, key2]
                    if(assignment[key1][i] != assignment[key2][j]):
                        return False
        return True

    def order_domain_values(self, var, assignment):
        """
        Return a list of values in the domain of `var`, in order by
        the number of values they rule out for neighboring variables.
        The first value in the list, for example, should be the one
        that rules out the fewest values among the neighbors of `var`.
        """
        n_dict = {}
        for option in self.domains[var]:
            n = 0
            neighbors = self.crossword.neighbors(var)
            if not neighbors:
                return self.domains[var]
            for neighbor in neighbors:
                (i, j) = self.crossword.overlaps[var, neighbor]
                for neighbor_option in self.domains[neighbor]:
                    if option[i] != neighbor_option[j]:
                        n += 1
            
            n_dict[option] = n
        
        return list(sorted(n_dict, key=n_dict.get))

    def select_unassigned_variable(self, assignment: Dict[Variable, str]):
        """
        Return an unassigned variable not already part of `assignment`.
        Choose the variable with the minimum number of remaining values
        in its domain. If there is a tie, choose the variable with the highest
        degree. If there is a tie, any of the tied variables are acceptable
        return values.
        """
        min_domain_var = Variable(0, 0, Variable.ACROSS, 0)
        min_len = math.inf
        for key, value in self.domains.items():
            if key in assignment.keys():
                continue
            if min_len > len(value):
                min_domain_var = key
                min_len = len(value)
            elif min_len == len(value):
                min_domain_var = self.max_neighbors(min_domain_var, key)
        return min_domain_var
    
    def max_neighbors(self, v1: Variable, v2: Variable) -> Variable:
        neighbors1 = self.crossword.neighbors(v1)
        neighbors2 = self.crossword.neighbors(v2)

        if len(neighbors1) >= len(neighbors2): return v1
        else: return v2
        



    def backtrack(self, assignment: Dict[Variable, str]):
        """
        Using Backtracking Search, take as input a partial assignment for the
        crossword and return a complete assignment if possible to do so.

        `assignment` is a mapping from variables (keys) to words (values).

        If no assignment is possible, return None.
        """
        if self.assignment_complete(assignment=assignment):
            return assignment
        var = self.select_unassigned_variable(assignment=assignment)
        
        for val in self.order_domain_values(var, assignment):
            assignment[var] = val
            if self.consistent(assignment=assignment):
                result = self.backtrack(assignment=assignment)
                if result != None:
                    return result
            del assignment[var]
        return None


def main():

    # Check usage
    if len(sys.argv) not in [3, 4]:
        sys.exit("Usage: python generate.py structure words [output]")

    # Parse command-line arguments
    structure = sys.argv[1]
    words = sys.argv[2]
    output = sys.argv[3] if len(sys.argv) == 4 else None

    # Generate crossword
    crossword = Crossword(structure, words)
    creator = CrosswordCreator(crossword)
    assignment = creator.solve()

    # Print result
    if assignment is None:
        print("No solution.")
    else:
        creator.print(assignment)
        if output:
            creator.save(assignment, output)


if __name__ == "__main__":
    main()
