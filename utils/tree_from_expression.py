import sys
import collections
import tokenize
import subprocess
import io # io.StringIO


# TODO
# * function names: construct_tree, build_tree are not good; they don't describe 
# what these functions do
# * eventually I could turn this into a library and have the main program 
# access it directly instead of creating intermediate files between this, 
# the gp, the segmentation and use these intermediate files for the communication
#


# parse() needs a list of operations (OPERATIONS) and primitives (PRIMITIVES) 
# used in the expression. The list of operations is fixed so it can be hard-
# coded, but the list of primitives depends on the data being processed.
# Solutions: 1) read it from the .fit file used by the gp or 2) make the gp 
# output a list of primitives and operations

UNARY_OPERATIONS = ['negation']
BINARY_OPERATIONS = ['union', 'intersection', 'subtraction']
OPERATIONS = ['union', 'intersection', 'subtraction', 'negation']

PRIMITIVES = []

def read_primitives_list(primitives_list_filename):
    '''
    Read the list of primitives used in the expression from a file produced by
    the GP.
    '''
    f = open(primitives_list_filename)

    # TODO skip empty lines

    line = f.readline()
    line = line.strip()
    list_prim = line.split(',')

    f.close()
    return list_prim


def read_expression_from_file(filename):
    # Given a filename, create a string with containing the expression read in 
    # the file
    f = open(filename)
    line = f.readline()
    expression = line.strip()
    f.close()
    return expression


def construct_tree(expression):
    # Given the expression as a string, construct a parse tree corresponding to 
    # a pre-order traversal of the tree corresponding to the expression
    src = io.StringIO(expression).readline
    src = tokenize.generate_tokens(src)
    
    # Tree as a list of symbols corresponding to a prefix order traversal of 
    # the tree
    tree = []
    tree = parse(src)

    return tree


def parse(token_src):
    '''
    Given an iterator to a list of tokens, returns a list of operations and 
    primitives.
    '''
    expression = []
    token = next(token_src)

    while token[1]:
        
        token_type = token[0]

        if token[1] in PRIMITIVES or token[1] in OPERATIONS:
            # if it is a primitive or an operation
            label = token[1]
            expression.append(label)

        token = next(token_src)

    return expression


def count_operations(tree_preorder):
    num_operations = 0
    for op in OPERATIONS:
        num_operations = num_operations + tree_preorder.count(op)
    return num_operations


class Node(object):
    def __init__(self, key, label, left=None, right=None):
        # unique id
        self.key = key
        # the node label
        self.label = label
        self.left = left
        self.right = right


# Global variable used to create an unique key for each node in the tree built 
# by build_tree
g_operation_count = 0

def build_tree(prefix):
    '''
    Given a list of tokens corresponding to a pre-order traversal of the tree 
    (stored in a deque), reconstruct the tree
    '''
    global g_operation_count

    label = prefix.popleft()
    if label in PRIMITIVES:
        # a leaf
        g_operation_count = g_operation_count + 1
        return Node(g_operation_count, label)
    elif label in UNARY_OPERATIONS:
        g_operation_count = g_operation_count + 1
        return Node(g_operation_count, label, build_tree(prefix), None)
    else:
        g_operation_count = g_operation_count + 1
        return Node(g_operation_count, label, build_tree(prefix), build_tree(prefix))


def save_tree_to_file(tree, figure_filename):
    tree_string = binary_tree_to_dot_string(tree)
    f = open(figure_filename, 'w')
    f.write(tree_string)
    f.close()


def binary_tree_to_dot_string(tree):
    '''
    Generate a string using the graphviz syntax that describes the tree passed
    as argument.
    '''
    stream = io.StringIO()
    binary_tree_to_dot(tree, stream)
    return stream.getvalue()


def escape(text):
    return '"%s"' % text


# TODO isn't print >>s, "something" deprecated? fix it
def binary_tree_to_dot(tree, stream):
    style = 'fontname="Arial"'

    print("digraph BST {", file=stream)
    # See http://stackoverflow.com/questions/9215803/graphviz-binary-tree-left-and-right-child
    print(' graph [ordering="out"];', file=stream)
    print("    node [%s];" % style,  file=stream) 

    if tree is None:
        # print >>stream
        pass
    elif tree.left is None and tree.right is None:
        print("     %s [label=%s];" % (escape(tree.key), escape(tree.label)), file=stream)
        print("     %s;" % escape(tree.key), file=stream)
    else:
        node_to_dot(tree, stream)

    print("}", file=stream)


# In order to have different nodes with the same label, see:
# http://stackoverflow.com/questions/10579041/graphviz-create-new-node-with-this-same-label
def node_to_dot(node, stream):
    # recursively print the nodes
    if node.left is not None:
        print("    %s [label=%s];" % (escape(node.key), escape(node.label)), file=stream)
        print("    %s [label=%s];" % (escape((node.left).key), escape((node.left).label)), file=stream)
        print("    %s -> %s;" % (escape(node.key), escape((node.left).key)), file=stream)
        node_to_dot(node.left, stream)

    if node.right is not None:
        print("    %s [label=%s];" % (escape(node.key), escape(node.label)), file=stream)
        print("    %s [label=%s];" % (escape((node.right).key), escape((node.right).label)), file=stream)
        print("     %s -> %s;" % (escape(node.key), escape((node.right).key)), file=stream)
        node_to_dot(node.right, stream)


def main():
    try:
        expression_filename = sys.argv[1]
        primitives_list_filename = sys.argv[2]
        figure_filename = sys.argv[3]
    except IndexError:
        sys.exit("Usage: %s expression_file.txt primitives_list.txt figure_file.dot\n" % (sys.argv[0]))

    # Read the expression as a string
    expression = read_expression_from_file(expression_filename)

    # Build the list of primitives
    global PRIMITIVES
    PRIMITIVES = read_primitives_list(primitives_list_filename)
    
    # Transform the expression in a pre-order traversal of the tree
    tree_preorder = construct_tree(expression)
    number_operations = count_operations(tree_preorder)
    print('Number of operations in the tree (internal nodes): ')
    print(number_operations)

    # Construct a tree from the pre-order traversal
    tree = build_tree(collections.deque(tree_preorder))
    save_tree_to_file(tree, figure_filename)


if __name__ == "__main__":
    main()
    
