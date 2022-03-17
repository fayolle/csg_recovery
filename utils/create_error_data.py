import sys
import os
import create_eval_source


def main(fit_fn, exp_fn, cpp_fn, ps_fn, csv_fn):
    # Create the c++ source code corresponding to the expression in the file
    # exp_fn. 
    create_eval_source.main(fit_fn, exp_fn, cpp_fn, ps_fn)

    # Create the filename for the binary obtained from compiling cpp_fn
    basename = cpp_fn.split('.')[0]
    basename = basename.split('/')[-1]
    prog_name = basename + '_at_input'

    # Generate the command line for compiling the program
    compile_cmd = 'g++ operations.cpp primitives.cpp '
    compile_cmd = compile_cmd + 'sample_expression_input_ps.cpp '
    compile_cmd = compile_cmd + cpp_fn
    compile_cmd = compile_cmd + ' -o ' + prog_name
    print(compile_cmd)
    os.system(compile_cmd)

    # Run
    error_data_cmd = './' + prog_name + ' '
    error_data_cmd = error_data_cmd + ps_fn + ' ' 
    error_data_cmd = error_data_cmd + csv_fn
    print(error_data_cmd)
    os.system(error_data_cmd)


def usage(prog):
    print('Usage: ')
    print(prog + ' ' + 'ex.fit ex.txt ex.cpp ex.xyzn ex.csv')
    print('where:')
    print('\tex.fit: a list of fitted primitives for ex.xyzn')
    print('\tex.txt: a model for ex.xyzn')
    print('\tex.cpp: the model ex.txt translated to c++')
    print('\tex.xyzn: the finite point-set')
    print('\tex.csv: a list of coordinates and values separated by comma')


if __name__ == '__main__':
    num_args = len(sys.argv)
    if num_args != 6:
        usage(sys.argv[0])
        sys.exit(1)

    fit_filename = sys.argv[1]
    exp_filename = sys.argv[2]
    cpp_filename = sys.argv[3]
    ps_filename = sys.argv[4]
    csv_filename = sys.argv[5]

    main(fit_filename, exp_filename, cpp_filename, ps_filename, csv_filename)
