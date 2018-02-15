import re
import os
import argparse 

def pre_to_top(pre_file):

    with open(pre_file, 'r') as f:
        pre_string = f.read().strip().replace(" ","")

        mappings = {}
        nums = []
        for char in pre_string:
            if char not in mappings:
                mappings[char] = len(mappings)
            nums.append(mappings[char])  

        top_string = ",".join([str(num) for num in nums])

        top_file = pre_file[:-4]+".top"
        with open(top_file, 'w') as g:
            g.write(top_string)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("pre",  help="path to file of pre-tops")
    args = parser.parse_args()
    print(args.pre)
    
    pre_to_top(args.pre)