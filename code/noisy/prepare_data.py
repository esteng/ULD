import re
import os
import subprocess
import argparse 

def convert_dir(src_path, mfcc_dest, tg_dest):
    for root, dirs, files in os.walk(src_path):
        for filename in files:
            if filename.endswith(".wav"):
                src_path = os.path.join(root, filename)
                mfcc(src_path, mfcc_dest)

def mfcc(src, dest):
    src_filename = re.sub("\.wav", "", os.path.split(src)[-1])
    src
    dest_filename = os.path.join(dest, src_filename + ".fea")
    subprocess.Popen(['hcopy', '-C', 'mfcc.conf', src, dest_filename])

def get_textgrid(file):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("src",  help="source of wav files")
    parser.add_argument("dest",  help="desired location for mfcc feature files")
    args = parser.parse_args()
    print(args.src, args.dest)
    convert_dir(args.src, args.dest, None)