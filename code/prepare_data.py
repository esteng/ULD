import re
import os
import subprocess
import argparse 
import textgrid as tg

def convert_dir(src_path,  mfcc_dest, conf):
    for root, dirs, files in os.walk(src_path):
        for filename in files:
            if filename.endswith(".wav") or filename.endswith(".WAV"):
                src_path = os.path.join(root, filename)
                mfcc(src_path, mfcc_dest, conf)
            if filename.endswith(".TextGrid"):
                src_path = os.path.join(root, filename)
                print(filename)
                get_textgrid(tg_path, mfcc_dest)

def mfcc(src, dest, conf):
    src_filename = re.sub("\.WAV", "", os.path.split(src)[-1])
    speaker = os.path.split(os.path.split(src)[0])[-1]
    dest_filename = os.path.join(dest,speaker, src_filename + ".fea")
    print(dest_filename)
    dir_path = os.path.split(dest_filename)[0]
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    subprocess.Popen(['hcopy', '-C', conf, src, dest_filename])

def get_textgrid(file,dest):
    print(file)
    filename = re.sub('\.TextGrid' , '', file)
    print("filename: ", filename)
    t = tg.TextGrid()
    t.read(file)
    phones = [x.mark for x in t.tiers[1]]
    phone_to_int={}
    int_to_phone={}
    for i,phone in enumerate(set(phones)):
        phone_to_int[phone] = i 
        int_to_phone[i] = phone
    with open(filename+".top", "w") as f1:
        ints = [str(phone_to_int[phone]) for phone in phones]
        f1.write(",".join(ints))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("src",  help="source of wav files")
    parser.add_argument("dest",  help="desired location for mfcc feature files")
    parser.add_argument("conf")
    args = parser.parse_args()
    print(args.src, args.dest)
    convert_dir(args.src, args.dest, args.conf)