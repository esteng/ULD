
import os
import argparse 
from shutil import copyfile



def relocate_textgrids(tgd_dir, timit_dir):

	for root, dirs, files in os.walk(tgd_dir):

		for filename in files:
			if filename.lower().endswith('.textgrid'):
				filename_parts = filename.split('_')
				tg_dir = filename_parts[-2]
				new_tg_filename = filename_parts[-1]
				tg_location = os.path.join(timit_dir, tg_dir)
				if not os.path.exists(tg_location):
					print("path "+tg_location+" does not exist.")
					continue
				copyfile(os.path.join(root, filename), os.path.join(tg_location, new_tg_filename))




if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("tgd_dir",  help="directory containing textgrid files")
    parser.add_argument("timit_dir",  help="directory containing TIMIT corpus")
    args = parser.parse_args()

    relocate_textgrids(args.tgd_dir, args.timit_dir)
