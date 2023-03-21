import shutil
import glob


def create_compound_file():
    subject_data = glob.glob("data/*.csv")
    subject_data.sort()  # glob lacks reliable ordering, so impose your own if output order matters
    with open('synth_data/user_all_data.csv', 'wb') as outfile:
        for i, csv in enumerate(subject_data):
            with open(csv, 'rb') as infile:
                # If we are not in the first file, skip the header line
                if i != 0:
                    infile.readline()
                if i > 1:
                    outfile.write(b'\n')
                print(f"copying subject {csv}")
                # Block copy rest of file from input to output without parsing
                shutil.copyfileobj(infile, outfile)


if __name__ == "__main__":
    create_compound_file()
