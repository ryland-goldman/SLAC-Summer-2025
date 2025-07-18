import sys
n = sys.argv[1]

def filter_pdgid(input_filename, output_filename, target_pdgid=-11):
    with open(input_filename, 'r') as infile, open(output_filename, 'w') as outfile:
        for line in infile:
            # Preserve comments
            if line.startswith("#"):
                outfile.write(line)
                continue

            # Parse line
            parts = line.strip().split()
            if len(parts) < 12:
                continue  # Skip malformed lines

            try:
                pdgid = int(parts[7])  # PDGid is the 8th column (index 7)
            except ValueError:
                continue  # Skip lines where PDGid isn't an integer

            if pdgid == target_pdgid:
                outfile.write(line)

filter_pdgid(f"TargetOut{n}.txt", f"TargetOutFiltered{n}.txt")