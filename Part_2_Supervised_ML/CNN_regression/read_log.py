


with open("/processing/ertugrul/read_log/output.log", 'r') as f:
    for line in f:
        if "Epoch" in line:
            print(line)