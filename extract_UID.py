import argparse
import os

def main(args):
    path = args.info
    with open(path, "r") as f:
        lines = f.readlines()
    dateStack = []
    dates, UIs = [], []
    DAstr = " DA "
    UIstr = " UI "
    for line in lines:
        if DAstr in line:
            DA_begin = line.find(DAstr)
            tempDate = line[DA_begin+5: DA_begin+13]
            dateStack.append(tempDate)
        if UIstr in line:
            UI_begin = line.find(UIstr)
            subline = line[UI_begin: ]
            l_begin = subline.find(" [")
            if l_begin > 0:
                r_end = subline.find("] ")
                UI = subline[l_begin+2: r_end].strip("\x00")
                dates.append(dateStack.pop())
                UIs.append(UI)


    print("Dates are: \n" + "\n".join(dates))
    print("UIs are: \n" + "\n".join(UIs))

    folder = os.path.dirname(path)
    csvFile = os.path.join(folder, "uids.csv")
    # csvFile = os.path.join("./", "uids.csv")
    with open(csvFile, "w") as f:
        for i in range(len(UIs)):
            line = dates[i] + "," + UIs[i] + "\n"
            f.write(line)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--info", type=str, help="patient infos file",
                        required=False, default="/Users/yuan_pengyu/Downloads/IncidentalLungCTs_sample/Clinical_info_no_name/01DATAINFO.txt")
    arguments = parser.parse_args()
    main(arguments)