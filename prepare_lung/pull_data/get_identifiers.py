import pandas as pd
import argparse
import os

def main(args):
    print("Infos path: ", args.infos)
    infos = pd.read_excel(args.infos)
    surgery = infos["Date Of Surgery {1340}"]
    searchStart = surgery - pd.DateOffset(months=args.range)
    surgery = surgery.apply(lambda x: x.strftime("%Y%m%d"))
    searchStart = searchStart.apply(lambda x: x.strftime("%Y%m%d"))
    pid = infos["MRN"]
    pid = pid.apply(lambda x: '{0:0>9}'.format(x))
    identifiers = pd.concat([pid, searchStart, surgery], axis=1)
    folder = os.path.dirname(args.infos)
    csvFile = os.path.join(folder, "identifiers.csv")
    # csvFile = os.path.join("./", "identifiers.csv")
    identifiers.to_csv(csvFile, index=False, header=False)
    print()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--range", type=int, help="range of scan searching before surgery (unit: month)",
                        required=False, default=3)
    parser.add_argument("--infos", type=str, help="all patient infos file",
                        required=False, default="/Users/yuan_pengyu/Downloads/IncidentalLungCTs_sample/Clinical_info_no_name/04.Lung Nodule Clinical Data_no name.xlsx")
    arguments = parser.parse_args()
    main(arguments)