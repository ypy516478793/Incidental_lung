import pandas as pd
import argparse
import os

def main(args):
    print("Infos path: ", args.infos)
    infos = pd.read_excel(args.infos, sheet_name=[0, 1, 2, 3])
    chest_CT_sheet, biopsy_sheet = infos[1], infos[2]
    MRNs = biopsy_sheet["MRN"]
    unique_MRNs = MRNs.apply(lambda x: '{0:0>9}'.format(x)).unique()

    labeled_sheet = infos[3]
    biopsy_result = labeled_sheet["Results (M-Malignant or B-Benign)"]
    existId = (biopsy_result == "M") | (biopsy_result == "B")
    labeled_sheet = labeled_sheet[existId]
    MRNs = labeled_sheet["MRN"].apply(lambda x: '{0:0>9}'.format(x))
    scan_date = labeled_sheet["Biopsy Date"]
    scan_date = scan_date.apply(lambda x: x.strftime("%Y%m%d"))
    biopsy_result = labeled_sheet["Results (M-Malignant or B-Benign)"]

    identifiers = pd.concat([MRNs, scan_date, biopsy_result], axis=1)
    folder = os.path.dirname(args.infos)
    csvFile = os.path.join(folder, "identifiers_mamta_20210414.csv")
    # csvFile = os.path.join("./", "identifiers.csv")
    identifiers.to_csv(csvFile, index=False, header=["MRN", "date", "Category of Disease"])
    print("shape of saved dataframe: ", identifiers.shape)



    # chest_CT_sheet["MRN"] = chest_CT_sheet["MRN"].apply(lambda x: '{0:0>9}'.format(x))
    # chest_CT_with_biopsy = chest_CT_sheet[chest_CT_sheet['MRN'].isin(unique_MRNs)]
    #
    # scan_date = chest_CT_with_biopsy["Procedure Begin Date"]
    # search_Start = scan_date - pd.DateOffset(days=1)
    # search_End = scan_date + pd.DateOffset(days=1)
    # search_Start = search_Start.apply(lambda x: x.strftime("%Y%m%d"))
    # search_End = search_End.apply(lambda x: x.strftime("%Y%m%d"))
    #
    # # surgery = infos["Date Of Surgery {1340}"]
    # # searchStart = surgery - pd.DateOffset(months=args.range)
    # # surgery = surgery.apply(lambda x: x.strftime("%Y%m%d"))
    # # searchStart = searchStart.apply(lambda x: x.strftime("%Y%m%d"))
    # pid = chest_CT_with_biopsy["MRN"]
    # # pid = pid.apply(lambda x: '{0:0>9}'.format(x))
    # identifiers = pd.concat([pid, search_Start, search_End], axis=1)
    # folder = os.path.dirname(args.infos)
    # csvFile = os.path.join(folder, "identifiers_v2_20210414.csv")
    # # csvFile = os.path.join("./", "identifiers.csv")
    # identifiers.to_csv(csvFile, index=False, header=False)
    # print("shape of saved dataframe: ", identifiers.shape)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # parser.add_argument("--range", type=int, help="range of scan searching before surgery (unit: month)",
    #                     required=False, default=3)
    parser.add_argument("--infos", type=str, help="all patient infos file",
                        required=False, default="/home/cougarnet.uh.edu/pyuan2/Projects/Incidental_Lung/data/Copy of Copy of SLN Data 11.05.2020.CFE.v1.xlsx")
    arguments = parser.parse_args()
    main(arguments)